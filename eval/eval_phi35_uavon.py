#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

from eval_utils import (
    AirsimTrajRecorder,
    calculate_distance_uavon,
    getPoseAfterMakeAction,
    kill_all_env_process,
    print_results,
    process_results,
    to_eularian_angles,
)
from vlm_baseline.action_redirect import build_action_redirect
from vlm_baseline.actions import ACTION_IDS, parse_action_text
from vlm_baseline.depth_avoidance import build_depth_avoidance
from vlm_baseline.memory_context import build_episodic_memory
from vlm_baseline.prompting import build_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT.parent / "UAV-ON_dataset" / "splits" / "uavon_raw_json" / "test.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phi35_uavon"
ACTION_LABELS = [
    "stop",
    "forward 3m",
    "turn left 30 degree",
    "turn right 30 degree",
    "ascend 3m",
    "descend 3m",
]


def patch_transformers_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "get_max_length") or not hasattr(DynamicCache, "get_max_cache_shape"):
        return
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_eval_datasplit(eval_dataset: Path, eval_samples_per_env: Optional[int], output_folder: Path):
    all_eval_info = json.loads(eval_dataset.read_text(encoding="utf-8"))
    temp_eval_save_folder = output_folder / "temp"
    already_env_samples_dict = defaultdict(list)
    already_samples = set()

    if temp_eval_save_folder.exists():
        for json_file in temp_eval_save_folder.glob("*.json"):
            row = json.loads(json_file.read_text(encoding="utf-8"))
            already_env_samples_dict[row["map_name"]].append(row)
            already_samples.add((row["map_name"], str(row["episode_id"])))

    env_groups = defaultdict(list)
    for item in all_eval_info:
        key = (item["map_name"], str(item["episode_id"]))
        if key in already_samples:
            continue
        env_groups[item["map_name"]].append(item)

    if eval_samples_per_env is not None:
        for env_name, eval_info in env_groups.items():
            env_groups[env_name] = eval_info[:eval_samples_per_env]

    total = sum(len(items) for items in env_groups.values())
    print(f"all examples {len(all_eval_info)}, all envs: {len(env_groups)}, remaining examples: {total}")
    return env_groups, already_env_samples_dict


def load_model_and_processor(model_path: str, base_model_path: Optional[str], device: str):
    model_dir = Path(model_path)
    adapter_config = model_dir / "adapter_config.json"
    processor_path = model_path
    torch_dtype = torch.bfloat16

    if adapter_config.is_file():
        if base_model_path is None:
            adapter_meta = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model_path = adapter_meta.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError("LoRA adapter requires --base_model_path or base_model_name_or_path in adapter_config.json")
        processor_path = base_model_path
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("peft is required to evaluate an unmerged LoRA adapter") from exc
        base_model = load_base_model(base_model_path, torch_dtype)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = load_base_model(model_path, torch_dtype)

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, processor


def load_base_model(model_path: str, torch_dtype):
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        return AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)


def move_inputs_to_device(inputs, device: str, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def get_generation_token_ids(processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_ids = []
    for token_id in (
        tokenizer.convert_tokens_to_ids("<|end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None,
        getattr(tokenizer, "eos_token_id", None),
    ):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
            eos_ids.append(token_id)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[-1]

    if not eos_ids:
        eos_token_id = None
    elif len(eos_ids) == 1:
        eos_token_id = eos_ids[0]
    else:
        eos_token_id = eos_ids
    return eos_token_id, pad_token_id


def build_phi35_prompt(
    processor,
    target_description: str,
    depth_context: str | None = None,
    memory_context: str | None = None,
) -> str:
    plain_prompt = build_prompt(
        target_description,
        depth_context=depth_context,
        memory_context=memory_context,
    ).replace("<image>\n", "")
    content = f"<|image_1|>\n{plain_prompt}"
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|user|>\n{content}<|end|>\n<|assistant|>\n"


def processor_call(processor, prompt: str, image: Image.Image):
    images = image if isinstance(image, list) else [image]
    try:
        return processor(prompt, images, return_tensors="pt")
    except Exception:
        try:
            return processor(text=prompt, images=images, return_tensors="pt")
        except Exception:
            return processor(text=prompt, images=image, return_tensors="pt")


def generate_action_text(
    model,
    processor,
    image,
    target_description: str,
    device: str,
    max_new_tokens: int,
    depth_context: str | None = None,
    memory_context: str | None = None,
) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    prompt = build_phi35_prompt(
        processor,
        target_description,
        depth_context=depth_context,
        memory_context=memory_context,
    )
    inputs = processor_call(processor, prompt, image)
    inputs = move_inputs_to_device(inputs, device, torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    eos_token_id, pad_token_id = get_generation_token_ids(processor)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    new_tokens = generated[:, input_len:] if input_len else generated
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return text.strip()


def score_action_candidates(
    model,
    processor,
    image,
    target_description: str,
    device: str,
    normalization: str,
    depth_context: str | None = None,
    memory_context: str | None = None,
):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    prompt = build_phi35_prompt(
        processor,
        target_description,
        depth_context=depth_context,
        memory_context=memory_context,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    candidate_texts = [f"{label}<|end|>\n" for label in ACTION_LABELS]
    full_texts = [prompt + candidate for candidate in candidate_texts]
    inputs = processor_call(processor, full_texts, [image] * len(full_texts))
    inputs = move_inputs_to_device(inputs, device, torch.bfloat16)

    input_ids = inputs["input_ids"]
    labels = torch.full_like(input_ids, -100)
    for row_idx, candidate in enumerate(candidate_texts):
        answer_len = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        labels[row_idx, -answer_len:] = input_ids[row_idx, -answer_len:]
    labels[input_ids < 0] = -100

    with torch.inference_mode():
        outputs = model(**inputs)

    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    score_sum = token_log_probs.sum(dim=1)
    token_count = mask.sum(dim=1).clamp_min(1)
    score_mean = score_sum / token_count

    candidate_scores = {}
    for idx, label in enumerate(ACTION_LABELS):
        candidate_scores[label] = {
            "sum_logprob": float(score_sum[idx].detach().cpu()),
            "mean_logprob": float(score_mean[idx].detach().cpu()),
            "token_count": float(token_count[idx].detach().cpu()),
        }
    key = "sum_logprob" if normalization == "sum" else "mean_logprob"
    best_command = max(ACTION_LABELS, key=lambda label: candidate_scores[label][key])
    return best_command, candidate_scores


def get_action(
    model,
    processor,
    image_list,
    target_description: str,
    device: str,
    max_new_tokens: int,
    inference_mode: str,
    score_normalization: str,
    depth_context: str | None = None,
    memory_context: str | None = None,
):
    candidate_scores = None
    if inference_mode == "score":
        parsed_command, candidate_scores = score_action_candidates(
            model,
            processor,
            image_list[-1],
            target_description,
            device,
            score_normalization,
            depth_context=depth_context,
            memory_context=memory_context,
        )
        action_id = ACTION_IDS[parsed_command]
        raw_text = parsed_command
        matched = True
    else:
        raw_text = generate_action_text(
            model,
            processor,
            image_list[-1],
            target_description,
            device,
            max_new_tokens,
            depth_context=depth_context,
            memory_context=memory_context,
        )
        parsed = parse_action_text(raw_text)
        parsed_command = parsed.command
        action_id = parsed.action_id
        matched = parsed.matched
    print(f"raw action text: {raw_text!r}, parsed: {parsed_command}, action_id: {action_id}, matched: {matched}")
    return action_id, raw_text, parsed_command, matched, candidate_scores


def reset_client_for_episode(airsim_env):
    status = {
        "enabled": True,
        "ok": False,
        "elapsed": None,
        "collision_before": None,
        "collision_after_reset": None,
        "error": "",
    }
    start_time = time.time()
    try:
        try:
            airsim_env._client.cancelLastTask()
        except Exception:
            pass
        status["collision_before"] = airsim_env.get_collision_info()
        airsim_env._client.reset()
        airsim_env._client.enableApiControl(True)
        airsim_env._client.armDisarm(True)
        status["collision_after_reset"] = airsim_env.get_collision_info()
        status["ok"] = True
    except Exception as exc:
        status["error"] = repr(exc)
        try:
            airsim_env._client.enableApiControl(True)
            airsim_env._client.armDisarm(True)
        except Exception:
            pass
    status["elapsed"] = time.time() - start_time
    return status


def set_pose_and_wait(airsim_env, pose, args):
    wait_status = None
    reset_attempts = []
    client_reset_status = {"enabled": False}
    if args.client_reset_per_episode:
        client_reset_status = reset_client_for_episode(airsim_env)
        print("Client reset before episode:", client_reset_status, flush=True)
    for attempt in range(max(1, args.initial_pose_retries)):
        use_zero_kinematics = bool(args.zero_kinematics_reset)
        if use_zero_kinematics:
            reset_status = airsim_env.zero_kinematics_at_pose(
                pose,
                settle_frames=args.initial_pose_settle_frames,
            )
        else:
            airsim_env._set_camera_pose(
                pose[0],
                pose[1],
                pose[2],
                pose[3],
                0,
                0,
                settle_frames=args.initial_pose_settle_frames,
            )
            reset_status = {
                "actual_pose": airsim_env.get_vehicle_pose_xyzyaw(),
                "vehicle_pose": airsim_env.get_vehicle_pose_xyzrpyyaw(),
                "collision_info": airsim_env.get_collision_info(),
            }
        wait_status = airsim_env.wait_until_pose(
            pose,
            position_tol=args.pose_wait_position_tol,
            yaw_tol=args.pose_wait_yaw_tol,
            timeout=args.pose_wait_timeout,
            poll_interval=args.pose_wait_poll_interval,
        )
        if wait_status.get("actual_pose") is None:
            wait_status["actual_pose"] = reset_status.get("actual_pose")
        wait_status["attempt"] = attempt + 1
        wait_status["used_zero_kinematics"] = use_zero_kinematics
        wait_status["client_reset_status"] = client_reset_status
        wait_status["reset_status"] = reset_status
        collision_info = airsim_env.get_collision_info()
        wait_status["baseline_collision_info"] = collision_info
        reached = wait_status.get("reached")
        reached_ok = True if reached is None else bool(reached)
        collision_free = not bool(collision_info.get("has_collided"))
        wait_status["reached_ok"] = reached_ok
        wait_status["collision_free"] = collision_free
        reset_attempts.append(dict(wait_status))
        if reached_ok and collision_free:
            break
        print(
            "Initial pose reset retry needed:",
            {
                "attempt": attempt + 1,
                "reached_ok": reached_ok,
                "collision_free": collision_free,
                "used_zero_kinematics": use_zero_kinematics,
                "collision_info": collision_info,
            },
        )
    if args.render_settle_seconds > 0:
        time.sleep(args.render_settle_seconds)
    if wait_status is None:
        wait_status = {}
    wait_status["client_reset_status"] = client_reset_status
    wait_status["reset_attempts"] = reset_attempts
    return wait_status


def execute_action_and_wait(airsim_env, action_id, target_pose, args):
    if action_id == 0:
        actual_pose = airsim_env.get_vehicle_pose_xyzyaw()
        return {
            "mode": "stop",
            "fly_type": "none",
            "frames": 0,
            "velocity": 0.0,
            "target_pose": list(target_pose),
            "actual_pose": actual_pose,
            "position_error": float(np.linalg.norm(np.array(actual_pose[:3]) - np.array(target_pose[:3]))),
            "yaw_error": abs(float(airsim_env._angle_diff(actual_pose[3], target_pose[3]))),
            "elapsed": 0.0,
        }

    if args.action_execution_mode == "apex_join":
        status = airsim_env.execute_action_to_pose_join(
            action_id,
            target_pose,
            velocity=args.action_velocity,
            move_timeout=args.action_move_timeout,
            rotate_timeout=args.action_rotate_timeout,
            level_after_action=args.level_after_action,
            level_settle_frames=args.level_settle_frames,
        )
    else:
        status = airsim_env.execute_action_to_pose(
            action_id,
            target_pose,
            frames=args.action_sim_frames,
            velocity=args.action_velocity,
        )
    if args.render_settle_seconds > 0:
        time.sleep(args.render_settle_seconds)
    return status


def evaluate(args) -> None:
    patch_transformers_cache_compat()
    output_folder = Path(args.output_foler)
    temp_eval_save_folder = output_folder / "temp"
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_eval_save_folder.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model, processor = load_model_and_processor(args.model_path, args.base_model_path, args.device)
    eval_groups, result_items = load_eval_datasplit(Path(args.eval_dataset), args.eval_samples_per_env, output_folder)
    selected_scenes = set(args.scene_list.split(",")) if args.scene_list else None
    depth_avoidance = build_depth_avoidance(
        args.depth_avoidance,
        grid_size=args.depth_grid_size,
        max_meters=args.depth_max_meters,
        forward_threshold=args.depth_forward_threshold,
        turn_threshold=args.depth_turn_threshold,
        descend_threshold=args.depth_descend_threshold,
        ascend_top_threshold=args.depth_ascend_top_threshold,
    )
    action_redirect = build_action_redirect(
        args.action_redirect,
        search_radius=args.action_redirect_search_radius,
        near_obstacle_threshold=args.action_redirect_near_obstacle_threshold,
    )
    capture_depth = bool(getattr(depth_avoidance, "enabled", False))
    print(f"depth_avoidance={args.depth_avoidance}, capture_depth={capture_depth}")
    print(f"action_redirect={args.action_redirect}, enabled={getattr(action_redirect, 'enabled', False)}")
    print(f"memory_context={args.memory_context}")
    print(f"memory_pose_yaw_unit={args.memory_pose_yaw_unit}")

    for env_name, eval_info in eval_groups.items():
        if args.scene and env_name != args.scene:
            continue
        if selected_scenes and env_name not in selected_scenes:
            continue
        print(f"{env_name}, {len(eval_info)}")
        if not eval_info:
            continue

        airsim_env = AirsimTrajRecorder(
            env_name,
            airsim_port=args.airsim_default_port,
            device_id=args.simulator_gpu,
        )
        cur_env_result = result_items[env_name]
        try:
            pbar = tqdm.tqdm(enumerate(eval_info), total=len(eval_info), desc=f"Evaluating {env_name}")
            for idx, item in pbar:
                start_pose = item["start_pose"]
                object_pose = item["pose"]
                target_description = item["description"]
                start_position = start_pose["start_position"]
                start_quaternionr = start_pose["start_quaternionr"]

                _, _, yaw = to_eularian_angles(
                    start_quaternionr[0],
                    start_quaternionr[1],
                    start_quaternionr[2],
                    start_quaternionr[3],
                )
                new_pose = [start_position[0], start_position[1], start_position[2], yaw]
                pose_wait_status = set_pose_and_wait(airsim_env, new_pose, args)
                if not pose_wait_status.get("reached_ok"):
                    print(f"Initial pose reset failed for sample {idx}: {pose_wait_status}")
                    cur_res = {**item}
                    cur_res["acc"] = 0
                    cur_res["osr"] = 0
                    cur_res["ne"] = calculate_distance_uavon(new_pose[:3], object_pose)
                    cur_res["metric_semantics"] = "official_uavon_stop_success_v1"
                    cur_res["stop_error"] = 1
                    cur_res["oracle_success"] = 0
                    cur_res["collision"] = False
                    cur_res["termination_reason"] = "initial_pose_reset_failed"
                    cur_res["final_distance_within_20m"] = int(cur_res["ne"] <= 20)
                    cur_res["initial_pose_wait_status"] = pose_wait_status
                    cur_res["gen_pose_list"] = []
                    cur_res["raw_action_texts"] = []
                    cur_res["parsed_commands"] = []
                    cur_res["action_ids"] = []
                    cur_res["parse_matched"] = []
                    cur_res["candidate_scores"] = []
                    cur_res["step_records"] = []
                    cur_res["image_paths"] = []
                    cur_env_result.append(cur_res)
                    result_path = temp_eval_save_folder / f"{safe_name(cur_res['map_name'])}-{safe_name(str(cur_res['episode_id']))}.json"
                    result_path.write_text(json.dumps(cur_res, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue

                if pose_wait_status.get("actual_pose") is not None:
                    new_pose = list(pose_wait_status["actual_pose"])
                else:
                    new_pose = list(new_pose)
                episodic_memory = build_episodic_memory(
                    args.memory_context,
                    start_pose=list(new_pose),
                    search_center=list(start_position),
                    history_size=args.memory_history_size,
                    search_radius=args.memory_search_radius,
                    pose_yaw_unit=args.memory_pose_yaw_unit,
                    include_search_bounds=bool(args.memory_include_search_bounds),
                    max_steps=args.eval_max_steps,
                )

                gen_pose_list = [(new_pose[0], new_pose[1], new_pose[2], new_pose[3])]
                image_list = []
                raw_action_texts = []
                parsed_commands = []
                action_ids = []
                parse_matched = []
                candidate_scores_list = []
                step_records = []
                image_paths = []
                cur_osr = 0
                stop_error = 1
                collision = False
                collision_info = None
                termination_reason = "step_limit"
                stop_within_success = False
                image_error = False
                next_pose_wait_status = pose_wait_status
                baseline_collision_info = (
                    pose_wait_status.get("baseline_collision_info")
                    or airsim_env.get_collision_info()
                )
                baseline_collision_timestamp = baseline_collision_info.get("time_stamp")

                step = 0
                while step < args.eval_max_steps:
                    try:
                        pose_wait_before_capture = next_pose_wait_status
                        st_time = time.time()
                        raw_image_dict = airsim_env._capture_images(camera_names=["uav_on_0"], capture_depth=capture_depth)
                        raw_image_entry = raw_image_dict["uav_on_0"]
                        raw_image = raw_image_entry["rgb"]
                        raw_depth = raw_image_entry.get("depth")
                        depth_context_obj = depth_avoidance.build_context(raw_depth)
                        memory_context_obj = episodic_memory.build_context()
                        image_metainfo = raw_image_entry.get("metainfo", {})
                        image_list.append(raw_image)
                        print("capture images:", time.time() - st_time)
                        saved_image_path = None
                        if args.save_step_images and step % args.image_save_stride == 0:
                            image_dir = output_folder / "images" / safe_name(item["map_name"]) / safe_name(str(item["episode_id"]))
                            image_dir.mkdir(parents=True, exist_ok=True)
                            image_path = image_dir / f"step_{step:03d}.{args.image_format}"
                            pil_image = Image.fromarray(raw_image).convert("RGB")
                            if args.image_format.lower() in {"jpg", "jpeg"}:
                                pil_image.save(image_path, quality=args.image_quality)
                            else:
                                pil_image.save(image_path)
                            saved_image_path = str(image_path.relative_to(output_folder))
                            image_paths.append(saved_image_path)

                        st_time = time.time()
                        pose_before = list(new_pose)
                        distance_before = calculate_distance_uavon(pose_before[:3], object_pose)
                        action_id, raw_text, parsed_command, matched, candidate_scores = get_action(
                            model,
                            processor,
                            image_list,
                            target_description,
                            args.device,
                            args.max_new_tokens,
                            args.inference_mode,
                            args.score_normalization,
                            depth_context=depth_context_obj.prompt_text,
                            memory_context=memory_context_obj.prompt_text,
                        )
                        print("action:", action_id, time.time() - st_time)

                        action_redirect_result = action_redirect.redirect(
                            action_id=action_id,
                            current_pose=pose_before,
                            start_position=start_position,
                            depth_grid=depth_context_obj.depth_grid,
                            get_pose_after_action=lambda pose, candidate_action_id: getPoseAfterMakeAction(
                                pose,
                                candidate_action_id,
                                fix_vertical_actions=args.fix_vertical_actions,
                                fix_yaw_actions=args.fix_yaw_actions,
                            ),
                        )
                        if action_redirect_result.changed:
                            print(
                                "action redirected:",
                                action_redirect_result.original_command,
                                "->",
                                action_redirect_result.final_command,
                                action_redirect_result.reason,
                            )
                        original_action_id = action_id
                        original_parsed_command = parsed_command
                        action_id = action_redirect_result.final_action_id
                        parsed_command = action_redirect_result.final_command

                        raw_action_texts.append(raw_text)
                        parsed_commands.append(parsed_command)
                        action_ids.append(action_id)
                        parse_matched.append(matched)
                        candidate_scores_list.append(candidate_scores)

                        target_pose = getPoseAfterMakeAction(
                            new_pose,
                            action_id,
                            fix_vertical_actions=args.fix_vertical_actions,
                            fix_yaw_actions=args.fix_yaw_actions,
                        )
                        next_pose_wait_status = execute_action_and_wait(airsim_env, action_id, target_pose, args)
                        new_pose = list(next_pose_wait_status["actual_pose"])
                        episodic_memory.update(pose_before, list(new_pose), parsed_command)
                        collision_info = (
                            next_pose_wait_status.get("collision_info_after_command")
                            or airsim_env.get_collision_info()
                        )
                        collided = bool(collision_info.get("has_collided"))
                        collision_timestamp = collision_info.get("time_stamp")
                        if baseline_collision_timestamp is not None and collision_timestamp == baseline_collision_timestamp:
                            collided = False
                        gen_pose_list.append((new_pose[0], new_pose[1], new_pose[2], new_pose[3]))
                        distance_after = calculate_distance_uavon(new_pose[:3], object_pose)
                        step_records.append(
                            {
                                "step": step,
                                "pose_before": pose_before,
                                "pose_after": list(new_pose),
                                "target_pose_after": list(target_pose),
                                "distance_before": distance_before,
                                "distance_after": distance_after,
                                "raw_action_text": raw_text,
                                "parsed_command": parsed_command,
                                "original_parsed_command": original_parsed_command,
                                "action_id": action_id,
                                "original_action_id": original_action_id,
                                "parse_matched": matched,
                                "candidate_scores": candidate_scores,
                                "depth_avoidance": depth_context_obj.to_record(),
                                "memory_context": memory_context_obj.to_record(),
                                "action_redirect": action_redirect_result.to_record(),
                                "image_path": saved_image_path,
                                "image_camera_pos": image_metainfo.get("camera_pos"),
                                "image_quat_wb": image_metainfo.get("quat_wb"),
                                "image_fov": image_metainfo.get("fov"),
                                "pose_wait_before_capture": pose_wait_before_capture,
                                "pose_wait_after_action": next_pose_wait_status,
                                "collision_info": collision_info,
                                "collided": collided,
                            }
                        )

                        if collided:
                            collision = True
                            cur_osr = 0
                            termination_reason = "collision"
                            break

                        if distance_after <= 20.0:
                            cur_osr = 1
                        if action_id == 0:
                            stop_error = 0
                            termination_reason = "stop"
                            stop_within_success = distance_after <= 20.0
                            break
                        step += 1
                    except Exception as exc:
                        print(f"Error processing sample {idx}: {exc}")
                        image_error = True
                        break

                if image_error:
                    continue

                dis = calculate_distance_uavon(new_pose[:3], object_pose)
                if termination_reason == "step_limit" and dis <= 20.0:
                    cur_osr = 1
                if collision:
                    cur_acc = 0
                    cur_osr = 0
                    oracle_success = 0
                else:
                    cur_acc = int(termination_reason == "stop" and stop_within_success)
                    oracle_success = int(cur_osr and not cur_acc)
                cur_res = {**item}
                cur_res["acc"] = cur_acc
                cur_res["osr"] = cur_osr
                cur_res["ne"] = dis
                cur_res["metric_semantics"] = "official_uavon_stop_success_v1"
                cur_res["stop_error"] = stop_error
                cur_res["oracle_success"] = oracle_success
                cur_res["collision"] = collision
                cur_res["termination_reason"] = termination_reason
                cur_res["final_distance_within_20m"] = int(dis <= 20)
                cur_res["baseline_collision_info"] = baseline_collision_info
                cur_res["initial_pose_wait_status"] = pose_wait_status
                cur_res["final_collision_info"] = collision_info
                cur_res["gen_pose_list"] = gen_pose_list
                cur_res["raw_action_texts"] = raw_action_texts
                cur_res["parsed_commands"] = parsed_commands
                cur_res["action_ids"] = action_ids
                cur_res["parse_matched"] = parse_matched
                cur_res["candidate_scores"] = candidate_scores_list
                cur_res["step_records"] = step_records
                cur_res["image_paths"] = image_paths
                cur_env_result.append(cur_res)

                sr = sum(cur["acc"] for cur in cur_env_result) / len(cur_env_result)
                osr = sum(cur["osr"] for cur in cur_env_result) / len(cur_env_result)
                pbar.set_postfix_str(f"SR:{sr:.4f}, OSR:{osr:.4f}")

                result_path = temp_eval_save_folder / f"{safe_name(cur_res['map_name'])}-{safe_name(str(cur_res['episode_id']))}.json"
                result_path.write_text(json.dumps(cur_res, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            airsim_env.cleanup()
            del airsim_env
            gc.collect()

    final_results = process_results(result_items)
    (output_folder / "results.json").write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_results(final_results)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Phi-3.5-Vision UAV-ON policy.")
    parser.add_argument("--eval_dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--eval_samples_per_env", type=int, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--output_foler", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval_max_steps", type=int, default=100)
    parser.add_argument("--airsim_default_port", type=int, default=30000)
    parser.add_argument("--simulator_gpu", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--inference_mode", choices=["generate", "score"], default="generate")
    parser.add_argument("--score_normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument(
        "--depth_avoidance",
        choices=["none", "uavon_single_view_prompt"],
        default="uavon_single_view_prompt",
    )
    parser.add_argument("--depth_grid_size", type=int, default=3)
    parser.add_argument("--depth_max_meters", type=float, default=100.0)
    parser.add_argument("--depth_forward_threshold", type=float, default=4.0)
    parser.add_argument("--depth_turn_threshold", type=float, default=1.5)
    parser.add_argument("--depth_descend_threshold", type=float, default=6.0)
    parser.add_argument("--depth_ascend_top_threshold", type=float, default=8.0)
    parser.add_argument("--action_redirect", choices=["none", "uavon_depth", "uavon_bounds_depth"], default="none")
    parser.add_argument("--action_redirect_search_radius", type=float, default=50.0)
    parser.add_argument("--action_redirect_near_obstacle_threshold", type=float, default=2.0)
    parser.add_argument(
        "--memory_context",
        choices=[
            "none",
            "uavon_pose_history",
            "uavon_pose_history_v1",
            "uavon_pose_history_v2",
            "uavon_pose_history_target_directed_v1",
            "uavon_pose_history_target_directed_v1_1",
            "uavon_pose_history_v3",
            "uavon_transition_history",
            "uavon_pose_action_history",
        ],
        default="uavon_pose_history",
    )
    parser.add_argument("--memory_history_size", type=int, default=5)
    parser.add_argument("--memory_search_radius", type=float, default=50.0)
    parser.add_argument("--memory_include_search_bounds", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--memory_pose_yaw_unit",
        choices=["radians", "legacy"],
        default="radians",
        help="Unit of pose yaw passed to memory; use legacy only to reproduce pre-fix runs.",
    )
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--scene_list", type=str, default=None)
    parser.add_argument("--skip_kill_env_process", action="store_true")
    parser.add_argument("--save_step_images", action="store_true")
    parser.add_argument("--image_save_stride", type=int, default=1)
    parser.add_argument("--image_format", type=str, choices=["jpg", "png"], default="jpg")
    parser.add_argument("--image_quality", type=int, default=85)
    parser.add_argument("--fix_vertical_actions", action="store_true")
    parser.add_argument("--fix_yaw_actions", action="store_true")
    parser.add_argument("--pose_wait_timeout", type=float, default=0.0)
    parser.add_argument("--pose_wait_position_tol", type=float, default=0.2)
    parser.add_argument("--pose_wait_yaw_tol", type=float, default=0.05)
    parser.add_argument("--pose_wait_poll_interval", type=float, default=0.05)
    parser.add_argument("--render_settle_seconds", type=float, default=0.0)
    parser.add_argument("--action_execution_mode", choices=["official_frames", "apex_join"], default="official_frames")
    parser.add_argument("--action_sim_frames", type=int, default=150)
    parser.add_argument("--action_velocity", type=float, default=1.0)
    parser.add_argument("--action_move_timeout", type=float, default=5.0)
    parser.add_argument("--action_rotate_timeout", type=float, default=3.0)
    parser.add_argument("--level_after_action", action="store_true")
    parser.add_argument("--level_settle_frames", type=int, default=1)
    parser.add_argument("--initial_pose_retries", type=int, default=3)
    parser.add_argument("--initial_pose_settle_frames", type=int, default=1)
    parser.add_argument("--zero_kinematics_reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--client_reset_per_episode", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_kill_env_process:
        kill_all_env_process()
    evaluate(args)

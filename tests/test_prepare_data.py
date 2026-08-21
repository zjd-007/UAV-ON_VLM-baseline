import json
from pathlib import Path

from scripts.prepare_data import convert


def test_prepare_data_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    source = repo / "UAV-ON_dataset" / "processed" / "nomemory_baseline" / "train_frames.jsonl"
    dataset_root = repo / "UAV-ON_dataset"
    output = tmp_path / "sample.jsonl"
    manifest = tmp_path / "manifest.json"

    stats = convert(source, output, manifest, dataset_root, limit=3)
    assert stats["rows"] == 3
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["conversations"][0]["from"] == "human"
    assert rows[0]["conversations"][0]["value"].startswith("<image>\nWhat action should the UAV take")
    assert rows[0]["conversations"][1]["from"] == "gpt"
    assert Path(rows[0]["images"][0]).is_file()

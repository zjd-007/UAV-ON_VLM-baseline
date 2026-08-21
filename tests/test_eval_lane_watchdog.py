from scripts.eval_lane_watchdog import lane_activity_time, restart_reason


def test_fresh_log_activity_prevents_restart_even_if_result_is_old() -> None:
    assert (
        restart_reason(
            alive=True,
            stale=2.0,
            stale_seconds=300.0,
            restart_if_no_activity=True,
        )
        is None
    )


def test_stale_lane_restarts() -> None:
    assert restart_reason(
        alive=True,
        stale=301.0,
        stale_seconds=300.0,
        restart_if_no_activity=True,
    ) == "stale_301s"


def test_missing_screen_restarts() -> None:
    assert restart_reason(
        alive=False,
        stale=1.0,
        stale_seconds=300.0,
        restart_if_no_activity=True,
    ) == "screen_missing"


def test_no_activity_respects_flag() -> None:
    assert restart_reason(
        alive=True,
        stale=None,
        stale_seconds=300.0,
        restart_if_no_activity=True,
    ) == "no_activity"
    assert (
        restart_reason(
            alive=True,
            stale=None,
            stale_seconds=300.0,
            restart_if_no_activity=False,
        )
        is None
    )


def test_new_lane_directory_provides_startup_activity(tmp_path) -> None:
    lane_dir = tmp_path / "lane0"
    lane_dir.mkdir()

    activity = lane_activity_time(lane_dir, tmp_path / "lane0.log")

    assert activity is not None
    assert activity == lane_dir.stat().st_mtime

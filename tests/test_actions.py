from vlm_baseline.actions import ACTION_IDS, action_name_to_command, parse_action_text


def test_action_name_to_command():
    assert action_name_to_command("Move Forward") == "forward 3m"
    assert action_name_to_command("Turn Left") == "turn left 30 degree"
    assert action_name_to_command("Descend") == "descend 3m"


def test_parse_standard_and_variants():
    variants = {
        "stop": 0,
        "Forward 3 m.": 1,
        "move forward 3 meters": 1,
        "turn left 30 degrees": 2,
        "TURN RIGHT 30 DEGREE": 3,
        "go up 3m": 4,
        "descend 3 metres": 5,
        "go down": 5,
    }
    for text, action_id in variants.items():
        parsed = parse_action_text(text)
        assert parsed.matched
        assert parsed.action_id == action_id


def test_parse_unknown_defaults_to_stop():
    parsed = parse_action_text("look around")
    assert not parsed.matched
    assert parsed.command == "stop"
    assert parsed.action_id == ACTION_IDS["stop"]

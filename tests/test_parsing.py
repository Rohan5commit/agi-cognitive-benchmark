from agi_cognitive_benchmark import parse_plan_answer_response


def test_parse_plan_answer_response_accepts_json_code_block():
    response = """
    ```json
    {
      "applicable_packets": ["P2", "P4"],
      "final_schedule": ["A", "B", "C"],
      "moved_tasks": 1,
      "confidence": 87
    }
    ```
    """

    answer = parse_plan_answer_response(response)

    assert answer is not None
    assert answer.applicable_packets == ["P2", "P4"]
    assert answer.final_schedule == ["A", "B", "C"]
    assert answer.moved_tasks == 1
    assert answer.confidence == 87


def test_parse_plan_answer_response_accepts_yamlish_fields():
    response = """
    applicable_packets: P3, P5
    final_schedule: T1 T2 T3 T4
    moved_tasks: 2 tasks changed
    confidence: 91
    """

    answer = parse_plan_answer_response(response)

    assert answer is not None
    assert answer.applicable_packets == ["P3", "P5"]
    assert answer.final_schedule == ["T1", "T2", "T3", "T4"]
    assert answer.moved_tasks == 2
    assert answer.confidence == 91


def test_parse_plan_answer_response_returns_none_for_irrelevant_text():
    assert parse_plan_answer_response("I cannot solve this right now.") is None

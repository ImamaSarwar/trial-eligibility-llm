from src.metrics import evidence_metrics, clf_metrics
from src.common import parse_output, number_note


def test_evidence_empty_rules():
    m = evidence_metrics([[], [], [1, 2], [1]], [[], [3], [], [1]])
    assert m["macro_all"]["F"] == (1 + 0 + 0 + 1) / 4
    assert m["micro"]["P"] == 1 / 2 and m["micro"]["R"] == 1 / 3


def test_clf():
    m = clf_metrics(["a", "a", "b"], ["a", "b", "b"])
    assert abs(m["accuracy"] - 2 / 3) < 1e-9


def test_parse_and_number():
    p = parse_output('{"label": "included", "evidence_sentences": [2, 0]}', "inclusion")
    assert p["ok"] and p["evidence"] == [0, 2]
    assert not parse_output('{"label": "excluded", "evidence_sentences": []}', "inclusion")["ok"]
    assert not parse_output("garbage", "inclusion")["ok"]
    assert number_note("Hi there. Second one!") == {0: "Hi there.", 1: "Second one!"}
    assert number_note("0. a\n1. b") == {0: "a", 1: "b"}

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from binding_verdict import bind  # noqa: E402


def causal(vb="pass", va="pass"):
    return dict(oracle_minus_none=dict(value=70, ci90=[60, 80]), verdict_b=vb, verdict_a=va,
                learning_failure_supported=True)


def test_clean_result_unchanged():
    out = bind(causal(), dict(oracle_minus_none=dict(ci90=[-1, 1]), b_minus_none=dict(ci90=[-1, 1]),
                              a_minus_gru=dict(ci90=[-1, 1])))
    assert out["verdict_b"] == "pass" and out["learning_failure_supported"] and not out["binding"]["invalid"]


def test_arm_placebo_fail_invalidates_that_arm_only():
    out = bind(causal(), dict(oracle_minus_none=dict(ci90=[-0.06, 1.37]), b_minus_none=dict(ci90=[-1.46, 0.31]),
                              a_minus_gru=dict(ci90=[1.34, 2.98])))
    assert out["verdict_a"] == "invalid" and out["verdict_b"] == "pass" and out["learning_failure_supported"]


def test_oracle_placebo_fail_invalidates_all():
    out = bind(causal(), dict(oracle_minus_none=dict(ci90=[0.5, 1.0]), b_minus_none=dict(ci90=[-1, 1])))
    assert out["verdict_b"] == out["verdict_a"] == "invalid" and not out["learning_failure_supported"]


def test_nested_valid_false_invalidates():
    out = bind(dict(placebo=dict(valid=False), memory=dict(verdict="memory helps", helps=True)))
    assert out["memory"]["verdict"] == "invalid" and out["memory"]["helps"] is False and out["verdict"] == "invalid"


def test_gates_dict_and_placebo_gate():
    assert bind(dict(gates=dict(known_answer=False), verdict="pass"))["verdict"] == "invalid"
    assert bind(dict(placebo_gate="fail", verdict="pass"))["verdict"] == "invalid"
    assert bind(dict(placebo_gate="pass", verdict="pass"))["verdict"] == "pass"

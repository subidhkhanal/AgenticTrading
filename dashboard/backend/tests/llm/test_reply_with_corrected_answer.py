"""A reply holding more than one JSON answer is read by its last complete one.

In 15 real Sonnet 4.6 backtests (week 2026-10-26, 525 decision calls) 14
replies held a full answer, then "Wait - I need to recalculate position size
for AMGN. Cash is only $56." and a corrected answer. The first-``{``-to-
last-``}`` slice spans both, fails with "Extra data", and the hour was lost
(on a billed run, the run aborted). The answer the model ends on is the one
it stands by; the withdrawn one must never be traded.
"""

import json

import dashboard.backend.infrastructure.llm.backtest_harness as harness


def _answer(*orders):
    return json.dumps({"actions": [
        {"symbol": symbol, "action": side, "position_size": shares,
         "confidence": 0.8, "reasoning": "x"}
        for symbol, side, shares in orders
    ]}, indent=2)


FIRST = _answer(("AMGN", "buy", 1), ("WMT", "buy", 1))
CORRECTED = _answer(("WMT", "buy", 1))


def _symbols(decision):
    return [a["symbol"] for a in decision["actions"]]


def test_a_corrected_answer_replaces_the_one_before_it():
    reply = (
        f"```json\n{FIRST}\n```\n\n"
        "Wait - I need to recalculate position size for AMGN. Cash is only $56.\n\n"
        f"```json\n{CORRECTED}\n```"
    )
    assert _symbols(harness.parse_llm_response(reply)) == ["WMT"]


def test_the_same_without_code_fences():
    reply = f"{FIRST}\n\nWait, cash is only $160.69 - cannot afford AMGN.\n\n{CORRECTED}"
    assert _symbols(harness.parse_llm_response(reply)) == ["WMT"]


def test_braces_in_the_reasoning_before_the_answer_are_skipped():
    reply = f"Considering {{RSI, MACD}} confluence first.\n{CORRECTED}"
    assert _symbols(harness.parse_llm_response(reply)) == ["WMT"]


def test_braces_in_the_text_after_the_answer_are_skipped():
    reply = f"{CORRECTED}\nNext hour I will watch {{RSI}} again."
    assert _symbols(harness.parse_llm_response(reply)) == ["WMT"]


def test_a_trailing_object_without_actions_does_not_replace_the_answer():
    reply = f"{CORRECTED}\n\nFor reference: {{\"cash\": 56.0}}"
    assert _symbols(harness.parse_llm_response(reply)) == ["WMT"]


def test_pipeline_step_outputs_without_actions_take_the_last_object():
    reply = '{"signal": "bullish"}\nWait, MACD turned.\n{"signal": "neutral"}'
    assert harness.parse_llm_response(reply) == {"signal": "neutral"}


def test_a_single_answer_parses_as_before():
    assert _symbols(harness.parse_llm_response(f"Here you go:\n{CORRECTED}")) == ["WMT"]


def test_no_complete_object_is_still_nothing():
    assert harness.parse_llm_response('{"actions": [ {"symbol": "WMT", ') is None
    assert harness.parse_llm_response("{RSI, MACD} and nothing else") is None


def test_a_truncated_reply_is_not_read_by_its_complete_inner_orders():
    # The envelope never closes, so its finished order objects are not
    # answers: the reply stays None and keeps its truncation retry.
    reply = (
        '{"actions": [\n'
        '  {"symbol": "WMT", "action": "buy", "position_size": 1, "confidence": 0.8},\n'
        '  {"symbol": "AMGN", "action": "buy", "posi'
    )
    assert harness.parse_llm_response(reply) is None


def test_a_withdrawn_answer_is_never_returned():
    reply = f"{FIRST}\nWait - AMGN is $450 and cash is $56.\n{CORRECTED}"
    assert "AMGN" not in _symbols(harness.parse_llm_response(reply))

"""LLM buys are fitted to the 25% per-stock limit and to the cash left.

Before, each buy was checked against the whole cash balance on its own, then
filled in the model's list order: the first stock listed took what it asked
for, later buys were rejected, and one stock could end up holding most of the
account although the prompt says "never more than 25%".
"""

import json
from datetime import datetime

from dashboard.backend.domain.backtesting.constants import MAX_POSITION_FRACTION
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.tests.backtesting.test_portfolio_manager_move import (
    _FakeClient,
    _FakeResp,
    _FakeUsage,
)

PRICES = {"AAPL": 100.0, "MSFT": 50.0, "JNJ": 270.0}


def _decide(pm, orders, *, equity=1000.0):
    state = {
        "timestamp": datetime(2026, 1, 1),
        "cash": pm.cash,
        "positions": [],
        "positions_value": equity - pm.cash,
        "total_equity": equity,
        "market_signals": {
            symbol: {"price": price, "rsi": 55.0, "macd": 1.0, "macd_signal": 0.5,
                     "sma20": price, "sma50": price, "bb_upper": price, "bb_lower": price}
            for symbol, price in PRICES.items()
        },
    }
    reply = json.dumps({"actions": [
        {"symbol": symbol, "action": "buy", "position_size": shares,
         "confidence": 0.9, "reasoning": "x"}
        for symbol, shares in orders
    ]})
    actions = pm.make_trading_decision_with_llm(
        state, _FakeClient(_FakeResp(reply, _FakeUsage(1, 1)))
    )["actions"]
    return [(a["symbol"], a["shares"]) for a in actions]


def _pm(cash=1000.0, **kwargs):
    pm = PortfolioManager(1000, allowed_symbols=list(PRICES), **kwargs)
    pm.cash = cash
    return pm


def test_the_limit_is_the_one_the_prompt_states():
    assert MAX_POSITION_FRACTION == 0.25


def test_a_buy_within_both_limits_is_unchanged():
    pm = _pm()
    assert _decide(pm, [("AAPL", 2)]) == [("AAPL", 2)]
    assert pm.buys_resized == 0


def test_a_buy_over_the_per_stock_limit_is_cut_to_it():
    pm = _pm()
    assert _decide(pm, [("AAPL", 8)]) == [("AAPL", 2)]  # $800 asked, $250 allowed
    assert pm.buys_resized == 1


def test_a_stock_whose_one_share_is_over_the_limit_is_not_bought():
    pm = _pm()
    assert _decide(pm, [("JNJ", 1)]) == []
    assert pm.buys_resized == 1


def test_shares_already_held_count_toward_the_limit():
    pm = _pm(cash=800.0)
    pm.positions["AAPL"] = 2  # $200 of a $1,000 account
    assert _decide(pm, [("AAPL", 2), ("MSFT", 1)]) == [("MSFT", 1)]


def test_two_orders_for_one_stock_share_one_limit():
    assert _decide(_pm(), [("AAPL", 2), ("AAPL", 2)]) == [("AAPL", 2)]


def test_list_order_no_longer_decides_who_gets_the_cash():
    # $400 of buys, $300 of cash: both shrink by the same fraction.
    forward = _decide(_pm(cash=300.0), [("AAPL", 2), ("MSFT", 4)])
    backward = _decide(_pm(cash=300.0), [("MSFT", 4), ("AAPL", 2)])
    assert sorted(forward) == sorted(backward) == [("AAPL", 1), ("MSFT", 3)]


def test_the_fitted_buys_never_cost_more_than_the_cash():
    pm = _pm(cash=330.0)
    fitted = _decide(pm, [("AAPL", 2), ("MSFT", 5), ("AAPL", 1)])
    assert sum(PRICES[symbol] * shares for symbol, shares in fitted) <= 330.0


def test_lot_sized_markets_are_left_to_the_executor():
    pm = PortfolioManager(100000, allowed_symbols=list(PRICES), lot_size=100)
    assert _decide(pm, [("AAPL", 500)], equity=100000.0) == [("AAPL", 500)]
    assert pm.buys_resized == 0


class _PromptRecorder:
    """Answers with no actions and keeps the prompt the model was sent."""

    def __init__(self):
        self.prompts = []
        self.messages = self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        return _FakeResp(json.dumps({"actions": []}), _FakeUsage(1, 1))


def _snapshot_sent(pm, *, equity=1000.0):
    recorder = _PromptRecorder()
    state = {
        "timestamp": datetime(2026, 1, 1),
        "cash": pm.cash,
        "positions": [],
        "positions_value": equity - pm.cash,
        "total_equity": equity,
        "market_signals": {
            symbol: {"price": price, "rsi": 55.0, "macd": 1.0, "macd_signal": 0.5,
                     "sma20": price, "sma50": price, "bb_upper": price, "bb_lower": price}
            for symbol, price in PRICES.items()
        },
    }
    pm.make_trading_decision_with_llm(state, recorder)
    (prompt,) = recorder.prompts
    return prompt


def _max_buy_shares_in(prompt):
    start = prompt.find("{", prompt.find("MARKET SNAPSHOT:"))
    snapshot, _ = json.JSONDecoder().raw_decode(prompt, start)
    return {
        symbol: snapshot["top_signals"][symbol].get("max_buy_shares")
        for symbol in PRICES
    }


def test_the_prompt_shows_each_stocks_buy_limit():
    prompt = _snapshot_sent(_pm())
    assert _max_buy_shares_in(prompt) == {"AAPL": 2, "MSFT": 5, "JNJ": 0}
    assert "max_buy_shares 0 means you cannot buy" in prompt


def test_the_shown_limit_counts_held_shares_and_cash():
    pm = _pm(cash=120.0)
    pm.positions["AAPL"] = 1  # $100 of the $250 room is used
    limits = _max_buy_shares_in(_snapshot_sent(pm))
    assert limits["AAPL"] == 1  # $150 room left, but $120 cash: one share
    assert limits["MSFT"] == 2  # $250 room, $120 cash


def test_the_shown_limit_is_what_the_fit_allows():
    pm = _pm()
    limits = _max_buy_shares_in(_snapshot_sent(pm))
    for symbol, shares in limits.items():
        if shares:
            assert _decide(_pm(), [(symbol, shares)]) == [(symbol, shares)]


def test_lot_sized_prompts_carry_no_buy_limit():
    pm = PortfolioManager(100000, allowed_symbols=list(PRICES), lot_size=100)
    assert set(_max_buy_shares_in(_snapshot_sent(pm, equity=100000.0)).values()) == {None}

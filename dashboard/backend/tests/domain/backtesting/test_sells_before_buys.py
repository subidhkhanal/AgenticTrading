"""A sell in the same answer pays for a buy, whatever order the model listed them.

Before, a buy was checked against the cash held before the step, so a sell
in the same answer never helped pay for it, and the executor then ran the
actions in listed order, so a buy listed ahead of its sell spent cash first.
"""

import json
from datetime import datetime

import pandas as pd

from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.domain.trading.execution import execute_actions
from dashboard.backend.tests.backtesting.test_portfolio_manager_move import (
    _FakeClient,
    _FakeResp,
    _FakeUsage,
)

PRICES = {"AAPL": 100.0, "MSFT": 50.0}


def _execute(actions, *, cash, positions):
    trades = []
    cash = execute_actions(
        actions=actions,
        market_data={s: pd.Series({"close": p}) for s, p in PRICES.items()},
        timestamp=datetime(2026, 1, 1, 15),
        cash=cash,
        positions=dict(positions),
        entry_prices={s: PRICES[s] for s in positions},
        trades=trades,
    )
    return [(t["symbol"], t["side"], t["shares"]) for t in trades], cash


def test_the_executor_runs_a_sell_before_a_buy_listed_ahead_of_it():
    fills, cash = _execute(
        [{"symbol": "MSFT", "action": "buy", "shares": 10},   # $500, listed first
         {"symbol": "AAPL", "action": "sell", "shares": 5}],  # frees the $500
        cash=0.0, positions={"AAPL": 5},
    )
    assert [side for _, side, _ in fills] == ["SELL", "BUY"]
    assert cash == 0.0


def test_the_executor_keeps_the_listed_order_within_each_side():
    fills, _ = _execute(
        [{"symbol": "MSFT", "action": "buy", "shares": 2},
         {"symbol": "AAPL", "action": "buy", "shares": 1}],
        cash=1000.0, positions={},
    )
    assert [symbol for symbol, _, _ in fills] == ["MSFT", "AAPL"]


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
        {"symbol": symbol, "action": side, "position_size": shares,
         "confidence": 0.9, "reasoning": "x"}
        for symbol, side, shares in orders
    ]})
    actions = pm.make_trading_decision_with_llm(
        state, _FakeClient(_FakeResp(reply, _FakeUsage(1, 1)))
    )["actions"]
    return sorted((a["symbol"], a["action"], a["shares"]) for a in actions)


def _pm(cash):
    pm = PortfolioManager(1000, allowed_symbols=list(PRICES))
    pm.cash = cash
    pm.positions["AAPL"] = 5
    pm.entry_prices["AAPL"] = 100.0
    return pm


def test_a_buy_paid_by_a_sell_listed_after_it_goes_through():
    pm = _pm(cash=100.0)
    decided = _decide(pm, [("MSFT", "buy", 10), ("AAPL", "sell", 5)])
    assert decided == [("AAPL", "sell", 5), ("MSFT", "buy", 10)]
    assert pm.buys_paid_by_sells == 1


def test_the_listed_order_does_not_change_the_decision():
    forward = _decide(_pm(cash=100.0), [("MSFT", "buy", 10), ("AAPL", "sell", 5)])
    backward = _decide(_pm(cash=100.0), [("AAPL", "sell", 5), ("MSFT", "buy", 10)])
    assert forward == backward


def test_a_buy_the_sells_cannot_cover_is_still_skipped():
    pm = _pm(cash=100.0)
    # $100 cash + $200 from selling 2 AAPL < $500 for 10 MSFT.
    assert _decide(pm, [("MSFT", "buy", 10), ("AAPL", "sell", 2)]) == [("AAPL", "sell", 2)]
    assert pm.buys_paid_by_sells == 0


def test_without_a_sell_a_buy_over_the_cash_is_skipped_as_before():
    pm = _pm(cash=100.0)
    assert _decide(pm, [("MSFT", "buy", 10)]) == []
    assert pm.buys_paid_by_sells == 0


def test_a_buy_within_the_cash_does_not_wait_for_sells():
    pm = _pm(cash=1000.0)
    assert _decide(pm, [("MSFT", "buy", 2)]) == [("MSFT", "buy", 2)]
    assert pm.buys_paid_by_sells == 0


def test_a_sell_stays_behind_an_earlier_buy_of_its_own_stock():
    fills, _ = _execute(
        [{"symbol": "AAPL", "action": "buy", "shares": 10},
         {"symbol": "MSFT", "action": "buy", "shares": 2},
         {"symbol": "AAPL", "action": "sell", "shares": 4}],
        cash=10000.0, positions={},
    )
    assert fills == [("AAPL", "BUY", 10), ("MSFT", "BUY", 2), ("AAPL", "SELL", 4)]

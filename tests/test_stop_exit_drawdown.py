from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_engine_module():
    package_name = "trading_engine_sample_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT)]
    sys.modules[package_name] = package

    quality_module = types.ModuleType(f"{package_name}.quality")
    quality_module.generate_data_quality_report = lambda data: {}
    sys.modules[quality_module.__name__] = quality_module

    stats_module = types.ModuleType(f"{package_name}.stats")
    stats_module.compute_performance_stats = lambda **kwargs: {}
    stats_module.infer_periods_per_year = lambda index, default=252: default
    sys.modules[stats_module.__name__] = stats_module

    strategy_module = types.ModuleType(f"{package_name}.strategy")

    class Strategy:
        pass

    strategy_module.Strategy = Strategy
    sys.modules[strategy_module.__name__] = strategy_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.engine",
        REPO_ROOT / "engine.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = _load_engine_module()


class StopStrategy(ENGINE.Strategy):
    execute_on_signal_bar = True

    def __init__(
        self,
        index: pd.Index,
        signals: list[int],
        stops: list[float] | None = None,
        *,
        fills: list[float] | None = None,
        exit_reasons: list[object] | None = None,
        intrabar_events: dict[int, list[dict[str, object]]] | None = None,
    ) -> None:
        self._signals = pd.Series(signals, index=index, dtype="float64")
        if stops is not None:
            self.signal_stop_loss_prices = pd.Series(stops, index=index, dtype="float64")
        if fills is not None:
            self.signal_fill_prices = pd.Series(fills, index=index, dtype="float64")
        if exit_reasons is not None:
            self.signal_exit_reason = pd.Series(exit_reasons, index=index, dtype="object")
        if intrabar_events is not None:
            self.signal_intrabar_events = intrabar_events

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._signals.copy()


class StopExitDrawdownTests(unittest.TestCase):
    def _run_trade(
        self,
        rows: list[dict[str, float]],
        signals: list[int],
        stops: list[float] | None = None,
        *,
        fills: list[float] | None = None,
        exit_reasons: list[object] | None = None,
        intrabar_events: dict[int, list[dict[str, object]]] | None = None,
        max_loss: float | None = None,
    ):
        index = pd.date_range("2026-01-01", periods=len(rows), freq="h")
        data = pd.DataFrame(rows, index=index, dtype="float64")
        strategy = StopStrategy(
            index,
            signals,
            stops,
            fills=fills,
            exit_reasons=exit_reasons,
            intrabar_events=intrabar_events,
        )
        config = ENGINE.BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            max_loss=max_loss,
            close_open_position_on_last_bar=False,
        )
        result = ENGINE.BacktestEngine(config).run(data, strategy)
        self.assertEqual(len(result.trades), 1)
        return result.trades[0]

    def test_long_stop_uses_fill_instead_of_later_bar_low(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 98.0, "close": 98.0},
            ],
            [0, 1, 1],
            [np.nan, 99.0, 99.0],
        )
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.01)

    def test_short_stop_uses_fill_instead_of_later_bar_high(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 102.0, "low": 100.0, "close": 102.0},
            ],
            [0, -1, -1],
            [np.nan, 101.0, 101.0],
        )
        self.assertAlmostEqual(trade.exit_price, 101.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.01)

    def test_gap_through_stop_uses_actual_gap_fill(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 98.0, "high": 99.0, "low": 97.0, "close": 97.5},
            ],
            [0, 1, 1],
            [np.nan, 99.0, 99.0],
        )
        self.assertAlmostEqual(trade.exit_price, 98.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.02)

    def test_prior_worse_excursion_is_preserved(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 97.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 98.0, "close": 98.0},
            ],
            [0, 1, 1, 1],
            [np.nan, np.nan, np.nan, 99.0],
        )
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.03)

    def test_engine_max_loss_exit_is_bounded_to_fill(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 98.0, "close": 98.0},
            ],
            [0, 1, 1],
            max_loss=1.0,
        )
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.01)

    def test_explicit_intrabar_stop_exit_is_bounded_to_fill(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 98.0, "close": 98.0},
            ],
            [0, 1, 1],
            intrabar_events={
                2: [
                    {
                        "event_type": "exit",
                        "side": -1,
                        "price": 99.0,
                        "reason": "Strategy Stop Loss Bullish 1W",
                    }
                ]
            },
        )
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.01)

    def test_signal_stop_exit_is_bounded_to_fill(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 98.0, "close": 98.0},
            ],
            [0, 1, 0],
            fills=[np.nan, np.nan, 99.0],
            exit_reasons=[None, None, "1W Reversal Stop"],
        )
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.01)

    def test_non_stop_exit_keeps_full_exit_bar_range(self) -> None:
        trade = self._run_trade(
            [
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                {"open": 99.0, "high": 100.0, "low": 98.0, "close": 98.0},
            ],
            [0, 1, 0],
        )
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.worst_unrealized_return_pct, -0.02)


if __name__ == "__main__":
    unittest.main()

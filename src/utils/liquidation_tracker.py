
import numpy as np

class LiquidationTracker:
    def __init__(self, initial_balance, margin_rate, liquidation_threshold):
        self.balance = initial_balance
        self.margin_rate = margin_rate
        self.liquidation_threshold = liquidation_threshold
        self.open_trades = {}

    def open_trade(self, trade_id, position_size, entry_price):
        self.open_trades[trade_id] = {
            'position_size': position_size,
            'entry_price': entry_price,
        }

    def close_trade(self, trade_id, exit_price):
        if trade_id in self.open_trades:
            realized_pnl = (exit_price - self.open_trades[trade_id]['entry_price']) * self.open_trades[trade_id]['position_size']
            self.balance += realized_pnl
            del self.open_trades[trade_id]
            return realized_pnl
        return 0

    def calculate_unrealized_pnl(self, current_prices):
        unrealized_pnl = 0
        for trade_id, trade in self.open_trades.items():
            unrealized_pnl += (current_prices[trade_id] - trade['entry_price']) * trade['position_size']
        return unrealized_pnl

    def calculate_margin_level(self, current_prices):
        equity = self.balance + self.calculate_unrealized_pnl(current_prices)
        used_margin = self.calculate_used_margin()
        if used_margin == 0:
            return float('inf')
        return equity / used_margin

    def calculate_used_margin(self):
        used_margin = 0
        for trade in self.open_trades.values():
            used_margin += (trade['position_size'] * trade['entry_price']) * self.margin_rate
        return used_margin

    def is_liquidation_imminent(self, current_prices):
        margin_level = self.calculate_margin_level(current_prices)
        return margin_level < self.liquidation_threshold

import yaml
from .risk import RiskManager

class Executor:
    def __init__(self, cfg_path="configs/trading.yaml"):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.risk = RiskManager(self.cfg["account_balance"], self.cfg["risk_per_trade"])

    def place_order(self, side: str, entry: float, sl: float, tp: float):
        stop_pips = abs(entry - sl) * 10000  # EURUSD pips
        size_lots = self.risk.compute_lots(stop_pips)
        order = dict(symbol=self.cfg["symbol"], side=side, entry=entry, sl=sl, tp=tp, size_lots=size_lots)
        # TODO: connect to broker API here
        print("ORDER:", order)
        return order

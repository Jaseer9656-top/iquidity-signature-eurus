from src.execution.risk import RiskManager
def test_lots():
    rm = RiskManager(balance=100000, risk_per_trade=0.01)
    lots = rm.compute_lots(stop_loss_pips=20)  # $1,000 / (20 pips * $10/pip) = 5.0
    assert lots == 5.0

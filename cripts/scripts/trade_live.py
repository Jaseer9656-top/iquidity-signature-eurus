from src.execution.executor import Executor
ex = Executor()
ex.place_order("buy", entry=1.0950, sl=1.0930, tp=1.0990)  # 20 pip SL -> 1% sizing

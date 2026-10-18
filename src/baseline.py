import pandas as pd
from environment import TradingEnv
from metrics import print_table, run_episode

INITIAL_BALANCE = 100000.0
SEED = 42


def buy_and_hold_action(_obs, _env):
    return 2  # 100% invertido en todo el episodio


def calculate_buy_and_hold():
    df = pd.read_csv("data/test_tesla.csv")
    env = TradingEnv(df=df, initial_balance=INITIAL_BALANCE)
    result = run_episode(env, buy_and_hold_action, seed=SEED)

    print("=== BASELINE: COMPRAR Y MANTENER (BUY & HOLD) ===")
    print(f"Balance Inicial:   ${INITIAL_BALANCE:,.2f}")
    print(f"Patrimonio Final:  ${result['final_net_worth']:,.2f}")
    print("Valuación: mark-to-market al último close, sin venta forzada.")
    print_table([("Comprar y mantener", result)])
    return result


if __name__ == "__main__":
    calculate_buy_and_hold()

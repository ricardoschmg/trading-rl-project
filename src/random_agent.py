import numpy as np
import pandas as pd
from environment import TradingEnv
from metrics import print_table, run_episode

INITIAL_BALANCE = 100000.0
SEED = 42
N_RUNS = 10


def random_action(_obs, env):
    return env.action_space.sample()


def evaluate_random_agent():
    df_test = pd.read_csv("data/test_tesla.csv")
    env = TradingEnv(df=df_test, initial_balance=INITIAL_BALANCE)

    runs = [run_episode(env, random_action, seed=SEED + i) for i in range(N_RUNS)]
    returns = np.array([r["return_pct"] for r in runs])
    drawdowns = np.array([r["max_drawdown_pct"] for r in runs])
    costs = np.array([r["costs"] for r in runs])

    mean_row = {
        "return_pct": float(returns.mean()),
        "max_drawdown_pct": float(drawdowns.mean()),
        "costs": float(costs.mean()),
        "final_net_worth": float(np.mean([r["final_net_worth"] for r in runs])),
    }

    print("=== BASELINE: AGENTE ALEATORIO ===")
    print(f"Corridas: {N_RUNS}  |  semillas {SEED}–{SEED + N_RUNS - 1}")
    print(
        f"Retorno:        {returns.mean():.2f}%  ± {returns.std(ddof=1):.2f}%"
    )
    print(
        f"Máx. drawdown:  {drawdowns.mean():.2f}%  ± {drawdowns.std(ddof=1):.2f}%"
    )
    print(
        f"Costos:         ${costs.mean():,.2f}  ± ${costs.std(ddof=1):,.2f}"
    )
    print_table([(f"Aleatorio (media {N_RUNS})", mean_row)])
    return mean_row


if __name__ == "__main__":
    evaluate_random_agent()

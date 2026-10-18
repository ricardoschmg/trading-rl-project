import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from stable_baselines3 import DQN
from environment import TradingEnv
from metrics import print_table, run_episode
from baseline import buy_and_hold_action
from random_agent import random_action, N_RUNS, SEED as RANDOM_SEED

INITIAL_BALANCE = 100000.0
SEED = 42


def main():
    print("Evaluación held-out (Tesla) — mismas condiciones para todas las estrategias.")
    test_path = os.path.join("data", "test_tesla.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró {test_path}. Ejecuta el EDA primero.")

    df_test = pd.read_csv(test_path)
    env = TradingEnv(df=df_test, initial_balance=INITIAL_BALANCE)

    bh = run_episode(env, buy_and_hold_action, seed=SEED)
    random_runs = [
        run_episode(env, random_action, seed=RANDOM_SEED + i) for i in range(N_RUNS)
    ]
    random_mean = {
        "return_pct": sum(r["return_pct"] for r in random_runs) / N_RUNS,
        "max_drawdown_pct": sum(r["max_drawdown_pct"] for r in random_runs) / N_RUNS,
        "costs": sum(r["costs"] for r in random_runs) / N_RUNS,
    }

    rows = [
        ("Comprar y mantener", bh),
        (f"Aleatorio (media {N_RUNS})", random_mean),
    ]

    model_path = os.path.join("saves", "dqn_tesla_model.zip")
    dqn = None
    if os.path.exists(model_path):
        model = DQN.load(model_path, env=env)

        def dqn_action(obs, _env):
            action, _ = model.predict(obs, deterministic=True)
            return action

        dqn = run_episode(env, dqn_action, seed=SEED)
        rows.insert(0, ("Agente DQN", dqn))
        _save_plots(bh, dqn)
        _log_mlflow(dqn)
    else:
        print(f"Sin modelo en {model_path}. Se reportan solo baselines (Fase 3).")

    print_table(rows)


def _save_plots(bh, dqn):
    os.makedirs("saves", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(bh["equity"], label="Comprar y mantener")
    plt.plot(dqn["equity"], label="Agente DQN")
    plt.title("Curva de capital (held-out)")
    plt.xlabel("Día")
    plt.ylabel("Patrimonio (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("saves", "equity_curve.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(dqn["positions"], drawstyle="steps-post")
    plt.title("Posición del agente DQN (held-out)")
    plt.xlabel("Día")
    plt.ylabel("Fracción invertida")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join("saves", "dqn_positions.png"), dpi=120)
    plt.close()
    print("Figuras guardadas en saves/equity_curve.png y saves/dqn_positions.png")


def _log_mlflow(dqn):
    mlflow.set_experiment("tesla_dqn_trading")
    with mlflow.start_run(run_name="evaluation_test"):
        mlflow.log_metric("eval/final_balance", dqn["final_net_worth"])
        mlflow.log_metric("eval/profit_percentage", dqn["return_pct"])
        mlflow.log_metric("eval/max_drawdown_pct", dqn["max_drawdown_pct"])
        mlflow.log_metric("eval/transaction_costs", dqn["costs"])


if __name__ == "__main__":
    main()

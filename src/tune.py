"""
Búsqueda de hiperparámetros con Optuna.

Antes el objetivo leía el env de entrenamiento al terminar learn()
(último rollout, no la política). Ahora cada trial:

  1. Entrena solo en el primer 80% de train_tesla.csv.
  2. Evalúa greedy en el 20% final de train (validación).
  3. El test no se abre.

El objetivo es el retorno porcentual en validación.
"""
import os
import random

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from stable_baselines3 import DQN

from environment import TradingEnv
from metrics import run_episode

CONFIG_PATH = os.path.join("configs", "tesla_dqn.yaml")
INITIAL_BALANCE = 100000.0
VAL_FRACTION = 0.2


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_train_val(df):
    corte = int(len(df) * (1.0 - VAL_FRACTION))
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


def objective(trial, df_fit, df_val, config):
    tune = config["tune"]
    space = tune["search_space"]
    seed = int(config.get("seed", 42))
    reward_type = config.get("reward", "simple")

    lr = trial.suggest_float(
        "learning_rate",
        float(space["learning_rate"]["low"]),
        float(space["learning_rate"]["high"]),
        log=bool(space["learning_rate"]["log"]),
    )
    batch_size = trial.suggest_categorical("batch_size", space["batch_size"]["choices"])

    random.seed(seed)
    np.random.seed(seed)

    env = TradingEnv(
        df=df_fit,
        initial_balance=INITIAL_BALANCE,
        reward_type=reward_type,
    )
    env.reset(seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        batch_size=batch_size,
        buffer_size=config["hyperparams"].get("buffer_size", 50_000),
        learning_starts=config["hyperparams"].get("learning_starts", 1000),
        gamma=config["hyperparams"].get("gamma", 0.99),
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=int(tune["timesteps"]))

    eval_env = TradingEnv(
        df=df_val,
        initial_balance=INITIAL_BALANCE,
        reward_type=reward_type,
    )

    def dqn_action(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    result = run_episode(eval_env, dqn_action, seed=seed)

    with mlflow.start_run(nested=True):
        mlflow.log_params({"learning_rate": lr, "batch_size": batch_size})
        mlflow.log_metric("val/return_pct", result["return_pct"])
        mlflow.log_metric("val/max_drawdown_pct", result["max_drawdown_pct"])
        mlflow.log_metric("val/costs", result["costs"])

    trial.set_user_attr("max_drawdown_pct", result["max_drawdown_pct"])
    trial.set_user_attr("costs", result["costs"])
    print(
        f"Trial {trial.number}: lr={lr:.5g} batch={batch_size} "
        f"val_return={result['return_pct']:.2f}% "
        f"mdd={result['max_drawdown_pct']:.2f}%"
    )
    return result["return_pct"]


def write_best_params(config, best_params):
    config["hyperparams"]["learning_rate"] = float(best_params["learning_rate"])
    config["hyperparams"]["batch_size"] = int(best_params["batch_size"])
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"YAML actualizado: {CONFIG_PATH}")


if __name__ == "__main__":
    config = load_config()
    df_all_train = pd.read_csv("data/train_tesla.csv")
    df_fit, df_val = split_train_val(df_all_train)
    print(
        f"Optuna | fit={len(df_fit)} días | val={len(df_val)} días | "
        f"test NO se usa | trials={config['tune']['trials']} | "
        f"steps/trial={config['tune']['timesteps']}"
    )

    mlflow.set_experiment("tesla_dqn_tuning")
    with mlflow.start_run(run_name="Optuna_Optimization"):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, df_fit, df_val, config),
            n_trials=int(config["tune"]["trials"]),
        )

        print("\n=== MEJORES HIPERPARÁMETROS (según validación) ===")
        print(yaml.dump(study.best_params, sort_keys=False))
        print(f"Mejor retorno en val: {study.best_value:.2f}%")
        write_best_params(config, study.best_params)

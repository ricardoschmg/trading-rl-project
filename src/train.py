import os
import random
import yaml
import numpy as np
import pandas as pd
import mlflow
from stable_baselines3 import DQN
from environment import TradingEnv
from metrics import position_mix, print_table, run_episode

INITIAL_BALANCE = 100000.0


def main():
    print("Iniciando entrenamiento del agente DQN para Trading (Tesla) con MLflow...")

    mlflow.set_experiment("tesla_dqn_trading")

    with mlflow.start_run():
        config_path = os.path.join("configs", "tesla_dqn.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        hp = config.get("hyperparams", {})
        timesteps = config.get("timesteps", 30_000)
        seed = int(config.get("seed", 42))
        reward_type = config.get("reward", "simple")
        random.seed(seed)
        np.random.seed(seed)

        mlflow.log_params(hp)
        mlflow.log_param("timesteps", timesteps)
        mlflow.log_param("seed", seed)
        mlflow.log_param("reward", reward_type)
        mlflow.log_param("env", "Tesla-Daily")
        mlflow.log_param("algorithm", "DQN")

        train_path = os.path.join("data", "train_tesla.csv")
        df_train = pd.read_csv(train_path)

        env = TradingEnv(
            df=df_train,
            initial_balance=INITIAL_BALANCE,
            transaction_fee=0.001,
            reward_type=reward_type,
        )
        env.reset(seed=seed)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=hp.get("learning_rate", 1e-4),
            buffer_size=hp.get("buffer_size", 50_000),
            batch_size=hp.get("batch_size", 32),
            learning_starts=hp.get("learning_starts", 1000),
            gamma=hp.get("gamma", 0.99),
            verbose=1,
            seed=seed,
        )

        print(f"Entrenando {timesteps} timesteps | reward={reward_type} | seed={seed}")
        model.learn(total_timesteps=timesteps)

        os.makedirs("saves", exist_ok=True)
        save_path = os.path.join("saves", f"dqn_tesla_{reward_type}.zip")
        model.save(save_path)
        model.save(os.path.join("saves", "dqn_tesla_model.zip"))
        mlflow.log_artifact(save_path)

        eval_env = TradingEnv(df=df_train, initial_balance=INITIAL_BALANCE, reward_type=reward_type)

        def dqn_action(obs, _env):
            action, _ = model.predict(obs, deterministic=True)
            return action

        result = run_episode(eval_env, dqn_action, seed=seed)
        mix = position_mix(result["positions"])
        print(f"\n=== Evaluación GREEDY sobre TRAIN (no es test) | reward={reward_type} ===")
        print_table([(f"DQN-{reward_type}", result)])
        print(
            f"Tiempo en 0% / 50% / 100%:  "
            f"{mix['cash_0']:.1f}% / {mix['half_50']:.1f}% / {mix['full_100']:.1f}%"
        )
        print(f"Modelo guardado en: {save_path}")


if __name__ == "__main__":
    main()

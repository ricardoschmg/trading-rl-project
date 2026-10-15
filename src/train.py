import os
import yaml
import pandas as pd
from stable_baselines3 import DQN
from environment import TradingEnv

def main():
    print("Iniciando entrenamiento del agente DQN para Trading (Tesla)...")
    
    # 1. CARGAR EL ARCHIVO YAML
    config_path = os.path.join("configs", "tesla_dqn.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    hp = config.get("hyperparams", {})
    timesteps = config.get("timesteps", 30_000)
    
    # 2. Cargar los datos de entrenamiento
    train_path = os.path.join("data", "train_tesla.csv")
    df_train = pd.read_csv(train_path)
    
    # 3. Instanciar el entorno
    env = TradingEnv(df=df_train, initial_balance=10000.0, transaction_fee=0.001)
    
    # 4. Configurar el modelo usando 'hp' (¡Ahora sí lo reconoce!)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hp.get("learning_rate", 1e-4),
        buffer_size=hp.get("buffer_size", 50_000),
        batch_size=hp.get("batch_size", 32),
        learning_starts=hp.get("learning_starts", 1000),
        gamma=hp.get("gamma", 0.99),
        verbose=1,
        seed=42
    )
    
    print(f"Entrenando el modelo durante {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    os.makedirs("saves", exist_ok=True)
    save_path = os.path.join("saves", "dqn_tesla_model.zip")
    model.save(save_path)
    print(f"¡Entrenamiento completado! Modelo guardado en: {save_path}")

if __name__ == "__main__":
    main()
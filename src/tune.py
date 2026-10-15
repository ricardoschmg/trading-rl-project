import optuna
import yaml
import pandas as pd
import mlflow
from stable_baselines3 import DQN
from environment import TradingEnv

# Cargar configuración
with open("configs/tesla_dqn.yaml", "r") as f:
    config = yaml.safe_load(f)

df_train = pd.read_csv("data/train_tesla.csv")
tune_config = config["tune"]

def objective(trial):
    # Sugerir hiperparámetros basados en el espacio de búsqueda del YAML
    lr = trial.suggest_float("learning_rate", 
                             tune_config["search_space"]["learning_rate"]["low"], 
                             tune_config["search_space"]["learning_rate"]["high"], 
                             log=tune_config["search_space"]["learning_rate"]["log"])
    
    batch_size = trial.suggest_categorical("batch_size", 
                                           tune_config["search_space"]["batch_size"]["choices"])
    
    # Crear entorno para el trial
    env = TradingEnv(df=df_train)
    
    # Iniciar tracking anidado en MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params({"learning_rate": lr, "batch_size": batch_size})
        
        # Iniciar modelo fresco
        model = DQN("MlpPolicy", env, learning_rate=lr, batch_size=batch_size, verbose=0)
        
        # Entrenar por los timesteps definidos en el YAML
        model.learn(total_timesteps=tune_config["timesteps"])
        
        # La recompensa para Optuna será el balance final del entorno
        final_balance = env.balance + (env.shares_held * float(env.df.iloc[-1]['close']))
        mlflow.log_metric("final_balance", final_balance)
        
    return final_balance

# Ejecutar el estudio de Optuna
if __name__ == "__main__":
    mlflow.set_experiment("tesla_dqn_tuning")
    
    with mlflow.start_run(run_name="Optuna_Optimization"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=tune_config["trials"])
        
        print("\n=== MEJORES HIPERPARÁMETROS ===")
        print(yaml.dump(study.best_params))
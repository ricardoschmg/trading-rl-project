import os
import pandas as pd
import mlflow
from stable_baselines3 import DQN
from environment import TradingEnv

def main():
    print("Iniciando evaluación del agente DQN con datos de prueba (Tesla) y MLflow...")
    
    # 1. Configurar el experimento en MLflow
    mlflow.set_experiment("tesla_dqn_trading")
    
    with mlflow.start_run(run_name="evaluation_test"):
        # 2. Cargar el dataset de prueba
        test_path = os.path.join("data", "test_tesla.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"No se encontró el archivo de prueba en {test_path}. Ejecuta el EDA primero.")
        
        df_test = pd.read_csv(test_path)
        
        # 3. Instanciar el entorno de Gymnasium
        env = TradingEnv(df=df_test, initial_balance=10000.0, transaction_fee=0.001)
        
        # 4. Cargar el modelo entrenado
        model_path = os.path.join("saves", "dqn_tesla_model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo entrenado en {model_path}. Ejecuta train.py primero.")
        
        model = DQN.load(model_path, env=env)
        
        # 5. Ejecutar la evaluación de forma determinista
        obs, info = env.reset()
        done = False
        truncated = False
        
        total_reward = 0.0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        # 6. Calcular métricas financieras de test
        final_balance = info.get("net_worth", 10000.0)
        initial_balance = 10000.0
        profit_percentage = ((final_balance - initial_balance) / initial_balance) * 100

        # Registrar métricas en MLflow
        mlflow.log_metric("eval/final_balance", final_balance)
        mlflow.log_metric("eval/profit_percentage", profit_percentage)
        mlflow.log_metric("eval/total_reward", total_reward)

        print("\n================ RESULTADOS DE LA EVALUACIÓN ================")
        print(f"Balance Inicial:  ${initial_balance:,.2f}")
        print(f"Balance Final:    ${final_balance:,.2f}")
        print(f"Retorno Total:    {profit_percentage:.2f}%")
        print(f"Recompensa Acum.: {total_reward:.4f}")
        print("=============================================================")
        print("¡Resultados de evaluación registrados exitosamente en MLflow!")

if __name__ == "__main__":
    main()
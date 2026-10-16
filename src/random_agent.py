import pandas as pd
from environment import TradingEnv

def evaluate_random_agent():
    # Cargar los datos de prueba
    df_test = pd.read_csv("data/test_tesla.csv")
    
    # Inicializar el entorno
    env = TradingEnv(df=df_test)
    observation, info = env.reset()
    
    terminated = False
    truncated = False
    
    # Ejecutar pasos aleatorios hasta que termine el episodio
    while not (terminated or truncated):
        # Tomar una acción completamente al azar (0, 1 o 2)
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        
    # Extraer métricas finales
    initial_balance = env.initial_balance
    final_net_worth = info["net_worth"]
    return_pct = ((final_net_worth - initial_balance) / initial_balance) * 100
    
    print("=== BASELINE: AGENTE ALEATORIO ===")
    print(f"Balance Inicial:  ${initial_balance:,.2f}")
    print(f"Patrimonio Final: ${final_net_worth:,.2f}")
    print(f"Retorno Total:    {return_pct:.2f}%")

if __name__ == "__main__":
    evaluate_random_agent()
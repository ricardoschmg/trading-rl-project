import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Entorno de trading personalizado compatible con Gymnasium para DQN.
    Utiliza datos históricos y un espacio de acciones discreto.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee: float = 0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Espacio de acciones discreto:
        # 0: 0% invertido (Efectivo)
        # 1: 50% invertido
        # 2: 100% invertido
        self.action_space = spaces.Discrete(3)
        
        # Espacio de observaciones: 
        # [precio de cierre normalizado o actual, balance actual, posición anterior]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.current_position = 0.0  # 0.0, 0.5 o 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.current_position = 0.0
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Mapear la acción discreta a un porcentaje de inversión
        # 0 -> 0%, 1 -> 50%, 2 -> 100%
        target_position = action * 0.5 
        
        current_price = float(self.df.loc[self.current_step, 'close'])
        
        # Calcular el valor objetivo en acciones según el balance y patrimonio actual
        self.net_worth = self.balance + (self.shares_held * current_price)
        target_investment = self.net_worth * target_position
        current_investment = self.shares_held * current_price
        
        # Calcular la diferencia para comprar o vender con costos de transacción
        trade_value = target_investment - current_investment
        
        if trade_value > 0:  # Comprar
            cost = trade_value * (1 + self.transaction_fee)
            if self.balance >= cost:
                self.balance -= cost
                self.shares_held += trade_value / current_price
            else:
                # Si no hay suficiente efectivo, ajustamos al máximo posible
                max_trade = self.balance / (1 + self.transaction_fee)
                self.balance -= max_trade * (1 + self.transaction_fee)
                self.shares_held += max_trade / current_price
                
        elif trade_value < 0:  # Vender
            proceeds = abs(trade_value) * (1 - self.transaction_fee)
            self.balance += proceeds
            self.shares_held -= abs(trade_value) / current_price
            
        self.current_position = target_position
        
        # Avanzar al siguiente paso temporal (evitando look-ahead bias)
        self.current_step += 1
        
        # Verificar si el episodio ha terminado
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Calcular el nuevo patrimonio y la recompensa (cambio porcentual del patrimonio)
        new_price = float(self.df.loc[self.current_step, 'close']) if not terminated else current_price
        self.net_worth = self.balance + (self.shares_held * new_price)
        
        reward = (self.net_worth - self.previous_net_worth) / self.previous_net_worth
        self.previous_net_worth = self.net_worth
        
        observation = self._get_observation()
        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held
        }
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        current_price = float(self.df.loc[self.current_step, 'close'])
        
        # Calcular el retorno porcentual diario del precio (si es posible)
        if self.current_step > 0:
            prev_price = float(self.df.loc[self.current_step - 1, 'close'])
            price_change = (current_price - prev_price) / prev_price
        else:
            price_change = 0.0
            
        # Normalizar el balance dividiéndolo por el balance inicial
        normalized_balance = float(self.balance) / self.initial_balance
        
        # Vector de estado normalizado
        return np.array([
            price_change,
            normalized_balance,
            float(self.current_position)
        ], dtype=np.float32)

    def render(self):
        print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f}")
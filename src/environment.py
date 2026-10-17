import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Entorno de trading compatible con Gymnasium.

    Reloj (lookahead-safe):
      - En el día t el agente ve el retorno de ayer (P_{t-1} vs P_{t-2}).
      - Ejecuta al close de hoy P_t.
      - El PnL de la recompensa usa el close de mañana P_{t+1}.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_fee: float = 0.001,
        reward_type: str = "simple",
    ):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        if reward_type not in {"simple", "log"}:
            raise ValueError("reward_type debe ser 'simple' o 'log'")
        self.reward_type = reward_type

        # 0: 0% invertido, 1: 50%, 2: 100%
        self.action_space = spaces.Discrete(3)

        # [retorno de ayer, efectivo / capital inicial, posición real]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.current_position = 0.0
        self.peak_net_worth = self.initial_balance
        self.max_drawdown = 0.0
        self.last_trade_cost = 0.0
        self.total_trade_cost = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.current_position = 0.0
        self.peak_net_worth = self.initial_balance
        self.max_drawdown = 0.0
        self.last_trade_cost = 0.0
        self.total_trade_cost = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        action = int(np.asarray(action).reshape(-1)[0])
        target_position = action * 0.5

        current_price = float(self.df.loc[self.current_step, "close"])
        self.net_worth = self.balance + (self.shares_held * current_price)
        target_investment = self.net_worth * target_position
        current_investment = self.shares_held * current_price
        trade_value = target_investment - current_investment
        trade_cost = 0.0

        if trade_value > 0:
            cost = trade_value * (1 + self.transaction_fee)
            if self.balance >= cost:
                executed = trade_value
            else:
                executed = self.balance / (1 + self.transaction_fee)
            self.balance -= executed * (1 + self.transaction_fee)
            self.shares_held += executed / current_price
            trade_cost = executed * self.transaction_fee
        elif trade_value < 0:
            executed = abs(trade_value)
            self.balance += executed * (1 - self.transaction_fee)
            self.shares_held -= executed / current_price
            trade_cost = executed * self.transaction_fee

        self.last_trade_cost = trade_cost
        self.total_trade_cost += trade_cost

        # Posición realmente alcanzada (puede ser menor que el objetivo si no hay efectivo)
        self.net_worth = self.balance + (self.shares_held * current_price)
        if self.net_worth > 0:
            self.current_position = (self.shares_held * current_price) / self.net_worth
        else:
            self.current_position = 0.0

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        new_price = float(self.df.loc[self.current_step, "close"]) if not terminated else current_price
        self.net_worth = self.balance + (self.shares_held * new_price)

        ratio = self.net_worth / self.previous_net_worth
        if self.reward_type == "log":
            reward = float(np.log(max(ratio, 1e-12)))
        else:
            reward = float(ratio - 1.0)
        self.previous_net_worth = self.net_worth

        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        drawdown = (self.net_worth - self.peak_net_worth) / self.peak_net_worth
        self.max_drawdown = min(self.max_drawdown, drawdown)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Retorno de ayer: conocido antes de operar al close de hoy.
        if self.current_step >= 2:
            prev_price = float(self.df.loc[self.current_step - 1, "close"])
            prev2_price = float(self.df.loc[self.current_step - 2, "close"])
            price_change = (prev_price - prev2_price) / prev2_price
        else:
            price_change = 0.0

        price_change = float(np.clip(price_change, -1.0, 1.0))
        normalized_balance = float(self.balance) / self.initial_balance

        return np.array(
            [price_change, normalized_balance, float(self.current_position)],
            dtype=np.float32,
        )

    def _get_info(self):
        drawdown = (self.net_worth - self.peak_net_worth) / self.peak_net_worth if self.peak_net_worth else 0.0
        return {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "position": self.current_position,
            "shares_held": self.shares_held,
            "trade_cost": self.last_trade_cost,
            "total_trade_cost": self.total_trade_cost,
            "drawdown": drawdown,
            "max_drawdown": self.max_drawdown,
        }

    def render(self):
        print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f}")

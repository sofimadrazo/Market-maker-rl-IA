import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class MarketMakerEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    # Distance multipliers (x tick_size) from mid for each action level
    LEVEL_MULTIPLIERS = [0.5, 1.0, 2.0, 3.0, 5.0]

    def __init__(
        self,
        data_path=None,
        start_idx=0,
        end_idx=None,
        price_window_size=10,
        max_inventory=10,
        tick_size=0.01,
        spread_fraction=0.002,
        gamma=0.1,
        kappa=0.001,
        lambda_=30.0,
        liquidation_penalty=0.05,
        n_bid_levels=5,
        n_ask_levels=5,
        episode_length=1000,
        S0=100.0,
        mu=0.0,
        sigma=0.2,
        dt=1 / 252,
        render_mode=None,
    ):
        super().__init__()

        self.data_path = data_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.price_window_size = price_window_size
        self.max_inventory = max_inventory
        self.tick_size = tick_size
        self.spread_fraction = spread_fraction
        self.gamma = gamma
        self.kappa = kappa
        self.lambda_ = lambda_
        self.liquidation_penalty = liquidation_penalty
        self.n_bid_levels = n_bid_levels
        self.n_ask_levels = n_ask_levels
        self.episode_length = episode_length
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.render_mode = render_mode

        # Pre-load CSV prices once if a path is given
        self._csv_prices = None
        if data_path is not None:
            self._csv_prices = self._load_csv_data()

        obs_size = 7 + price_window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([n_bid_levels, n_ask_levels])

        # Episode state — initialised in reset()
        self._episode_prices = None
        self._price_history = None
        self._current_step = 0
        self._max_steps = episode_length
        self._inventory = 0
        self._realized_pnl = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._current_step = 0
        self._inventory = 0
        self._realized_pnl = 0.0

        if self._csv_prices is not None:
            self._episode_prices = self._csv_prices
            self._max_steps = len(self._episode_prices)
        else:
            self._episode_prices = self._generate_gbm_path()
            self._max_steps = len(self._episode_prices)

        self._price_history = np.full(
            self.price_window_size, self._episode_prices[0], dtype=np.float32
        )

        obs = self._get_obs()
        info = {
            "inventory": self._inventory,
            "realized_pnl": self._realized_pnl,
            "mid_price": float(self._episode_prices[0]),
        }
        return obs, info

    def step(self, action):
        bid_level = int(action[0])
        ask_level = int(action[1])

        mid_price = float(self._episode_prices[self._current_step])
        spread = self.spread_fraction * mid_price
        best_bid = mid_price - spread / 2.0
        best_ask = mid_price + spread / 2.0

        bid_price = mid_price - self.LEVEL_MULTIPLIERS[bid_level] * self.tick_size
        ask_price = mid_price + self.LEVEL_MULTIPLIERS[ask_level] * self.tick_size

        bid_filled, ask_filled = self._simulate_fills(bid_price, ask_price, mid_price)

        prev_pnl = self._realized_pnl
        if bid_filled:
            self._inventory += 1
            self._realized_pnl -= bid_price
        if ask_filled:
            self._inventory -= 1
            self._realized_pnl += ask_price

        delta_pnl = self._realized_pnl - prev_pnl
        reward = delta_pnl - self.gamma * (self._inventory ** 2) - self.kappa

        self._current_step += 1
        self._price_history = np.roll(self._price_history, -1)
        self._price_history[-1] = mid_price

        terminated = False
        if abs(self._inventory) >= self.max_inventory:
            terminated = True
            reward -= self.liquidation_penalty * abs(self._inventory)
        elif self._current_step >= self._max_steps:
            terminated = True
            if self._inventory != 0:
                reward -= self.liquidation_penalty * abs(self._inventory)

        truncated = False
        obs = self._get_obs()
        info = {
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "inventory": self._inventory,
            "realized_pnl": self._realized_pnl,
            "mid_price": mid_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        idx = min(self._current_step, self._max_steps - 1)
        mid = float(self._episode_prices[idx])
        msg = (
            f"Step {self._current_step:>4}/{self._max_steps} | "
            f"Mid: {mid:.4f} | Inv: {self._inventory:+d} | "
            f"PnL: {self._realized_pnl:.4f}"
        )
        if self.render_mode == "human":
            print(msg)
        return msg

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        idx = min(self._current_step, self._max_steps - 1)
        mid_price = float(self._episode_prices[idx])
        spread = self.spread_fraction * mid_price
        best_bid = mid_price - spread / 2.0
        best_ask = mid_price + spread / 2.0

        return np.array(
            [
                best_bid,
                best_ask,
                mid_price,
                spread,
                float(self._inventory),
                self._realized_pnl,
                self._current_step / self._max_steps,
                *self._price_history,
            ],
            dtype=np.float32,
        )

    def _simulate_fills(self, bid_price, ask_price, mid_price):
        """Exponential fill probability — tighter spread => higher probability."""
        bid_dist = max(mid_price - bid_price, 0.0)
        ask_dist = max(ask_price - mid_price, 0.0)
        p_bid = float(np.exp(-self.lambda_ * bid_dist))
        p_ask = float(np.exp(-self.lambda_ * ask_dist))
        return (
            bool(self.np_random.random() < p_bid),
            bool(self.np_random.random() < p_ask),
        )

    def _generate_gbm_path(self):
        """Geometric Brownian Motion price simulation."""
        prices = np.empty(self.episode_length, dtype=np.float32)
        prices[0] = self.S0
        for t in range(1, self.episode_length):
            z = self.np_random.standard_normal()
            prices[t] = prices[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt
                + self.sigma * np.sqrt(self.dt) * z
            )
        return prices

    def _load_csv_data(self):
        """Load close prices from a CSV file.
        Required columns: timestamp, open, high, low, close, volume
        """
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.lower()
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        prices = df["close"].values.astype(np.float32)
        end = self.end_idx if self.end_idx is not None else len(prices)
        prices = prices[self.start_idx : end]

        if len(prices) == 0:
            raise ValueError("CSV slice is empty — check start_idx and end_idx.")
        return prices

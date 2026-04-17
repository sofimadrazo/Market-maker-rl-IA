import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
 
 
class MarketMakerEnv(gym.Env):
    """
    Entorno de simulación para un Market Maker (hacedor de mercado).
 
    Un market maker es un agente que cotiza simultáneamente precios de compra
    (bid) y venta (ask) de un activo financiero, ganando dinero con la diferencia
    entre ambos precios (el spread). Este entorno simula ese proceso para que un
    agente de Reinforcement Learning aprenda a tomar esas decisiones de forma óptima.
 
    Modos de renderizado disponibles: "human" (imprime por consola) o "ansi".
    """
 
    metadata = {"render_modes": ["human", "ansi"]}
 
    # Distancias disponibles desde el precio medio para colocar las órdenes,
    # expresadas como múltiplos del tick_size (la unidad mínima de precio).
    # Por ejemplo, si tick_size=0.01, el nivel 0 coloca la orden a 0.5 céntimos
    # del precio medio, y el nivel 4 la coloca a 5 céntimos.
    LEVEL_MULTIPLIERS = [0.5, 1.0, 2.0, 3.0, 5.0]
 
    def __init__(
        self,
        data_path=None,          # Ruta a un CSV con precios reales (opcional)
        start_idx=0,             # Índice inicial dentro del CSV
        end_idx=None,            # Índice final dentro del CSV (None = hasta el final)
        price_window_size=10,    # Cuántos precios históricos incluir en la observación
        max_inventory=10,        # Límite máximo de unidades que el agente puede acumular
        tick_size=0.01,          # Variación mínima de precio (el "escalón" del mercado)
        spread_fraction=0.002,   # Spread de referencia del mercado, como fracción del precio
        gamma=0.1,               # Penalización por acumular inventario (riesgo de posición)
        kappa=0.001,             # Coste fijo por cada paso de tiempo (coste operativo)
        lambda_=30.0,            # Controla qué tan rápido cae la probabilidad de ejecución
        liquidation_penalty=0.05,# Penalización si el episodio termina con inventario sin cerrar
        n_bid_levels=5,          # Número de niveles disponibles para colocar órdenes de compra
        n_ask_levels=5,          # Número de niveles disponibles para colocar órdenes de venta
        episode_length=1000,     # Duración máxima de cada episodio (en pasos)
        S0=100.0,                # Precio inicial si se genera una trayectoria sintética
        mu=0.0,                  # Tendencia del precio (drift) en el modelo GBM
        sigma=0.2,               # Volatilidad anualizada del precio en el modelo GBM
        dt=1 / 252,              # Intervalo de tiempo por paso (1/252 ≈ un día de trading)
        render_mode=None,
    ):
        super().__init__()
 
        # Guardamos todos los parámetros de configuración
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
 
        # Si se proporciona un CSV, cargamos los precios una sola vez al iniciar
        # (para no repetir la lectura en cada episodio)
        self._csv_prices = None
        if data_path is not None:
            self._csv_prices = self._load_csv_data()
 
        # La observación que recibe el agente en cada paso tiene:
        # - 7 valores fijos (bid, ask, mid, spread, inventario, PnL, progreso del episodio)
        # - price_window_size valores históricos de precio
        obs_size = 7 + price_window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
 
        # La acción del agente es elegir simultáneamente:
        # - Un nivel para la orden de compra (bid): de 0 a n_bid_levels-1
        # - Un nivel para la orden de venta (ask): de 0 a n_ask_levels-1
        self.action_space = spaces.MultiDiscrete([n_bid_levels, n_ask_levels])
 
        # Variables internas del episodio; se inicializan en reset()
        self._episode_prices = None   # Secuencia de precios del episodio actual
        self._price_history = None    # Ventana deslizante de precios recientes
        self._current_step = 0        # Paso actual dentro del episodio
        self._max_steps = episode_length  # Duración total del episodio
        self._inventory = 0           # Unidades acumuladas (positivo = largo, negativo = corto)
        self._realized_pnl = 0.0      # Beneficio/pérdida acumulado en efectivo
 
    # ------------------------------------------------------------------
    # API de Gymnasium: los tres métodos principales del entorno
    # ------------------------------------------------------------------
 
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al comienzo de un nuevo episodio.
        Si hay precios del CSV disponibles, los usa; si no, genera una
        trayectoria sintética mediante el modelo GBM.
        Devuelve la primera observación e información inicial.
        """
        super().reset(seed=seed)
 
        # Reseteamos el estado interno
        self._current_step = 0
        self._inventory = 0
        self._realized_pnl = 0.0
 
        # Seleccionamos la fuente de precios para este episodio
        if self._csv_prices is not None:
            self._episode_prices = self._csv_prices
            self._max_steps = len(self._episode_prices)
        else:
            self._episode_prices = self._generate_gbm_path()
            self._max_steps = len(self._episode_prices)
 
        # Rellenamos el historial de precios con el precio inicial
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
        """
        Ejecuta un paso en el entorno dado la acción elegida por el agente.
 
        El agente elige en qué nivel colocar su orden de compra y venta.
        El entorno simula si esas órdenes se ejecutan, actualiza el inventario
        y el PnL, y calcula la recompensa.
 
        Devuelve: observación, recompensa, terminated, truncated, info.
        """
        # Desglosamos la acción en los niveles de bid y ask elegidos
        bid_level = int(action[0])
        ask_level = int(action[1])
 
        # Obtenemos el precio medio actual del mercado
        mid_price = float(self._episode_prices[self._current_step])
 
        # Calculamos el spread de referencia del mercado y los mejores precios teóricos
        spread = self.spread_fraction * mid_price
        best_bid = mid_price - spread / 2.0
        best_ask = mid_price + spread / 2.0
 
        # El agente coloca sus órdenes a la distancia elegida del precio medio
        bid_price = mid_price - self.LEVEL_MULTIPLIERS[bid_level] * self.tick_size
        ask_price = mid_price + self.LEVEL_MULTIPLIERS[ask_level] * self.tick_size
 
        # Simulamos si las órdenes del agente se ejecutan en este paso
        bid_filled, ask_filled = self._simulate_fills(bid_price, ask_price, mid_price)
 
        # Actualizamos el inventario y el PnL según las ejecuciones
        prev_pnl = self._realized_pnl
        if bid_filled:
            # Si se ejecuta la compra: sumamos 1 unidad y restamos el coste
            self._inventory += 1
            self._realized_pnl -= bid_price
        if ask_filled:
            # Si se ejecuta la venta: restamos 1 unidad y sumamos el ingreso
            self._inventory -= 1
            self._realized_pnl += ask_price
 
        # La recompensa tiene tres componentes:
        # 1. Beneficio neto de las ejecuciones en este paso
        # 2. Penalización proporcional al cuadrado del inventario (riesgo de posición)
        # 3. Coste fijo por cada paso de tiempo (coste operativo)
        delta_pnl = self._realized_pnl - prev_pnl
        reward = delta_pnl - self.gamma * (self._inventory ** 2) - self.kappa
 
        # Avanzamos al siguiente paso y actualizamos el historial de precios
        self._current_step += 1
        self._price_history = np.roll(self._price_history, -1)  # Desplazamos la ventana
        self._price_history[-1] = mid_price                      # Añadimos el precio actual
 
        # Comprobamos si el episodio debe terminar
        terminated = False
        if abs(self._inventory) >= self.max_inventory:
            # El agente ha acumulado demasiado inventario: fin forzado con penalización
            terminated = True
            reward -= self.liquidation_penalty * abs(self._inventory)
        elif self._current_step >= self._max_steps:
            # Se agotaron los pasos del episodio; penalizamos si queda inventario abierto
            terminated = True
            if self._inventory != 0:
                reward -= self.liquidation_penalty * abs(self._inventory)
 
        # truncated=False porque no usamos límite de tiempo externo (solo el interno)
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
 
        # Si el modo de renderizado es "human", mostramos el estado por consola
        if self.render_mode == "human":
            self.render()
 
        return obs, float(reward), terminated, truncated, info
 
    def render(self):
        """
        Muestra por consola el estado actual del entorno:
        paso actual, precio medio, inventario y PnL acumulado.
        """
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
        """Limpieza de recursos al cerrar el entorno (aquí no es necesaria)."""
        pass
 
    # ------------------------------------------------------------------
    # Métodos privados de apoyo
    # ------------------------------------------------------------------
 
    def _get_obs(self):
        """
        Construye el vector de observación que recibe el agente en cada paso.
 
        Incluye: mejor bid y ask del mercado, precio medio, spread actual,
        inventario actual, PnL acumulado, progreso del episodio (0 a 1)
        y los últimos `price_window_size` precios históricos.
        """
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
                self._current_step / self._max_steps,  # Fracción de episodio completada
                *self._price_history,                   # Ventana de precios históricos
            ],
            dtype=np.float32,
        )
 
    def _simulate_fills(self, bid_price, ask_price, mid_price):
        """
        Decide aleatoriamente si las órdenes del agente se ejecutan en este paso.
 
        La probabilidad de ejecución sigue una función exponencial decreciente:
        cuanto más lejos del precio medio esté la orden, menos probable es que
        se ejecute. Esto replica el comportamiento real del libro de órdenes,
        donde las órdenes más agresivas (más cercanas al mid) se ejecutan más.
 
        Fórmula: P(ejecución) = exp(-lambda * distancia_al_mid)
        """
        bid_dist = max(mid_price - bid_price, 0.0)  # Distancia de la compra al mid
        ask_dist = max(ask_price - mid_price, 0.0)  # Distancia de la venta al mid
        p_bid = float(np.exp(-self.lambda_ * bid_dist))
        p_ask = float(np.exp(-self.lambda_ * ask_dist))
        return (
            bool(self.np_random.random() < p_bid),
            bool(self.np_random.random() < p_ask),
        )
 
    def _generate_gbm_path(self):
        """
        Genera una trayectoria sintética de precios usando el modelo de
        Movimiento Browniano Geométrico (GBM), el modelo estándar para
        simular precios de activos financieros.
 
        El precio evoluciona paso a paso siguiendo la fórmula:
            S(t+1) = S(t) * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)
        donde Z es un número aleatorio con distribución normal estándar.
        """
        prices = np.empty(self.episode_length, dtype=np.float32)
        prices[0] = self.S0  # Precio inicial
        for t in range(1, self.episode_length):
            z = self.np_random.standard_normal()  # Choque aleatorio del mercado
            prices[t] = prices[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt
                + self.sigma * np.sqrt(self.dt) * z
            )
        return prices
 
    def _load_csv_data(self):
        """
        Carga una serie de precios de cierre desde un archivo CSV.
 
        El CSV debe tener obligatoriamente las columnas:
        timestamp, open, high, low, close, volume
 
        Solo se usa la columna 'close' como precio de referencia.
        Se puede recortar la serie usando start_idx y end_idx.
        """
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.lower()  # Normalizamos los nombres de columna
 
        # Verificamos que el CSV tenga todas las columnas necesarias
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
 
        # Extraemos solo la columna de precios de cierre y aplicamos el recorte
        prices = df["close"].values.astype(np.float32)
        end = self.end_idx if self.end_idx is not None else len(prices)
        prices = prices[self.start_idx : end]
 
        if len(prices) == 0:
            raise ValueError("CSV slice is empty — check start_idx and end_idx.")
        return prices
 
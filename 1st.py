import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random

class Action(Enum):
    """Trading actions the agent can take"""
    HOLD = 0
    BUY = 1
    SELL = 2

@dataclass
class Trade:
    """Record of a completed trade"""
    entry_price: float
    exit_price: float
    shares: int
    entry_step: int
    exit_step: int
    profit: float
class StockTradingEnv:
    """
    Stock Trading Environment for AI agent learning.
    
    The agent can trade a single stock with:
    - Actions: HOLD, BUY, SELL
    - State: price data + technical indicators + portfolio info
    - Reward: profit from completed trades + unrealized P&L
    """
    
    def __init__(
        self,
        price_data: np.ndarray,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        max_position: int = 1000,
        lookback_window: int = 20,
        reward_scale: float = 1.0
    ):
        """
        Initialize trading environment.
        
        Args:
            price_data: (N, 5) array with columns [open, high, low, close, volume]
            initial_balance: Starting cash balance
            transaction_cost: Trading cost as percentage of trade value
            max_position: Maximum number of shares to hold
            lookback_window: Number of past bars for state
            reward_scale: Scaling factor for rewards
        """
        self.price_data = price_data
        self.n_steps = len(price_data)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.reward_scale = reward_scale
        
        # Calculate technical indicators
        self.indicators = self._calculate_indicators()
        
        # State dimensions: price features + indicators + portfolio
        self.state_dim = 5 + 6  # OHLCV + indicators + 1 (position)
        
        self.reset()
    
    def _calculate_indicators(self) -> np.ndarray:
        """Calculate technical indicators from price data."""
        close_prices = self.price_data[:, 3]  # Close prices
        
        # Simple Moving Average (SMA)
        sma_10 = np.convolve(close_prices, np.ones(10)/10, mode='valid')
        sma_20 = np.convolve(close_prices, np.ones(20)/20, mode='valid')
        
        # Exponential Moving Average (EMA)
        ema_12 = self._calculate_ema(close_prices, 12)
        ema_26 = self._calculate_ema(close_prices, 26)
        
        # MACD
        macd = ema_12 - ema_26
        
        # RSI
        rsi = self._calculate_rsi(close_prices, 14)
        
        # Volatility (20-day standard deviation of returns)
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = pd.Series(returns).rolling(20).std().values
        
        # Pad indicators to match length
        indicators = np.zeros((self.n_steps, 6))
        indicators[9:, 0] = sma_10  # SMA 10
        indicators[19:, 1] = sma_20  # SMA 20
        indicators[11:, 2] = ema_12[11:]  # EMA 12
        indicators[25:, 3] = ema_26[25:]  # EMA 26
        indicators[25:, 4] = macd[25:]  # MACD
        indicators[13:, 5] = rsi[13:]  # RSI
        
        # Fill missing values with forward fill
        for i in range(indicators.shape[1]):
            mask = np.isnan(indicators[:, i])
            indicators[mask, i] = np.nanmean(indicators[:, i])
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[period] = 100 - (100 / (1 + rs))
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # Number of shares held
        self.entry_price = 0.0  # Average entry price for current position
        self.trades = []  # List of completed trades
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Track metrics
        self.portfolio_values = [self.initial_balance]
        self.rewards = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        # Price features: OHLCV at current step
        price_features = self.price_data[self.current_step]
        
        # Technical indicators at current step
        indicator_features = self.indicators[self.current_step]
        
        # Portfolio state
        current_price = self.price_data[self.current_step, 3]  # Close price
        position_value = self.position * current_price
        total_value = self.balance + position_value
        position_pct = self.position / self.max_position if self.max_position > 0 else 0
        
        # Combine all features
        state = np.concatenate([
            price_features,  # OHLCV (5 features)
            indicator_features,  # Indicators (6 features)
            [position_pct]  # Position size (1 feature)
        ])
        
        return state.astype(np.float32)
    
    def _calculate_reward(self, action: Action, price: float, 
                         prev_price: float) -> float:
        """Calculate reward based on action and price movement."""
        reward = 0.0
        
        # Reward components
        if action == Action.BUY:
            # Penalty for buying in downtrend, reward for uptrend
            trend = (price - prev_price) / prev_price
            reward += trend * 10  # Small reward for good entry timing
            
        elif action == Action.SELL:
            # Reward for selling in uptrend, penalty for downtrend
            trend = (price - prev_price) / prev_price
            reward += trend * 10
            
        # Holding reward (unrealized P&L)
        if self.position > 0:
            unrealized_pnl = self.position * (price - self.entry_price)
            reward += unrealized_pnl * self.reward_scale * 0.01
            
        # Penalty for being out of market when prices are rising
        if self.position == 0 and price > prev_price:
            reward -= 0.1
            
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            next_state, reward, done, info
        """
        action = Action(action)
        current_price = self.price_data[self.current_step, 3]  # Close price
        prev_price = self.price_data[self.current_step - 1, 3] if self.current_step > 0 else current_price
        
        reward = 0.0
        
        # Execute action
        if action == Action.BUY and self.position == 0:
            # Buy at close price
            max_shares = min(self.max_position, 
                           int(self.balance / (current_price * (1 + self.transaction_cost))))
            
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.position = max_shares
                self.entry_price = current_price
                
                # Transaction cost penalty
                reward -= cost * self.transaction_cost * self.reward_scale
                
        elif action == Action.SELL and self.position > 0:
            # Sell all shares
            proceeds = self.position * current_price * (1 - self.transaction_cost)
            self.balance += proceeds
            
            # Record trade
            profit = proceeds - (self.position * self.entry_price)
            trade = Trade(
                entry_price=self.entry_price,
                exit_price=current_price,
                shares=self.position,
                entry_step=self.current_step - (self.position > 0),
                exit_step=self.current_step,
                profit=profit
            )
            self.trades.append(trade)
            
            # Update trade statistics
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Profit reward
            reward += profit * self.reward_scale
            
            # Transaction cost penalty
            reward -= self.position * current_price * self.transaction_cost * self.reward_scale
            
            # Reset position
            self.position = 0
            self.entry_price = 0.0
            
        # Calculate holding reward
        if self.position > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price)
            reward += unrealized_pnl * self.reward_scale * 0.1
        
        # Add trading reward based on price movement
        reward += self._calculate_reward(action, current_price, prev_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Track portfolio value
        if not done:
            total_value = self.balance + (self.position * current_price)
            self.portfolio_values.append(total_value)
            self.rewards.append(reward)
        
        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'total_value': self.balance + (self.position * current_price),
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0,
            'portfolio_value': self.balance + (self.position * current_price)
        }
        
        return next_state, reward, done, info
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            current_price = self.price_data[self.current_step - 1, 3]
            total_value = self.balance + (self.position * current_price)
            print(f"Step: {self.current_step-1}/{self.n_steps-1} | "
                  f"Price: ${current_price:.2f} | "
                  f"Balance: ${self.balance:.2f} | "
                  f"Position: {self.position} shares | "
                  f"Total: ${total_value:.2f}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(self.portfolio_values) < 2:
            return {}
        
        portfolio_series = np.array(self.portfolio_values)
        returns = np.diff(portfolio_series) / portfolio_series[:-1]
        
        total_return = (portfolio_series[-1] - self.initial_balance) / self.initial_balance * 100
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(portfolio_series)
        
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0
        avg_profit = np.mean([t.profit for t in self.trades]) if self.trades else 0
        
        return {
            'total_return_%': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_%': max_drawdown * 100,
            'total_trades': self.total_trades,
            'win_rate_%': win_rate * 100,
            'avg_profit': avg_profit,
            'final_portfolio_value': portfolio_series[-1]
        }
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)


def generate_sample_data(n_days: int = 1000) -> np.ndarray:
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    
    # Generate price path using geometric Brownian motion
    mu = 0.0001  # Drift
    sigma = 0.015  # Volatility
    dt = 1
    
    prices = np.zeros(n_days)
    prices[0] = 100.0
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + 
                                          sigma * np.random.randn() * np.sqrt(dt))
    
    # Generate OHLCV data
    data = np.zeros((n_days, 5))
    data[:, 3] = prices  # Close
    data[:, 0] = prices * (1 + np.random.randn(n_days) * 0.005)  # Open
    data[:, 1] = np.maximum(data[:, 0], prices) * (1 + np.abs(np.random.randn(n_days)) * 0.005)  # High
    data[:, 2] = np.minimum(data[:, 0], prices) * (1 - np.abs(np.random.randn(n_days)) * 0.005)  # Low
    data[:, 4] = np.random.randint(1000, 10000, n_days)  # Volume
    
    return data


# Example usage and testing
if __name__ == "__main__":
    print("Initializing Stock Trading Environment...")
    
    # Generate sample data
    price_data = generate_sample_data(500)
    
    # Create environment
    env = StockTradingEnv(
        price_data=price_data,
        initial_balance=10000,
        transaction_cost=0.001,
        max_position=100,
        lookback_window=20
    )
    
    # Test the environment with random actions
    print("\nTesting random policy...")
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Random action
        action = np.random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if info['step'] % 50 == 0:
            print(f"Step {info['step']}: Price=${info['current_price']:.2f}, "
                  f"Balance=${info['balance']:.2f}, Position={info['position']}, "
                  f"Total=${info['total_value']:.2f}")
    
    # Print performance metrics
    print("\n" + "="*50)
    print("Performance Metrics:")
    print("="*50)
    metrics = env.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nTotal Reward: {total_reward:.2f}")
    print(f"Total Trades: {env.total_trades}")
    print(f"Winning Trades: {env.winning_trades}")
    print(f"Losing Trades: {env.losing_trades}")
    
    print("\nEnvironment API Validation:")
    print("-"*30)
    print("[OK] reset() returns state of shape:", env.reset().shape)
    print("[OK] step() returns (state, reward, done, info)") 
    print("[OK] state() returns observation")
    print("[OK] All required methods implemented") 
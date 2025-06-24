import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.environment import TradingEnvironment
from src.model import TradingModel
from src.utils.config_loader import setup_console_logging

# Setup logging
setup_console_logging('INFO')

@pytest.fixture
def sample_data():
    """
    Fixture to load a sample of the consolidated data for testing.
    """
    # Use a larger, more realistic dataset for testing
    file_path = Path(__file__).parent.parent / 'data' / 'processed' / 'BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv'
    if not file_path.exists():
        pytest.skip(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, nrows=500)  # Load more data for a longer episode
    
    # Basic preprocessing to match environment expectations
    df = df.rename(columns={
        'timestamp': 'timestamp',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Ensure all required columns are present and numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            df[col] = 1.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def test_multiple_trades_in_episode(sample_data):
    """
    Test if the model can perform multiple trades in a single episode.
    """
    # Initialize environment and model
    env = TradingEnvironment(df=sample_data, lookback_window=50, ultra_aggressive_small_trades=True)
    model = TradingModel(env=env, model_name="test_model")

    # Run a longer test episode
    obs, _ = env.reset()
    done = False
    step_count = 0
    max_steps = 400  # Run for more steps to allow for multiple trades

    while not done and step_count < max_steps:
        action, _ = model.model.predict(obs, deterministic=False)
        obs, reward, done, _, info = env.step(action)
        print(f"Step: {step_count}, Action: {action}, Reward: {reward:.4f}, Trades: {info['total_trades']}, Open Trades: {info['open_trades_count']}")
        step_count += 1

    # Check if multiple trades were made
    total_trades = info['total_trades']
    print(f"Total trades made in {step_count} steps: {total_trades}")
    assert total_trades > 1, f"Expected multiple trades, but only {total_trades} were made."

    # Check if trade logging and tracing are enabled
    assert env.trade_logger is not None, "Trade logging should be enabled."
    assert env.trade_tracer is not None, "Trade tracing should be enabled."

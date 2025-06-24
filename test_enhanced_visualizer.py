#!/usr/bin/env python3
"""
Test script for the enhanced live trade visualizer
Creates mock trade data to demonstrate the enhanced color coding and connection lines
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

def create_mock_trade_data():
    """Create mock trade trace data with LONG/SHORT entries and closes"""
    
    # Ensure logs directory exists
    logs_dir = Path("logs/trade_traces")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    trace_file = logs_dir / "test_traces.jsonl"
    
    # Sample data: mix of LONG and SHORT trades with closes
    base_time = datetime.now() - timedelta(minutes=30)
    base_price = 45000.0
    
    mock_trades = [
        # LONG trade pair
        {
            "event_type": "TRADE_OPENED",
            "timestamp": base_time,
            "trade_id": "LONG_001",
            "entry_price": base_price,
            "side": "LONG",
            "position_size": 0.01,
            "balance_before_entry": 10000.0,
            "balance_after_entry": 9995.0,
            "entry_timestamp": base_time.isoformat()
        },
        {
            "event_type": "TRADE_CLOSED", 
            "timestamp": base_time + timedelta(minutes=5),
            "trade_id": "LONG_001",
            "exit_price": base_price + 200,
            "side": "LONG",
            "net_pnl": 150.0,
            "balance_before_exit": 9995.0,
            "balance_after_exit": 10145.0,
            "entry_price": base_price,
            "entry_timestamp": base_time.isoformat(),
            "exit_timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            "win_loss": "WIN"
        },
        
        # SHORT trade pair
        {
            "event_type": "TRADE_OPENED",
            "timestamp": base_time + timedelta(minutes=10),
            "trade_id": "SHORT_001",
            "entry_price": base_price + 100,
            "side": "SHORT",
            "position_size": 0.01,
            "balance_before_entry": 10145.0,
            "balance_after_entry": 10140.0,
            "entry_timestamp": (base_time + timedelta(minutes=10)).isoformat()
        },
        {
            "event_type": "TRADE_CLOSED",
            "timestamp": base_time + timedelta(minutes=15),
            "trade_id": "SHORT_001", 
            "exit_price": base_price + 50,
            "side": "SHORT",
            "net_pnl": 45.0,
            "balance_before_exit": 10140.0,
            "balance_after_exit": 10185.0,
            "entry_price": base_price + 100,
            "entry_timestamp": (base_time + timedelta(minutes=10)).isoformat(),
            "exit_timestamp": (base_time + timedelta(minutes=15)).isoformat(),
            "win_loss": "WIN"
        },
        
        # Another LONG trade pair (losing)
        {
            "event_type": "TRADE_OPENED",
            "timestamp": base_time + timedelta(minutes=20),
            "trade_id": "LONG_002",
            "entry_price": base_price + 150,
            "side": "LONG", 
            "position_size": 0.01,
            "balance_before_entry": 10185.0,
            "balance_after_entry": 10180.0,
            "entry_timestamp": (base_time + timedelta(minutes=20)).isoformat()
        },
        {
            "event_type": "TRADE_CLOSED",
            "timestamp": base_time + timedelta(minutes=25),
            "trade_id": "LONG_002",
            "exit_price": base_price + 50,
            "side": "LONG",
            "net_pnl": -105.0,
            "balance_before_exit": 10180.0,
            "balance_after_exit": 10075.0,
            "entry_price": base_price + 150,
            "entry_timestamp": (base_time + timedelta(minutes=20)).isoformat(),
            "exit_timestamp": (base_time + timedelta(minutes=25)).isoformat(),
            "win_loss": "LOSS"
        },
        
        # Open LONG trade (no close yet)
        {
            "event_type": "TRADE_OPENED",
            "timestamp": base_time + timedelta(minutes=28),
            "trade_id": "LONG_003",
            "entry_price": base_price + 75,
            "side": "LONG",
            "position_size": 0.01,
            "balance_before_entry": 10075.0,
            "balance_after_entry": 10070.0,
            "entry_timestamp": (base_time + timedelta(minutes=28)).isoformat()
        },
        
        # Open SHORT trade (no close yet)
        {
            "event_type": "TRADE_OPENED",
            "timestamp": base_time + timedelta(minutes=29),
            "trade_id": "SHORT_002", 
            "entry_price": base_price + 25,
            "side": "SHORT",
            "position_size": 0.01,
            "balance_before_entry": 10070.0,
            "balance_after_entry": 10065.0,
            "entry_timestamp": (base_time + timedelta(minutes=29)).isoformat()
        }
    ]
    
    # Write trade traces to file
    with open(trace_file, 'w') as f:
        for trade in mock_trades:
            trace_entry = {
                "trace_metadata": {
                    "event_type": trade["event_type"],
                    "timestamp": trade["timestamp"].isoformat()
                },
                "event_data": trade
            }
            f.write(json.dumps(trace_entry, default=str) + '\n')
    
    print(f"✅ Created mock trade data in: {trace_file}")
    print(f"📊 Generated {len(mock_trades)} trade events")
    print(f"🟢 2 LONG trades (1 win, 1 loss)")
    print(f"🔴 1 SHORT trade (1 win)")
    print(f"📂 2 open trades (1 LONG, 1 SHORT)")
    
    return str(trace_file)

def test_visualizer():
    """Test the enhanced visualizer with mock data"""
    print("🧪 Testing Enhanced Live Trade Visualizer...")
    
    # Create mock data
    trace_file = create_mock_trade_data()
    
    print(f"\n🚀 Starting enhanced visualizer...")
    print(f"📁 Data file: {trace_file}")
    print("\n📋 Expected behavior:")
    print("   - GREEN connection lines for LONG->CLOSE trades")
    print("   - RED connection lines for SHORT->CLOSE trades") 
    print("   - Green upward triangles (^) for LONG entries")
    print("   - Red downward triangles (v) for SHORT entries")
    print("   - Purple squares for CLOSE operations")
    print("   - Entry and exit prices labeled clearly")
    print("   - P&L shown on connection lines")
    
    # Import and run the visualizer
    try:
        import sys
        sys.path.append('graphs')
        from live_trade_visualizer_enhanced import EnhancedLiveTradeVisualizer
        
        visualizer = EnhancedLiveTradeVisualizer(
            trace_file=trace_file,
            update_interval=1.0,
            max_points=100
        )
        
        print("\n🎯 Starting visualization... (Press Ctrl+C to stop)")
        visualizer.start_visualization()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the enhanced visualizer is in the graphs/ directory")
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")

if __name__ == "__main__":
    test_visualizer()

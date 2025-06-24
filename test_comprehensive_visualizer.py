#!/usr/bin/env python3
"""
Enhanced test script for live trade visualizer 
Creates comprehensive mock data with LONG, SHORT, and CLOSE trades
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

def create_comprehensive_mock_data():
    """Create comprehensive mock trade trace data with LONG/SHORT entries and closes"""
    
    # Ensure logs directory exists
    logs_dir = Path("logs/trade_traces")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    trace_file = logs_dir / "comprehensive_test_traces.jsonl"
    
    # Sample data: comprehensive mix of LONG and SHORT trades with closes
    base_time = datetime.now() - timedelta(minutes=60)
    base_price = 45000.0
    
    mock_traces = []
    trade_counter = 1
    
    # 1. LONG trade pair (winning)
    open_time = base_time
    open_price = base_price
    mock_traces.extend([
        {
            "trace_metadata": {
                "event_type": "TRADE_OPENED",
                "timestamp": open_time.isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "LONG",
                "entry_action": "BUY", 
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "position_size": 0.01,
                "balance_before_entry": 10000.0,
                "balance_after_entry": 9995.0
            }
        },
        {
            "trace_metadata": {
                "event_type": "TRADE_CLOSED",
                "timestamp": (open_time + timedelta(minutes=10)).isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "LONG",
                "exit_price": open_price + 500,
                "exit_timestamp": (open_time + timedelta(minutes=10)).isoformat(),
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "net_pnl": 485.0,
                "balance_before_exit": 9995.0,
                "balance_after_exit": 10480.0,
                "win_loss": "WIN"
            }
        }
    ])
    trade_counter += 1
    
    # 2. SHORT trade pair (winning)
    open_time = base_time + timedelta(minutes=15)
    open_price = base_price + 300
    mock_traces.extend([
        {
            "trace_metadata": {
                "event_type": "TRADE_OPENED",
                "timestamp": open_time.isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "SHORT",
                "entry_action": "SELL",
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "position_size": 0.01,
                "balance_before_entry": 10480.0,
                "balance_after_entry": 10475.0
            }
        },
        {
            "trace_metadata": {
                "event_type": "TRADE_CLOSED",
                "timestamp": (open_time + timedelta(minutes=8)).isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "SHORT",
                "exit_price": open_price - 200,
                "exit_timestamp": (open_time + timedelta(minutes=8)).isoformat(),
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "net_pnl": 195.0,
                "balance_before_exit": 10475.0,
                "balance_after_exit": 10670.0,
                "win_loss": "WIN"
            }
        }
    ])
    trade_counter += 1
    
    # 3. LONG trade pair (losing)
    open_time = base_time + timedelta(minutes=30)
    open_price = base_price + 400
    mock_traces.extend([
        {
            "trace_metadata": {
                "event_type": "TRADE_OPENED",
                "timestamp": open_time.isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "LONG",
                "entry_action": "BUY",
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "position_size": 0.01,
                "balance_before_entry": 10670.0,
                "balance_after_entry": 10665.0
            }
        },
        {
            "trace_metadata": {
                "event_type": "TRADE_CLOSED",
                "timestamp": (open_time + timedelta(minutes=12)).isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "LONG",
                "exit_price": open_price - 300,
                "exit_timestamp": (open_time + timedelta(minutes=12)).isoformat(),
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "net_pnl": -305.0,
                "balance_before_exit": 10665.0,
                "balance_after_exit": 10360.0,
                "win_loss": "LOSS"
            }
        }
    ])
    trade_counter += 1
    
    # 4. SHORT trade pair (losing)
    open_time = base_time + timedelta(minutes=45)
    open_price = base_price + 100
    mock_traces.extend([
        {
            "trace_metadata": {
                "event_type": "TRADE_OPENED",
                "timestamp": open_time.isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "SHORT",
                "entry_action": "SELL",
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "position_size": 0.01,
                "balance_before_entry": 10360.0,
                "balance_after_entry": 10355.0
            }
        },
        {
            "trace_metadata": {
                "event_type": "TRADE_CLOSED",
                "timestamp": (open_time + timedelta(minutes=6)).isoformat()
            },
            "event_data": {
                "trade_id": f"TRADE_{trade_counter:05d}",
                "side": "SHORT",
                "exit_price": open_price + 150,
                "exit_timestamp": (open_time + timedelta(minutes=6)).isoformat(),
                "entry_price": open_price,
                "entry_timestamp": open_time.isoformat(),
                "net_pnl": -155.0,
                "balance_before_exit": 10355.0,
                "balance_after_exit": 10200.0,
                "win_loss": "LOSS"
            }
        }
    ])
    trade_counter += 1
    
    # 5. Open LONG trade (no close yet)
    open_time = base_time + timedelta(minutes=55)
    open_price = base_price + 250
    mock_traces.append({
        "trace_metadata": {
            "event_type": "TRADE_OPENED",
            "timestamp": open_time.isoformat()
        },
        "event_data": {
            "trade_id": f"TRADE_{trade_counter:05d}",
            "side": "LONG",
            "entry_action": "BUY",
            "entry_price": open_price,
            "entry_timestamp": open_time.isoformat(),
            "position_size": 0.01,
            "balance_before_entry": 10200.0,
            "balance_after_entry": 10195.0
        }
    })
    trade_counter += 1
    
    # 6. Open SHORT trade (no close yet)
    open_time = base_time + timedelta(minutes=58)
    open_price = base_price + 350
    mock_traces.append({
        "trace_metadata": {
            "event_type": "TRADE_OPENED",
            "timestamp": open_time.isoformat()
        },
        "event_data": {
            "trade_id": f"TRADE_{trade_counter:05d}",
            "side": "SHORT",
            "entry_action": "SELL",
            "entry_price": open_price,
            "entry_timestamp": open_time.isoformat(),
            "position_size": 0.01,
            "balance_before_entry": 10195.0,
            "balance_after_entry": 10190.0
        }
    })
    
    # Write trade traces to file
    with open(trace_file, 'w') as f:
        for trace in mock_traces:
            f.write(json.dumps(trace, default=str) + '\n')
    
    print(f"✅ Created comprehensive mock trade data in: {trace_file}")
    print(f"📊 Generated {len(mock_traces)} trade events")
    print(f"🟢 LONG trades: 3 (1 win, 1 loss, 1 open)")
    print(f"🔴 SHORT trades: 3 (1 win, 1 loss, 1 open)")
    print(f"🟣 CLOSE operations: 4 total")
    print(f"📂 Open trades: 2 (1 LONG, 1 SHORT)")
    
    return str(trace_file)

def test_comprehensive_visualizer():
    """Test the enhanced visualizer with comprehensive mock data"""
    print("🧪 Testing Enhanced Live Trade Visualizer with LONG/SHORT/CLOSE data...")
    
    # Create comprehensive mock data
    trace_file = create_comprehensive_mock_data()
    
    print(f"\n🚀 Starting enhanced visualizer...")
    print(f"📁 Data file: {trace_file}")
    print("\n📋 Expected behavior:")
    print("   - 🟢 GREEN connection lines for LONG->CLOSE trades")
    print("   - 🔴 RED connection lines for SHORT->CLOSE trades") 
    print("   - 🟢 Green upward triangles (^) for LONG entries")
    print("   - 🔴 Red downward triangles (v) for SHORT entries")
    print("   - 🟣 Purple squares for CLOSE operations")
    print("   - Entry and exit prices labeled clearly")
    print("   - P&L shown on connection lines (green for profit, red for loss)")
    print("   - 2 open trades visible (1 LONG, 1 SHORT)")
    
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
    test_comprehensive_visualizer()

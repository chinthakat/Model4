#!/usr/bin/env python3
"""
Simple test to manually add trade trace events for visualizer testing
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

def add_test_trade_events():
    """Add some test trade events to the trace file"""
    trace_file = Path("logs/trade_traces/trade_traces.jsonl")
    
    if not trace_file.exists():
        print(f"❌ Trace file not found: {trace_file}")
        return
    
    print("📝 Adding test trade events...")
    
    # Get current time and create some test events
    base_time = datetime.now()
    
    # Test trade events
    test_events = [
        # Open a new trade
        {
            "trace_metadata": {
                "event_type": "TRADE_OPENED",
                "timestamp": (base_time + timedelta(seconds=0)).isoformat(),
            },
            "event_data": {
                "trade_id": "TEST_TRADE_001",
                "entry_action": "BUY",
                "entry_price": 51000.0,
                "entry_timestamp": (base_time + timedelta(seconds=0)).isoformat(),
                "entry_datetime": (base_time + timedelta(seconds=0)).strftime('%Y-%m-%d %H:%M:%S'),
                "side": "LONG",
                "position_size": 0.001,
                "balance_before_entry": 10000.0,
                "balance_after_entry": 9949.0,
                "leverage": 1.0,
                "episode_step_at_entry": 100
            }
        },
        # Close the trade after 2 minutes (profitable)
        {
            "trace_metadata": {
                "event_type": "TRADE_CLOSED",
                "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
            },
            "event_data": {
                "trade_id": "TEST_TRADE_001",
                "entry_action": "BUY",
                "exit_action": "CLOSE",
                "entry_price": 51000.0,
                "exit_price": 51150.0,
                "entry_timestamp": (base_time + timedelta(seconds=0)).isoformat(),
                "exit_timestamp": (base_time + timedelta(minutes=2)).isoformat(),
                "entry_datetime": (base_time + timedelta(seconds=0)).strftime('%Y-%m-%d %H:%M:%S'),
                "exit_datetime": (base_time + timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S'),
                "side": "LONG",
                "position_size": 0.001,
                "gross_pnl": 0.15,
                "net_pnl": 0.14,
                "pnl_percentage": 0.0014,
                "win_loss": "WIN",
                "duration_seconds": 120,
                "duration_steps": 8,
                "total_commission": 0.01,
                "balance_before_exit": 9949.0,
                "balance_after_exit": 9949.14,
                "equity_change": 0.14,
                "episode_step_at_entry": 100,
                "episode_step_at_exit": 108
            }
        },
        # Open another trade
        {
            "trace_metadata": {
                "event_type": "TRADE_OPENED",
                "timestamp": (base_time + timedelta(minutes=3)).isoformat(),
            },
            "event_data": {
                "trade_id": "TEST_TRADE_002",
                "entry_action": "SELL",
                "entry_price": 51200.0,
                "entry_timestamp": (base_time + timedelta(minutes=3)).isoformat(),
                "entry_datetime": (base_time + timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S'),
                "side": "SHORT",
                "position_size": -0.001,
                "balance_before_entry": 9949.14,
                "balance_after_entry": 9897.8,
                "leverage": 1.0,
                "episode_step_at_entry": 120
            }
        },
        # Close second trade (loss)
        {
            "trace_metadata": {
                "event_type": "TRADE_CLOSED",
                "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            },
            "event_data": {
                "trade_id": "TEST_TRADE_002",
                "entry_action": "SELL",
                "exit_action": "CLOSE",
                "entry_price": 51200.0,
                "exit_price": 51280.0,
                "entry_timestamp": (base_time + timedelta(minutes=3)).isoformat(),
                "exit_timestamp": (base_time + timedelta(minutes=5)).isoformat(),
                "entry_datetime": (base_time + timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S'),
                "exit_datetime": (base_time + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
                "side": "SHORT",
                "position_size": -0.001,
                "gross_pnl": -0.08,
                "net_pnl": -0.09,
                "pnl_percentage": -0.0009,
                "win_loss": "LOSS",
                "duration_seconds": 120,
                "duration_steps": 8,
                "total_commission": 0.01,
                "balance_before_exit": 9897.8,
                "balance_after_exit": 9897.71,
                "equity_change": -0.09,
                "episode_step_at_entry": 120,
                "episode_step_at_exit": 128
            }
        }
    ]
    
    # Append to trace file with delays to see live updates
    with open(trace_file, 'a') as f:
        for i, event in enumerate(test_events):
            # Write event
            f.write(json.dumps(event) + '\n')
            f.flush()  # Force write to disk
            
            event_type = event['trace_metadata']['event_type']
            trade_id = event['event_data']['trade_id']
            
            if event_type == 'TRADE_OPENED':
                price = event['event_data']['entry_price']
                action = event['event_data']['entry_action']
                print(f"  ✅ Added {event_type}: {trade_id} - {action} at ${price:.2f}")
            else:
                entry_price = event['event_data']['entry_price']
                exit_price = event['event_data']['exit_price']
                pnl = event['event_data']['net_pnl']
                win_loss = event['event_data']['win_loss']
                print(f"  ✅ Added {event_type}: {trade_id} - ${entry_price:.2f} → ${exit_price:.2f} | P&L: ${pnl:.2f} | {win_loss}")
            
            # Wait to see updates in visualizer
            time.sleep(3)
    
    print("\n🎉 Test events added! Check the live visualizer to see:")
    print("   🔗 Green line connecting winning trade points")
    print("   🔗 Red line connecting losing trade points")
    print("   📊 Updated performance statistics")

if __name__ == "__main__":
    add_test_trade_events()

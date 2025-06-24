#!/usr/bin/env python3
"""
Test script to verify trade data parsing without GUI
"""

import json
import pandas as pd
from pathlib import Path

def test_trade_data_parsing():
    """Test parsing of trade trace data"""
    trace_file = Path("logs/trade_traces/trade_traces.jsonl")
    
    if not trace_file.exists():
        print(f"❌ Trace file not found: {trace_file}")
        return
    
    trades = []
    prices = []
    
    print(f"📊 Reading trade data from: {trace_file}")
    
    with open(trace_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[:5]):  # Test first 5 lines
        if line.strip():
            try:
                trace = json.loads(line.strip())
                event_data = trace.get('event_data', {})
                metadata = trace.get('trace_metadata', {})
                
                if metadata.get('event_type') == 'TRADE_OPENED':
                    trade_info = {
                        'trade_id': event_data.get('trade_id'),
                        'timestamp': event_data.get('entry_timestamp'),
                        'price': event_data.get('entry_price'),
                        'action': event_data.get('entry_action'),
                        'side': event_data.get('side'),
                        'balance_after': event_data.get('balance_after_entry')
                    }
                    trades.append(trade_info)
                    
                    print(f"✅ Trade {i+1}: {trade_info['trade_id']} - {trade_info['action']} at ${trade_info['price']:.2f}")
                    
            except Exception as e:
                print(f"❌ Error parsing line {i+1}: {e}")
    
    print(f"\n📈 Summary:")
    print(f"   Total trades parsed: {len(trades)}")
    print(f"   Price range: ${min(t['price'] for t in trades):.2f} - ${max(t['price'] for t in trades):.2f}")
    print(f"   Time range: {trades[0]['timestamp']} to {trades[-1]['timestamp']}")

if __name__ == "__main__":
    test_trade_data_parsing()

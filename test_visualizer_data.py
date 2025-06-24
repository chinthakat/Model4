#!/usr/bin/env python3
"""
Test script to verify trade data parsing without GUI
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def test_trade_data_parsing():
    """Test parsing of trade trace data including opened and closed trades"""
    trace_file = Path("logs/trade_traces/trade_traces.jsonl")
    
    if not trace_file.exists():
        print(f"❌ Trace file not found: {trace_file}")
        return
    
    opened_trades = []
    closed_trades = []
    
    print(f"📊 Reading trade data from: {trace_file}")
    
    with open(trace_file, 'r') as f:
        lines = f.readlines()
    
    print(f"📄 Total lines in trace file: {len(lines)}")
    
    for i, line in enumerate(lines):
        if line.strip():
            try:
                trace = json.loads(line.strip())
                event_data = trace.get('event_data', {})
                metadata = trace.get('trace_metadata', {})
                event_type = metadata.get('event_type')
                
                if event_type == 'TRADE_OPENED':
                    trade_info = {
                        'trade_id': event_data.get('trade_id'),
                        'timestamp': event_data.get('entry_timestamp'),
                        'datetime': event_data.get('entry_datetime'),
                        'price': event_data.get('entry_price'),
                        'action': event_data.get('entry_action'),
                        'side': event_data.get('side'),
                        'position_size': event_data.get('position_size'),
                        'balance_after': event_data.get('balance_after_entry'),
                        'leverage': event_data.get('leverage'),
                        'episode_step': event_data.get('episode_step_at_entry')
                    }
                    opened_trades.append(trade_info)
                    
                elif event_type == 'TRADE_CLOSED':
                    close_info = {
                        'trade_id': event_data.get('trade_id'),
                        'entry_timestamp': event_data.get('entry_timestamp'),
                        'exit_timestamp': event_data.get('exit_timestamp'),
                        'entry_datetime': event_data.get('entry_datetime'),
                        'exit_datetime': event_data.get('exit_datetime'),
                        'entry_price': event_data.get('entry_price'),
                        'exit_price': event_data.get('exit_price'),
                        'entry_action': event_data.get('entry_action'),
                        'exit_action': event_data.get('exit_action'),
                        'side': event_data.get('side'),
                        'position_size': event_data.get('position_size'),
                        'gross_pnl': event_data.get('gross_pnl'),
                        'net_pnl': event_data.get('net_pnl'),
                        'pnl_percentage': event_data.get('pnl_percentage'),
                        'win_loss': event_data.get('win_loss'),
                        'duration_seconds': event_data.get('duration_seconds'),
                        'duration_steps': event_data.get('duration_steps'),
                        'total_commission': event_data.get('total_commission'),
                        'balance_before_exit': event_data.get('balance_before_exit'),
                        'balance_after_exit': event_data.get('balance_after_exit'),
                        'equity_change': event_data.get('equity_change'),
                        'episode_step_entry': event_data.get('episode_step_at_entry'),
                        'episode_step_exit': event_data.get('episode_step_at_exit')
                    }
                    closed_trades.append(close_info)
                    
            except Exception as e:
                print(f"❌ Error parsing line {i+1}: {e}")
    
    # Display opened trades
    print(f"\n🟢 OPENED TRADES ({len(opened_trades)}):")
    print("=" * 80)
    for i, trade in enumerate(opened_trades[:10]):  # Show first 10
        print(f"  {i+1:2d}. Trade {trade['trade_id']} - {trade['action']} {trade['side']}")
        print(f"      Entry: ${trade['price']:.2f} | Size: {trade['position_size']:.4f} | Step: {trade['episode_step']}")
        print(f"      Time: {trade['datetime']} | Balance: ${trade['balance_after']:.2f}")
        print()
    
    if len(opened_trades) > 10:
        print(f"      ... and {len(opened_trades) - 10} more opened trades")
    
    # Display closed trades
    print(f"\n🔴 CLOSED TRADES ({len(closed_trades)}):")
    print("=" * 80)
    for i, trade in enumerate(closed_trades[:10]):  # Show first 10
        duration_mins = trade['duration_seconds'] / 60 if trade['duration_seconds'] else 0
        pnl_pct = trade['pnl_percentage'] * 100 if trade['pnl_percentage'] else 0
        
        print(f"  {i+1:2d}. Trade {trade['trade_id']} - {trade['entry_action']} → {trade['exit_action']}")
        print(f"      {trade['side']}: ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}")
        print(f"      P&L: ${trade['net_pnl']:.2f} ({pnl_pct:.2f}%) | {trade['win_loss']}")
        print(f"      Duration: {duration_mins:.1f} mins ({trade['duration_steps']} steps)")
        print(f"      Entry: {trade['entry_datetime']}")
        print(f"      Exit:  {trade['exit_datetime']}")
        print()
    
    if len(closed_trades) > 10:
        print(f"      ... and {len(closed_trades) - 10} more closed trades")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("=" * 50)
    print(f"   📈 Total opened trades: {len(opened_trades)}")
    print(f"   📉 Total closed trades: {len(closed_trades)}")
    print(f"   🔄 Currently open: {len(opened_trades) - len(closed_trades)}")
    
    if opened_trades:
        all_prices = [t['price'] for t in opened_trades]
        print(f"   💰 Price range: ${min(all_prices):.2f} - ${max(all_prices):.2f}")
    
    if closed_trades:
        # P&L statistics
        pnls = [t['net_pnl'] for t in closed_trades if t['net_pnl'] is not None]
        wins = [t for t in closed_trades if t['win_loss'] == 'WIN']
        losses = [t for t in closed_trades if t['win_loss'] == 'LOSS']
        
        if pnls:
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(pnls)
            win_rate = len(wins) / len(closed_trades) * 100
            
            print(f"   💵 Total P&L: ${total_pnl:.2f}")
            print(f"   📊 Average P&L: ${avg_pnl:.2f}")
            print(f"   🎯 Win rate: {win_rate:.1f}% ({len(wins)}/{len(closed_trades)})")
            
            if wins:
                avg_win = sum(t['net_pnl'] for t in wins) / len(wins)
                print(f"   ✅ Average win: ${avg_win:.2f}")
            
            if losses:
                avg_loss = sum(t['net_pnl'] for t in losses) / len(losses)
                print(f"   ❌ Average loss: ${avg_loss:.2f}")
        
        # Duration statistics
        durations = [t['duration_seconds'] for t in closed_trades if t['duration_seconds']]
        if durations:
            avg_duration = sum(durations) / len(durations) / 60  # Convert to minutes
            print(f"   ⏱️  Average trade duration: {avg_duration:.1f} minutes")

def verify_trade_closure_logging():
    """Verify that trade closures are being properly logged"""
    print("\n🔍 VERIFYING TRADE CLOSURE LOGGING:")
    print("=" * 60)
    
    trace_file = Path("logs/trade_traces/trade_traces.jsonl")
    
    if not trace_file.exists():
        print("❌ No trace file found - run a trading test first")
        return
    
    events = []
    with open(trace_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    trace = json.loads(line.strip())
                    event_type = trace.get('trace_metadata', {}).get('event_type')
                    trade_id = trace.get('event_data', {}).get('trade_id')
                    events.append((event_type, trade_id))
                except:
                    continue
    
    # Count event types
    opened = [e for e in events if e[0] == 'TRADE_OPENED']
    closed = [e for e in events if e[0] == 'TRADE_CLOSED']
    
    print(f"📊 Event counts:")
    print(f"   TRADE_OPENED events: {len(opened)}")
    print(f"   TRADE_CLOSED events: {len(closed)}")
    
    # Check for matching pairs
    opened_ids = set(e[1] for e in opened)
    closed_ids = set(e[1] for e in closed)
    
    matched = opened_ids.intersection(closed_ids)
    unmatched_open = opened_ids - closed_ids
    unmatched_close = closed_ids - opened_ids
    
    print(f"\n🔗 Trade ID matching:")
    print(f"   Trades with both OPEN and CLOSE: {len(matched)}")
    print(f"   Trades opened but not closed: {len(unmatched_open)}")
    print(f"   Trades closed but no open record: {len(unmatched_close)}")
    
    if unmatched_open:
        print(f"   Still open: {list(unmatched_open)[:5]}")  # Show first 5
    
    print(f"\n✅ Trade closure logging is {'WORKING' if closed else 'NOT WORKING'}")

def analyze_trade_connections():
    """Analyze trade connections for visualization insights"""
    print("\n🔗 TRADE CONNECTION ANALYSIS:")
    print("=" * 60)
    
    trace_file = Path("logs/trade_traces/trade_traces.jsonl")
    
    if not trace_file.exists():
        print("❌ No trace file found")
        return
    
    # Load and organize trades
    open_trades = {}
    closed_trades = []
    trade_connections = []
    
    with open(trace_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    trace = json.loads(line.strip())
                    event_data = trace.get('event_data', {})
                    metadata = trace.get('trace_metadata', {})
                    event_type = metadata.get('event_type')
                    
                    if event_type == 'TRADE_OPENED':
                        trade_id = event_data.get('trade_id')
                        open_trades[trade_id] = {
                            'id': trade_id,
                            'entry_time': event_data.get('entry_datetime'),
                            'entry_price': event_data.get('entry_price'),
                            'side': event_data.get('side'),
                            'action': event_data.get('entry_action')
                        }
                    
                    elif event_type == 'TRADE_CLOSED':
                        trade_id = event_data.get('trade_id')
                        if trade_id in open_trades:
                            open_data = open_trades[trade_id]
                            connection = {
                                'trade_id': trade_id,
                                'side': event_data.get('side'),
                                'entry_time': open_data['entry_time'],
                                'exit_time': event_data.get('exit_datetime'),
                                'entry_price': event_data.get('entry_price'),
                                'exit_price': event_data.get('exit_price'),
                                'pnl': event_data.get('net_pnl'),
                                'win_loss': event_data.get('win_loss'),
                                'duration_seconds': event_data.get('duration_seconds')
                            }
                            trade_connections.append(connection)
                            closed_trades.append(connection)
                            del open_trades[trade_id]
                        
                except Exception as e:
                    continue
    
    print(f"📊 Connection Summary:")
    print(f"   Total trade connections: {len(trade_connections)}")
    print(f"   Currently open (no connection yet): {len(open_trades)}")
    
    if trade_connections:
        # Analyze connections
        winning_connections = [t for t in trade_connections if t['win_loss'] == 'WIN']
        losing_connections = [t for t in trade_connections if t['win_loss'] == 'LOSS']
        
        print(f"\n🟢 Winning Connections: {len(winning_connections)}")
        for i, conn in enumerate(winning_connections[-5:], 1):  # Last 5
            duration_mins = conn['duration_seconds'] / 60 if conn['duration_seconds'] else 0
            print(f"   {i}. {conn['trade_id']}: ${conn['entry_price']:.2f} → ${conn['exit_price']:.2f} ({duration_mins:.1f}min) +${conn['pnl']:.2f}")
        
        print(f"\n🔴 Losing Connections: {len(losing_connections)}")
        for i, conn in enumerate(losing_connections[-5:], 1):  # Last 5
            duration_mins = conn['duration_seconds'] / 60 if conn['duration_seconds'] else 0
            print(f"   {i}. {conn['trade_id']}: ${conn['entry_price']:.2f} → ${conn['exit_price']:.2f} ({duration_mins:.1f}min) ${conn['pnl']:.2f}")
        
        # Price movement analysis
        print(f"\n📈 Price Movement Analysis:")
        long_wins = [t for t in winning_connections if t['side'] == 'LONG']
        long_losses = [t for t in losing_connections if t['side'] == 'LONG']
        short_wins = [t for t in winning_connections if t['side'] == 'SHORT']
        short_losses = [t for t in losing_connections if t['side'] == 'SHORT']
        
        print(f"   LONG trades: {len(long_wins)} wins, {len(long_losses)} losses")
        print(f"   SHORT trades: {len(short_wins)} wins, {len(short_losses)} losses")
        
        if long_wins:
            avg_long_win_move = sum(t['exit_price'] - t['entry_price'] for t in long_wins) / len(long_wins)
            print(f"   Average LONG winning move: ${avg_long_win_move:.2f}")
        
        if short_wins:
            avg_short_win_move = sum(t['entry_price'] - t['exit_price'] for t in short_wins) / len(short_wins)
            print(f"   Average SHORT winning move: ${avg_short_win_move:.2f}")
    
    if open_trades:
        print(f"\n🔵 Open Trades (No Connection Yet):")
        for trade_id, trade in list(open_trades.items())[:5]:
            print(f"   {trade_id}: {trade['action']} {trade['side']} at ${trade['entry_price']:.2f} ({trade['entry_time']})")
    
    print(f"\n💡 Visualization Tips:")
    print(f"   🟢 Green lines = Winning trades")
    print(f"   🔴 Red lines = Losing trades")
    print(f"   📏 Line length = Trade duration") 
    print(f"   📐 Line slope = Price movement direction")

if __name__ == "__main__":
    test_trade_data_parsing()
    verify_trade_closure_logging()
    analyze_trade_connections()
    analyze_trade_connections()

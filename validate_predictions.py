"""
OPTIMIZED VALIDATION SCRIPT - Query PENDING from Database
Validates all PENDING match results from database and syncs back
No CSV required - uses database as single source of truth
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import warnings
import psycopg2
from psycopg2 import sql
import json
import os
warnings.filterwarnings('ignore')

# ==================== API CONFIGURATION ====================
API_KEY = os.getenv("FOOTYSTATSAPI")

# Try multiple API endpoint configurations
API_CONFIGS = [
    {"url": "https://api.football-data-api.com/match", "param": "match_id"},
    {"url": "https://api.footystats.org/match", "param": "id"},
    {"url": "https://api.footystats.org/match", "param": "match_id"},
]

# ==================== DATABASE CONFIGURATION ====================
# Primary database (old credentials)
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'database': os.getenv('DB_DATABASE'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

TABLE_NAME = 'predictions_soccer_v2_ourmodel'

print("\n" + "="*80)
print("AGILITY FOOTBALL PREDICTIONS - DATABASE-DRIVEN VALIDATION")
print("="*80)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ==================== DATABASE CONNECTION ====================
def connect_database(db_config, db_name):
    """Connect to a specific database"""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        print(f"✓ Connected to {db_name}")
        return conn, cursor
    except Exception as e:
        print(f"✗ Failed to connect to {db_name}: {e}")
        return None, None

print("\n[1/4] Connecting to PostgreSQL Database...")
print("="*80)

conn_primary, cursor_primary = connect_database(DB_CONFIG, "PRIMARY (Old Credentials)")

if not conn_primary:
    print(f"\n✗ CRITICAL: Cannot connect to database!")
    exit(1)

# ==================== FETCH PENDING MATCHES FROM DATABASE ====================
print("\n[2/4] Fetching PENDING matches from database...")
print("="*80)

try:
    query = sql.SQL("""
        SELECT 
            match_id,
            home_team,
            away_team,
            ou_prediction,
            ml_prediction,
            over_2_5_odds,
            under_2_5_odds,
            home_win_odds,
            away_win_odds,
            status,
            date
        FROM {}
        WHERE status = %s
        ORDER BY date DESC
    """).format(sql.Identifier(TABLE_NAME))
    
    cursor_primary.execute(query, ('PENDING',))
    results = cursor_primary.fetchall()
    
    if len(results) == 0:
        print(f"ℹ No PENDING matches found in database")
        cursor_primary.close()
        conn_primary.close()
        exit(0)
    
    # Convert to DataFrame for easier handling
    predictions_df = pd.DataFrame(results, columns=[
        'match_id', 'home_team', 'away_team', 'ou_prediction', 'ml_prediction',
        'over_2_5_odds', 'under_2_5_odds', 'home_win_odds', 'away_win_odds', 'status', 'date'
    ])
    
    print(f"✓ Found {len(predictions_df)} PENDING matches to validate")
    print(f"✓ Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
    
except Exception as e:
    print(f"✗ Error fetching PENDING matches: {e}")
    import traceback
    traceback.print_exc()
    cursor_primary.close()
    conn_primary.close()
    exit(1)

# ==================== TEST API FIRST ====================
print("\n[3/4] Testing API configurations...")
print("="*80)

working_api_config = None
test_match_id = predictions_df.iloc[0]['match_id']

print(f"Testing with match ID: {test_match_id}\n")

for i, config in enumerate(API_CONFIGS, 1):
    try:
        url = f"{config['url']}?key={API_KEY}&{config['param']}={test_match_id}"
        print(f"[{i}/{len(API_CONFIGS)}] Testing: {config['url']} with {config['param']}=...")
        
        response = requests.get(config['url'], 
                               params={'key': API_KEY, config['param']: test_match_id},
                               timeout=30)
        
        if response.status_code == 200 and response.text:
            try:
                data = response.json()
                if data.get('success') and data.get('data'):
                    print(f"✓ SUCCESS! This configuration works")
                    working_api_config = config
                    break
                else:
                    print(f"✗ API returned success=false")
            except:
                print(f"✗ Invalid JSON")
        else:
            print(f"✗ HTTP {response.status_code}")
            
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
    
    time.sleep(0.3)

if not working_api_config:
    print(f"\n✗ ERROR: No working API configuration found!")
    print(f"\n💡 SOLUTIONS:")
    print(f"   1. Your match IDs ({test_match_id}) are not compatible with these APIs")
    print(f"   2. Check if match IDs are from a different source (RapidAPI, etc.)")
    print(f"   3. Verify your API key has access to match data")
    print(f"   4. The matches might be too old or not yet in the API")
    cursor_primary.close()
    conn_primary.close()
    exit(1)

print(f"\n✓ Using: {working_api_config['url']} with parameter '{working_api_config['param']}'")

# ==================== FETCH & UPDATE ====================
print("\n[4/4] Fetching match results and updating database...")
print("="*80)

successful_updates = 0
failed_fetches = 0

for idx, row in predictions_df.iterrows():
    match_id = int(row['match_id'])
    
    # Get prediction data from database
    predicted_ou = row['ou_prediction']
    predicted_winner = row['ml_prediction']
    
    # Get odds data from database
    odds_over = float(row['over_2_5_odds']) if row['over_2_5_odds'] else 0
    odds_under = float(row['under_2_5_odds']) if row['under_2_5_odds'] else 0
    odds_home = float(row['home_win_odds']) if row['home_win_odds'] else 0
    odds_away = float(row['away_win_odds']) if row['away_win_odds'] else 0
    
    home_team = row['home_team']
    away_team = row['away_team']
    
    try:
        # Fetch match details using working config
        response = requests.get(
            working_api_config['url'],
            params={'key': API_KEY, working_api_config['param']: match_id},
            timeout=30
        )
        
        if response.status_code == 200 and response.text:
            try:
                data = response.json()
            except json.JSONDecodeError:
                print(f"✗ {match_id}: JSON error")
                failed_fetches += 1
                continue
            
            if data.get('success') and data.get('data'):
                match_data = data['data']
                status = match_data.get('status', '')
                
                if status == 'complete':
                    # Get scores
                    home_score = int(match_data.get('homeGoalCount', 0))
                    away_score = int(match_data.get('awayGoalCount', 0))
                    total_goals = home_score + away_score
                    
                    # Determine winner (for ml_actual)
                    if home_score > away_score:
                        actual_winner = 'Home Win'
                    elif away_score > home_score:
                        actual_winner = 'Away Win'
                    else:
                        actual_winner = 'Draw'
                    
                    # Determine O/U (for ou_actual)
                    actual_over_under = 'Over 2.5' if total_goals > 2.5 else 'Under 2.5'
                    
                    # Calculate correctness
                    ou_correct = 1 if predicted_ou == actual_over_under else 0
                    ml_correct = 1 if predicted_winner == actual_winner else 0
                    
                    # Calculate P/L for Over/Under (ou_pnl)
                    if 'Over' in str(predicted_ou):
                        ou_pnl = round(odds_over - 1, 2) if total_goals > 2.5 else -1.0
                    else:
                        ou_pnl = round(odds_under - 1, 2) if total_goals <= 2.5 else -1.0
                    
                    # Calculate P/L for Winner (ml_pnl)
                    if predicted_winner == 'Home Win':
                        ml_pnl = round(odds_home - 1, 2) if actual_winner == 'Home Win' else -1.0
                    elif predicted_winner == 'Away Win':
                        ml_pnl = round(odds_away - 1, 2) if actual_winner == 'Away Win' else -1.0
                    elif predicted_winner == 'Draw':
                        ml_pnl = round(0 - 1, 2) if actual_winner == 'Draw' else -1.0
                    else:
                        ml_pnl = 0.0
                    
                    # UPDATE DATABASE
                    try:
                        update_query = sql.SQL("""
                            UPDATE {}
                            SET 
                                ml_actual = %s,
                                ou_actual = %s,
                                home_goals = %s,
                                away_goals = %s,
                                total_goals = %s,
                                ou_correct = %s,
                                ml_correct = %s,
                                ou_pnl = %s,
                                ml_pnl = %s,
                                status = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE match_id = %s
                        """).format(sql.Identifier(TABLE_NAME))
                        
                        cursor_primary.execute(update_query, (
                            actual_winner,
                            actual_over_under,
                            home_score,
                            away_score,
                            total_goals,
                            ou_correct,
                            ml_correct,
                            ou_pnl,
                            ml_pnl,
                            'SETTLED',
                            match_id
                        ))
                        
                        conn_primary.commit()
                        successful_updates += 1
                        
                        print(f"✓ {match_id}: {home_team} {home_score}-{away_score} {away_team}")
                        print(f"  → Winner: {actual_winner} (Predicted: {predicted_winner}) {'✓' if ml_correct else '✗'}")
                        print(f"  → O/U: {actual_over_under} (Predicted: {predicted_ou}) {'✓' if ou_correct else '✗'}")
                        print(f"  → P/L: O/U=${ou_pnl:.2f} | ML=${ml_pnl:.2f}")
                        
                    except Exception as e:
                        print(f"⚠ Error updating DB for {match_id}: {str(e)[:50]}")
                        conn_primary.rollback()
                    
                else:
                    print(f"⏳ {match_id}: Not complete (status: {status})")
                    failed_fetches += 1
            else:
                print(f"⚠ {match_id}: No data")
                failed_fetches += 1
        else:
            print(f"✗ {match_id}: HTTP {response.status_code}")
            failed_fetches += 1
        
        time.sleep(0.25)
        
    except Exception as e:
        print(f"✗ {match_id}: {str(e)[:80]}")
        failed_fetches += 1
        conn_primary.rollback()

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Successfully updated: {successful_updates} matches")
print(f"✗ Failed/Pending: {failed_fetches} matches")
print(f"📊 Total PENDING checked: {len(predictions_df)} matches")

if successful_updates > 0:
    # Calculate accuracy from PRIMARY DB
    try:
        accuracy_query = sql.SQL("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN ou_correct = 1 THEN 1 ELSE 0 END) as ou_correct_count,
                SUM(CASE WHEN ml_correct = 1 THEN 1 ELSE 0 END) as ml_correct_count,
                SUM(ou_pnl) as total_ou_pnl,
                SUM(ml_pnl) as total_ml_pnl
            FROM {}
            WHERE status = %s AND ou_actual IS NOT NULL
        """).format(sql.Identifier(TABLE_NAME))
        
        cursor_primary.execute(accuracy_query, ('SETTLED',))
        result = cursor_primary.fetchone()
        
        if result and result[0] > 0:
            total, ou_correct_count, ml_correct_count, total_ou_pnl, total_ml_pnl = result
            print(f"\n📊 OVERALL ACCURACY METRICS (ALL SETTLED):")
            print(f"   O/U Accuracy: {ou_correct_count}/{total} ({100*ou_correct_count/total:.1f}%)")
            print(f"   ML Accuracy: {ml_correct_count}/{total} ({100*ml_correct_count/total:.1f}%)")
            print(f"\n💰 OVERALL PROFIT/LOSS (ALL SETTLED):")
            print(f"   O/U P/L: ${total_ou_pnl:.2f}")
            print(f"   ML P/L: ${total_ml_pnl:.2f}")
            print(f"   Total P/L: ${total_ou_pnl + total_ml_pnl:.2f}")
    except Exception as e:
        print(f"⚠ Could not retrieve accuracy metrics: {e}")

if successful_updates == 0:
    print(f"\n⚠️ WARNING: No matches were successfully validated")
    print(f"   This suggests the match IDs are incompatible with the API")

# Close connections
cursor_primary.close()
conn_primary.close()
print(f"\n✓ Database connection closed")

print("\n" + "="*80)
print("✅ VALIDATION COMPLETE - Database Updated!")
print("="*80)
print(f"⏰ Completed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("="*80)

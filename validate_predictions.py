"""
FIXED VALIDATION SCRIPT - Correct Column Names for agility_soccer_v2
Validates match results and syncs to BOTH databases (PRIMARY + WINBETS)
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import warnings
import psycopg2
from psycopg2 import sql
from pathlib import Path
import json
import os
warnings.filterwarnings('ignore')

# ==================== API CONFIGURATION ====================
API_KEY = "1eac22f8ec8e6da731a49adeae1148f14d6ceca13db5a9ffba65618f97406f4e"

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

# Secondary database (new credentials - WINBETS)
DB_CONFIG_WINBETS = {
    'host': os.getenv('WINBETS_DB_HOST'),
    'port': int(os.getenv('WINBETS_DB_PORT', 5432)),
    'database': os.getenv('WINBETS_DB_DATABASE'),
    'user': os.getenv('WINBETS_DB_USER'),
    'password': os.getenv('WINBETS_DB_PASSWORD')
}

TABLE_NAME = 'agility_soccer_v2'

print("\n" + "="*80)
print("AGILITY FOOTBALL PREDICTIONS - CSV-BASED VALIDATION - DUAL DATABASE")
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

print("\n[1/5] Connecting to PostgreSQL Databases...")
print("="*80)

conn_primary, cursor_primary = connect_database(DB_CONFIG, "PRIMARY (Old Credentials)")
conn_winbets, cursor_winbets = connect_database(DB_CONFIG_WINBETS, "WINBETS (New Credentials)")

if not conn_primary and not conn_winbets:
    print(f"\n✗ CRITICAL: Cannot connect to any database!")
    exit(1)

# ==================== CONFIGURATION ====================
VALIDATION_DATE = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
print(f"\n📅 Validation Date: {VALIDATION_DATE}")

# ==================== LOAD PREDICTIONS FROM CSV ====================
print("\n[2/5] Loading predictions from CSV...")
print("="*80)

try:
    # Try different possible locations
    possible_paths = [
        Path('/mnt/user-data/uploads/predictions_output.csv'),
        Path('predictions_output.csv'),
    ]
    
    predictions_df = None
    for path in possible_paths:
        if path.exists():
            predictions_df = pd.read_csv(path)
            print(f"✓ Loaded CSV from: {path}")
            break
    
    if predictions_df is None:
        print(f"✗ Could not find CSV file. Tried:")
        for p in possible_paths:
            print(f"  - {p}")
        if conn_primary:
            conn_primary.close()
        if conn_winbets:
            conn_winbets.close()
        exit(1)
    
    print(f"✓ Loaded {len(predictions_df)} total predictions")
    print(f"✓ CSV Columns: {list(predictions_df.columns)[:10]}...")

except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    import traceback
    traceback.print_exc()
    if conn_primary:
        conn_primary.close()
    if conn_winbets:
        conn_winbets.close()
    exit(1)

# ==================== FILTER BY DATE ====================
print("\n[3/5] Filtering predictions by date...")
print("="*80)

predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
validation_date_obj = pd.to_datetime(VALIDATION_DATE).date()
predictions_to_validate = predictions_df[predictions_df['date'] == validation_date_obj].copy()

if len(predictions_to_validate) == 0:
    print(f"ℹ No predictions found for {VALIDATION_DATE}")
    if conn_primary:
        conn_primary.close()
    if conn_winbets:
        conn_winbets.close()
    exit(0)

print(f"✓ Found {len(predictions_to_validate)} predictions to validate")

# ==================== TEST API FIRST ====================
print("\n[4/5] Testing API configurations...")
print("="*80)

working_api_config = None
test_match_id = predictions_to_validate.iloc[0]['match_id']

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
    if conn_primary:
        conn_primary.close()
    if conn_winbets:
        conn_winbets.close()
    exit(1)

print(f"\n✓ Using: {working_api_config['url']} with parameter '{working_api_config['param']}'")

# ==================== FETCH & UPDATE ====================
print("\n[5/5] Fetching match results and updating databases...")
print("="*80)

successful_updates = 0
failed_fetches = 0

for idx, row in predictions_to_validate.iterrows():
    match_id = row['match_id']
    
    # Get prediction data from CSV (using actual CSV column names)
    predicted_ou = row.get('ou_prediction', '')
    predicted_winner = row.get('ml_prediction', '')
    
    # Get odds data with correct CSV column names
    odds_over = float(row.get('over_2_5_odds', 0))
    odds_under = float(row.get('under_2_5_odds', 0))
    odds_home = float(row.get('home_win_odds', 0))
    odds_away = float(row.get('away_win_odds', 0))
    odds_draw = float(row.get('draw_odds', 0))
    
    home_team = row.get('home_team', '')
    away_team = row.get('away_team', '')
    
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
                        ml_pnl = round(odds_draw - 1, 2) if actual_winner == 'Draw' else -1.0
                    else:
                        ml_pnl = 0.0
                    
                    # UPDATE PRIMARY DATABASE
                    if conn_primary and cursor_primary:
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
                                float(home_score),
                                float(away_score),
                                float(total_goals),
                                ou_correct,
                                ml_correct,
                                ou_pnl,
                                ml_pnl,
                                'SETTLED',
                                str(match_id)
                            ))
                            
                            conn_primary.commit()
                        except Exception as e:
                            print(f"⚠ Error updating PRIMARY DB for {match_id}: {str(e)[:50]}")
                            conn_primary.rollback()
                    
                    # UPDATE WINBETS DATABASE
                    if conn_winbets and cursor_winbets:
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
                            
                            cursor_winbets.execute(update_query, (
                                actual_winner,
                                actual_over_under,
                                float(home_score),
                                float(away_score),
                                float(total_goals),
                                ou_correct,
                                ml_correct,
                                ou_pnl,
                                ml_pnl,
                                'SETTLED',
                                str(match_id)
                            ))
                            
                            conn_winbets.commit()
                        except Exception as e:
                            print(f"⚠ Error updating WINBETS DB for {match_id}: {str(e)[:50]}")
                            conn_winbets.rollback()
                    
                    successful_updates += 1
                    
                    print(f"✓ {match_id}: {home_team} {home_score}-{away_score} {away_team}")
                    print(f"  → Winner: {actual_winner} (Predicted: {predicted_winner}) {'✓' if ml_correct else '✗'}")
                    print(f"  → O/U: {actual_over_under} (Predicted: {predicted_ou}) {'✓' if ou_correct else '✗'}")
                    print(f"  → P/L: O/U=${ou_pnl:.2f} | ML=${ml_pnl:.2f}")
                    
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
        if conn_primary:
            conn_primary.rollback()
        if conn_winbets:
            conn_winbets.rollback()

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Successfully updated: {successful_updates} matches")
print(f"✗ Failed/Pending: {failed_fetches} matches")

if successful_updates > 0:
    # Calculate accuracy for PRIMARY DB
    if conn_primary and cursor_primary:
        try:
            accuracy_query = sql.SQL("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN ou_correct = 1 THEN 1 ELSE 0 END) as ou_correct_count,
                    SUM(CASE WHEN ml_correct = 1 THEN 1 ELSE 0 END) as ml_correct_count,
                    SUM(ou_pnl) as total_ou_pnl,
                    SUM(ml_pnl) as total_ml_pnl
                FROM {}
                WHERE date = %s AND ou_actual IS NOT NULL
            """).format(sql.Identifier(TABLE_NAME))
            
            cursor_primary.execute(accuracy_query, (VALIDATION_DATE,))
            result = cursor_primary.fetchone()
            
            if result and result[0] > 0:
                total, ou_correct_count, ml_correct_count, total_ou_pnl, total_ml_pnl = result
                print(f"\n📊 ACCURACY METRICS:")
                print(f"   O/U Accuracy: {ou_correct_count}/{total} ({100*ou_correct_count/total:.1f}%)")
                print(f"   ML Accuracy: {ml_correct_count}/{total} ({100*ml_correct_count/total:.1f}%)")
                print(f"\n💰 PROFIT/LOSS:")
                print(f"   O/U P/L: ${total_ou_pnl:.2f}")
                print(f"   ML P/L: ${total_ml_pnl:.2f}")
                print(f"   Total P/L: ${total_ou_pnl + total_ml_pnl:.2f}")
        except Exception as e:
            print(f"⚠ Could not retrieve accuracy metrics: {e}")

if successful_updates == 0:
    print(f"\n⚠️ WARNING: No matches were successfully validated")
    print(f"   This suggests the match IDs are incompatible with the API")

# Close connections
if conn_primary:
    cursor_primary.close()
    conn_primary.close()
    print(f"\n✓ PRIMARY database connection closed")

if conn_winbets:
    cursor_winbets.close()
    conn_winbets.close()
    print(f"✓ WINBETS database connection closed")

print("\n" + "="*80)
print("✅ VALIDATION COMPLETE - Both Databases Synced!")
print("="*80)
print(f"⏰ Completed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("="*80)

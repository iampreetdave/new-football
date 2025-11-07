"""
FOOTBALL PREDICTIONS VALIDATION SCRIPT
Fetches actual match results from API and updates database
Calculates profit/loss based on predictions vs actual results
"""

import pandas as pd
import requests
import time
from datetime import datetime
import warnings
import psycopg2
from psycopg2 import sql
import json
warnings.filterwarnings('ignore')

# ==================== API CONFIGURATION ====================
API_KEY = "633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49"

API_ENDPOINTS = [
    {"url": "https://api.football-data-api.com/match", "param": "match_id"},
    {"url": "https://api.footystats.org/match", "param": "id"},
    {"url": "https://api.footystats.org/match", "param": "match_id"},
]

# ==================== DATABASE CONFIGURATION ====================
DB_CONFIG = {
    'host': 'winbets-db.postgres.database.azure.com',
    'port': 5432,
    'database': 'postgres',
    'user': 'winbets',
    'password': 'deeptanshu@123'
}

TABLE_NAME = 'agility_football_v2'

print("="*80)
print("FOOTBALL PREDICTIONS VALIDATION")
print("="*80)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ==================== DATABASE CONNECTION ====================
print(f"\n[1/4] Connecting to database...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print(f"✓ Connected to {DB_CONFIG['database']}")
    print(f"✓ Table: {TABLE_NAME}")
except Exception as e:
    print(f"✗ Connection error: {e}")
    exit(1)

# ==================== LOAD PENDING PREDICTIONS ====================
print(f"\n[2/4] Loading pending predictions from database...")
try:
    query = sql.SQL("""
        SELECT match_id, home_team, away_team,
               ou_prediction, ou_probability, over_2_5_odds, under_2_5_odds,
               ml_prediction, ml_probability, home_win_odds, away_win_odds
        FROM {}
        WHERE home_goals IS NULL OR away_goals IS NULL
        LIMIT 100
    """).format(sql.Identifier(TABLE_NAME))
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    predictions_df = pd.DataFrame(rows, columns=columns)
    
    print(f"✓ Loaded {len(predictions_df)} pending predictions")
    
    if len(predictions_df) == 0:
        print("\n✓ No pending predictions to validate")
        cursor.close()
        conn.close()
        exit(0)
    
except Exception as e:
    print(f"✗ Error loading predictions: {e}")
    cursor.close()
    conn.close()
    exit(1)

# ==================== TEST API ====================
print(f"\n[3/4] Testing API configuration...")
working_config = None
test_match_id = predictions_df.iloc[0]['match_id']

print(f"Testing with match ID: {test_match_id}\n")

for i, config in enumerate(API_ENDPOINTS, 1):
    try:
        print(f"  [{i}/{len(API_ENDPOINTS)}] Testing {config['url']}...", end=" ")
        response = requests.get(
            config['url'],
            params={'key': API_KEY, config['param']: test_match_id},
            timeout=15
        )
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('success') and data.get('data'):
                    print("✓ WORKING")
                    working_config = config
                    break
                else:
                    print("✗ No data")
            except:
                print("✗ JSON error")
        else:
            print(f"✗ HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ {str(e)[:30]}")
    
    time.sleep(0.3)

if not working_config:
    print(f"\n✗ ERROR: No working API found")
    cursor.close()
    conn.close()
    exit(1)

print(f"\n✓ Using: {working_config['url']}")

# ==================== VALIDATE PREDICTIONS ====================
print(f"\n[4/4] Validating predictions...")
print("="*80)

validated = 0
failed = 0

for idx, row in predictions_df.iterrows():
    match_id = row['match_id']
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Get prediction data
    ou_pred = row['ou_prediction']  # "Over 2.5" or "Under 2.5"
    ml_pred = row['ml_prediction']  # "Home Win" or "Away Win"
    
    # Get odds data
    over_odds = row['over_2_5_odds']
    under_odds = row['under_2_5_odds']
    home_odds = row['home_win_odds']
    away_odds = row['away_win_odds']
    
    try:
        # Fetch match result
        response = requests.get(
            working_config['url'],
            params={'key': API_KEY, working_config['param']: match_id},
            timeout=15
        )
        
        if response.status_code != 200:
            print(f"✗ {match_id}: HTTP {response.status_code}")
            failed += 1
            continue
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"✗ {match_id}: Invalid JSON")
            failed += 1
            continue
        
        if not data.get('success') or not data.get('data'):
            print(f"⏳ {match_id}: No data yet")
            failed += 1
            continue
        
        match_data = data['data']
        status = match_data.get('status', '')
        
        # Check if match is complete
        if status != 'complete':
            print(f"⏳ {match_id}: Status {status}")
            failed += 1
            continue
        
        # Get actual scores
        home_goals = int(match_data.get('homeGoalCount', 0))
        away_goals = int(match_data.get('awayGoalCount', 0))
        total_goals = home_goals + away_goals
        
        # Determine actual O/U
        ou_actual = "OVER 2.5" if total_goals > 2.5 else "UNDER 2.5"
        
        # Determine actual winner
        if home_goals > away_goals:
            ml_actual = "Home Win"
        elif away_goals > home_goals:
            ml_actual = "Away Win"
        else:
            ml_actual = "Draw"
        
        # Calculate O/U correctness
        ou_correct = (ou_pred == ou_actual)
        
        # Calculate O/U P&L
        if "Over" in ou_pred:
            # Predicted Over
            ou_pnl = over_odds - 1 if ou_actual == "OVER 2.5" else -1.0
        else:
            # Predicted Under
            ou_pnl = under_odds - 1 if ou_actual == "UNDER 2.5" else -1.0
        
        # Calculate ML correctness
        ml_correct = (ml_pred == ml_actual)
        
        # Calculate ML P&L
        if ml_pred == "Home Win":
            ml_pnl = home_odds - 1 if ml_actual == "Home Win" else -1.0
        elif ml_pred == "Away Win":
            ml_pnl = away_odds - 1 if ml_actual == "Away Win" else -1.0
        else:
            ml_pnl = 0.0
        
        # Update database
        update_query = sql.SQL("""
            UPDATE {}
            SET 
                home_goals = %s,
                away_goals = %s,
                total_goals = %s,
                ou_actual = %s,
                ou_correct = %s,
                ou_pnl = %s,
                ml_actual = %s,
                ml_correct = %s,
                ml_pnl = %s
            WHERE match_id = %s
        """).format(sql.Identifier(TABLE_NAME))
        
        cursor.execute(update_query, (
            float(home_goals),
            float(away_goals),
            float(total_goals),
            ou_actual,
            ou_correct,
            round(ou_pnl, 2),
            ml_actual,
            ml_correct,
            round(ml_pnl, 2),
            match_id
        ))
        
        conn.commit()
        validated += 1
        
        print(f"✓ {match_id}: {home_team} {home_goals}-{away_goals} {away_team}")
        print(f"  O/U: {ou_pred} → {ou_actual} {'✓' if ou_correct else '✗'} | P/L: ${ou_pnl:.2f}")
        print(f"  ML: {ml_pred} → {ml_actual} {'✓' if ml_correct else '✗'} | P/L: ${ml_pnl:.2f}")
        
    except Exception as e:
        print(f"✗ {match_id}: {str(e)[:60]}")
        failed += 1
    
    time.sleep(0.25)

# ==================== SUMMARY ====================
print(f"\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print(f"✓ Successfully validated: {validated} matches")
print(f"✗ Failed/Pending: {failed} matches")

# ==================== STATS ====================
try:
    cursor.execute(sql.SQL("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN ou_correct THEN 1 ELSE 0 END) as ou_wins,
            SUM(CASE WHEN ml_correct THEN 1 ELSE 0 END) as ml_wins,
            ROUND(SUM(ou_pnl)::numeric, 2) as ou_total_pnl,
            ROUND(SUM(ml_pnl)::numeric, 2) as ml_total_pnl
        FROM {}
        WHERE home_goals IS NOT NULL
    """).format(sql.Identifier(TABLE_NAME)))
    
    stats = cursor.fetchone()
    if stats and stats[0] > 0:
        total, ou_wins, ml_wins, ou_pnl_total, ml_pnl_total = stats
        print(f"\nDatabase Statistics:")
        print(f"  Total validated: {total}")
        print(f"  O/U Wins: {ou_wins or 0} ({(ou_wins or 0)/total*100:.1f}%)")
        print(f"  ML Wins: {ml_wins or 0} ({(ml_wins or 0)/total*100:.1f}%)")
        print(f"  O/U Total P&L: ${ou_pnl_total or 0:.2f}")
        print(f"  ML Total P&L: ${ml_pnl_total or 0:.2f}")

except Exception as e:
    print(f"⚠ Could not fetch stats: {e}")

cursor.close()
conn.close()
print(f"\n✓ Database connection closed")

print(f"\n" + "="*80)
print("✅ VALIDATION COMPLETE!")
print("="*80)

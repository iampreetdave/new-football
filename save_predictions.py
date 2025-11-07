"""
Save Football Predictions to PostgreSQL Database
Reads predictions_output.csv and inserts into agility_fotball_v2 table
- Checks for existing match_ids to avoid duplicates
- Handles NULL values properly
- Simple and straightforward storage
"""

import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import datetime
import sys

# ==================== DATABASE CONFIGURATION ====================
DB_CONFIG = {
    'host': 'winbets-db.postgres.database.azure.com',
    'port': 5432,
    'database': 'postgres',
    'user': 'winbets',
    'password': 'deeptanshu@123'
}

TABLE_NAME = 'agility_fotball_v2'
SCHEMA_NAME = 'public'
CSV_FILE = 'predictions_output.csv'

print("="*80)
print("SAVING PREDICTIONS TO DATABASE")
print("="*80)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ==================== LOAD CSV DATA ====================
print(f"\n[1/4] Loading CSV file: {CSV_FILE}")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} records from CSV")
    print(f"  Columns: {len(df.columns)}")
    
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    sys.exit(1)

# ==================== CONNECT TO DATABASE ====================
print(f"\n[2/4] Connecting to database...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✓ Connected to database")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Database: {DB_CONFIG['database']}")
except Exception as e:
    print(f"✗ Connection error: {e}")
    sys.exit(1)

# ==================== CHECK FOR EXISTING RECORDS ====================
print(f"\n[3/4] Checking for existing records...")
try:
    cursor.execute(sql.SQL("SELECT match_id FROM {}.{}").format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(TABLE_NAME)
    ))
    existing_ids = set([row[0] for row in cursor.fetchall()])
    print(f"✓ Found {len(existing_ids)} existing records in database")
except Exception as e:
    print(f"✗ Error querying existing records: {e}")
    cursor.close()
    conn.close()
    sys.exit(1)

# Filter out existing records
new_data = df[~df['match_id'].isin(existing_ids)].copy()
duplicate_count = len(df) - len(new_data)

print(f"\n  Records breakdown:")
print(f"    • Total in CSV: {len(df)}")
print(f"    • Already in DB: {duplicate_count}")
print(f"    • New to insert: {len(new_data)}")

if len(new_data) == 0:
    print("\n✓ All records already exist in database. Nothing to insert.")
    cursor.close()
    conn.close()
    print("\n" + "="*80)
    print("✅ SAVE COMPLETE - NO NEW RECORDS")
    print("="*80)
    sys.exit(0)

# ==================== INSERT NEW RECORDS ====================
print(f"\n[4/4] Inserting {len(new_data)} new records...")

insert_query = sql.SQL("""
    INSERT INTO {}.{} (
        match_id, home_id, away_id, league_id, date, league,
        home_team, away_team, ou_prediction, ou_probability,
        over_2_5_odds, under_2_5_odds, ml_prediction, ml_probability,
        home_win_odds, away_win_odds, ou_confidence, ml_confidence,
        ou_confidence_level, ml_confidence_level
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
""").format(sql.Identifier(SCHEMA_NAME), sql.Identifier(TABLE_NAME))

inserted = 0
errors = 0
error_details = []

for idx, row in new_data.iterrows():
    try:
        # Replace NaN with None for proper NULL handling
        values = [
            row['match_id'],
            row['home_id'],
            row['away_id'],
            row['league_id'],
            row['date'],
            row['league'],
            row['home_team'],
            row['away_team'],
            row['ou_prediction'],
            row['ou_probability'],
            row['over_2_5_odds'],
            row['under_2_5_odds'],
            row['ml_prediction'],
            row['ml_probability'],
            row['home_win_odds'],
            row['away_win_odds'],
            row['ou_confidence'],
            row['ml_confidence'],
            row['ou_confidence_level'],
            row['ml_confidence_level']
        ]
        
        # Convert NaN to None
        values = [None if pd.isna(v) else v for v in values]
        
        cursor.execute(insert_query, values)
        inserted += 1
        
        # Commit every 10 records
        if inserted % 10 == 0:
            conn.commit()
            print(f"  Progress: {inserted}/{len(new_data)} records inserted...")
            
    except Exception as e:
        errors += 1
        error_msg = f"Match ID {row['match_id']}: {str(e)[:100]}"
        error_details.append(error_msg)
        print(f"  ⚠ Error: {error_msg}")
        conn.rollback()

# Final commit
try:
    conn.commit()
    print(f"\n✓ All records committed to database")
except Exception as e:
    print(f"\n✗ Error committing: {e}")
    conn.rollback()

# ==================== SUMMARY ====================
print(f"\n" + "="*80)
print("INSERTION SUMMARY")
print("="*80)
print(f"✓ Successfully inserted: {inserted} records")
if errors > 0:
    print(f"⚠ Errors encountered: {errors} records")
    if error_details:
        print(f"\nError details:")
        for i, error in enumerate(error_details[:3], 1):
            print(f"  {i}. {error}")
        if len(error_details) > 3:
            print(f"  ... and {len(error_details) - 3} more errors")

# ==================== VERIFY ====================
print(f"\n" + "="*80)
print("DATABASE STATE")
print("="*80)

try:
    cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(TABLE_NAME)
    ))
    total = cursor.fetchone()[0]
    print(f"✓ Total records in database: {total}")
    
except Exception as e:
    print(f"⚠ Could not retrieve count: {e}")

# Close connection
cursor.close()
conn.close()
print(f"✓ Database connection closed")

print("\n" + "="*80)
print("✅ SAVE COMPLETE!")
print("="*80)

"""
Update Over/Under Predictions in predictions_soccer_v3_ourmodel
====================================================

Reads predictions_output.csv and updates ONLY these columns:
- predicted_over_under (from CSV: ou_prediction)
- ou_confidence (from CSV: ou_confidence)

Matches records by match_id and updates existing records.
No new records are inserted, only updates existing ones.

Database: winbets-predictions.postgres.database.azure.com
Table: predictions_soccer_v3_ourmodel
"""

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
from datetime import datetime
import sys
import os

# ==================== DATABASE CONFIGURATION ====================
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'winbets-predictions.postgres.database.azure.com'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_DATABASE', 'postgres'),
    'user': os.getenv('DB_USER', 'winbets'),
    'password': os.getenv('DB_PASSWORD', 'Constantinople@1900')
}

TABLE_NAME = 'predictions_soccer_v3_ourmodel'
CSV_FILE = 'predictions_output.csv'
BATCH_SIZE = 100

print("="*80)
print("UPDATE OVER/UNDER PREDICTIONS IN predictions_soccer_v3_ourmodel")
print("="*80)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Table: {TABLE_NAME}")
print(f"CSV File: {CSV_FILE}")
print(f"\nColumns to update:")
print(f"  • predicted_over_under (from CSV: ou_prediction)")
print(f"  • ou_confidence (from CSV: ou_confidence)")

# ==================== LOAD CSV DATA ====================
print(f"\n[1/4] Loading CSV file: {CSV_FILE}")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} records from CSV")
    print(f"  Columns found: {len(df.columns)}")
    
    # Keep only required columns
    df = df[['match_id', 'ou_prediction', 'ou_confidence']].copy()
    df.columns = ['match_id', 'predicted_over_under', 'ou_confidence']
    
    print(f"✓ Extracted columns:")
    print(f"  • match_id")
    print(f"  • predicted_over_under (from ou_prediction)")
    print(f"  • ou_confidence")
    
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    sys.exit(1)

# ==================== CONNECT TO DATABASE ====================
print(f"\n[2/4] Connecting to database...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print(f"✓ Connected to database")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Database: {DB_CONFIG['database']}")
except Exception as e:
    print(f"✗ Connection error: {e}")
    sys.exit(1)

# ==================== CHECK MATCHING RECORDS ====================
print(f"\n[3/4] Checking for matching records in {TABLE_NAME}...")
try:
    # Get all match_ids from database
    cursor.execute(sql.SQL("SELECT match_id FROM {}").format(
        sql.Identifier(TABLE_NAME)
    ))
    db_ids = set([row[0] for row in cursor.fetchall()])
    print(f"✓ Found {len(db_ids)} total records in {TABLE_NAME}")
    
    # Check which CSV records exist in database
    csv_ids = set(df['match_id'].astype(float))
    matching_ids = csv_ids.intersection(db_ids)
    new_ids = csv_ids - db_ids
    
    print(f"\n  Record breakdown:")
    print(f"    • Total in CSV: {len(csv_ids)}")
    print(f"    • Matching in DB: {len(matching_ids)}")
    print(f"    • Not in DB: {len(new_ids)}")
    
    if len(matching_ids) == 0:
        print(f"\n✗ No matching records found. Nothing to update.")
        cursor.close()
        conn.close()
        sys.exit(1)
    
    # Filter to only matching records
    df_to_update = df[df['match_id'].isin(matching_ids)].copy()
    print(f"\n✓ Will update {len(df_to_update)} records")
    
except Exception as e:
    print(f"✗ Error checking records: {e}")
    cursor.close()
    conn.close()
    sys.exit(1)

# ==================== UPDATE RECORDS ====================
print(f"\n[4/4] Updating {len(df_to_update)} records in {TABLE_NAME}...")

update_query = f"""
UPDATE {TABLE_NAME}
SET 
    predicted_over_under = %s,
    ou_confidence = %s
WHERE match_id = %s
"""

# Prepare data for batch update
update_data = [
    (
        row['predicted_over_under'],
        float(row['ou_confidence']) if pd.notna(row['ou_confidence']) else None,
        float(row['match_id'])
    )
    for _, row in df_to_update.iterrows()
]

try:
    # Execute batch update
    execute_batch(cursor, update_query, update_data, page_size=BATCH_SIZE)
    conn.commit()
    print(f"✓ Successfully updated {len(df_to_update)} records")
    
except Exception as e:
    conn.rollback()
    print(f"✗ Error updating records: {e}")
    cursor.close()
    conn.close()
    sys.exit(1)

# ==================== VERIFY UPDATES ====================
print(f"\nVerifying updates...")
try:
    # Check a few updated records
    cursor.execute(f"""
        SELECT match_id, predicted_over_under, ou_confidence
        FROM {TABLE_NAME}
        WHERE match_id IN ({','.join([str(int(mid)) for mid in df_to_update['match_id'].head(3)])})
    """)
    
    results = cursor.fetchall()
    if results:
        print(f"✓ Sample updated records:")
        for match_id, ou_pred, ou_conf in results:
            print(f"  • Match {int(match_id)}: {ou_pred}, Confidence: {ou_conf}%")
    
    # Get overall counts
    cursor.execute(f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN predicted_over_under IS NOT NULL THEN 1 END) as with_ou_pred,
            COUNT(CASE WHEN ou_confidence IS NOT NULL THEN 1 END) as with_ou_conf
        FROM {TABLE_NAME}
    """)
    
    total, with_ou_pred, with_ou_conf = cursor.fetchone()
    print(f"\n✓ Table statistics:")
    print(f"  • Total records: {total}")
    print(f"  • With predicted_over_under: {with_ou_pred}")
    print(f"  • With ou_confidence: {with_ou_conf}")
    
except Exception as e:
    print(f"⚠ Could not verify updates: {e}")

# Close connection
cursor.close()
conn.close()

# ==================== FINAL SUMMARY ====================
print(f"\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"✓ Successfully updated {len(df_to_update)} records in {TABLE_NAME}")
print(f"  • Column: predicted_over_under ← ou_prediction")
print(f"  • Column: ou_confidence ← ou_confidence")
print(f"\n✓ Records matched by: match_id")
print("="*80 + "\n")

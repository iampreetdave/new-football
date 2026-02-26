"""
Save Football Predictions to PostgreSQL Database
Reads predictions_output.csv and inserts into predictions_soccer_v2_ourmodel table
Syncs to BOTH databases (PRIMARY + WINBETS)
- Checks for existing match_ids to avoid duplicates
- Maps league_id to league_name and stores in league column
- Calculates ou_grade and ml_grade from confidence values
- Sets status to PENDING for all predictions
- Handles NULL values properly
- Simple and straightforward storage
"""

import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import datetime
import sys
import os

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

TABLE_NAME = 'predictions_soccer_v2_ourmodel'
CSV_FILE = 'predictions_output.csv'

# ==================== LEAGUE ID MAPPING ====================
LEAGUE_MAPPING = {
    12325: "England Premier League",
    15050: "England Premier League",
    
    14924: "UEFA Champions League",
    12316: "Spain La Liga",
    14956: "Spain La Liga",
    12530: "Italy Serie A",
    15068: "Italy Serie A",
    12529: "Germany Bundesliga",
    14968: "Germany Bundesliga",
    13973: "USA MLS",
    12337: "France Ligue 1",
    14932: "France Ligue 1",
    12322: "Netherlands Eredivisie",
    14936: "Netherlands Eredivisie",
    15115: "Portugal Liga NOS",

    12136: "Mexico Liga MX",
    15234: "Mexico Liga MX"
}

def get_league_name(league_id):
    """Get league name from league_id using the mapping"""
    try:
        league_id_int = int(league_id)
        return LEAGUE_MAPPING.get(league_id_int, "Unknown League")
    except:
        return "Unknown League"

def get_ou_grade(confidence):
    """Convert ou_confidence value (0-100) to letter grade based on ROI thresholds"""
    if pd.isna(confidence):
        return 'C'
    try:
        conf = float(confidence)
        if conf >= 78.0:
            return "A"
        elif conf >= 65.7:
            return "B"
        elif conf >= 35.7:
            return "D"
        else:
            return "C"
    except:
        return 'C'

def get_ml_grade(confidence):
    """Convert confidence value (0-100) to letter grade"""
    if pd.isna(confidence):
        return None
    try:
        conf = float(confidence)
        if conf >= 90:
            return "A+"
        elif conf >= 85:
            return "A"
        elif conf >= 80:
            return "A-"
        elif conf >= 75:
            return "B+"
        elif conf >= 70:
            return "B"
        elif conf >= 65:
            return "B-"
        elif conf >= 60:
            return "C+"
        elif conf >= 55:
            return "C"
        elif conf >= 50:
            return "C-"
        else:
            return "D"
    except:
        return None

print("="*80)
print("SAVING PREDICTIONS TO DATABASE - V2 TABLE - DUAL DATABASE")
print("="*80)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ==================== LOAD CSV DATA ====================
print(f"\n[1/4] Loading CSV file: {CSV_FILE}")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} records from CSV")
    print(f"  Columns: {len(df.columns)}")
    
    # Map league_id to league_name
    df['league_name'] = df['league_id'].apply(get_league_name)
    print(f"✓ Mapped league names from league IDs")
    
    # Calculate grades from confidence values
    df['ou_grade'] = df['ou_confidence'].apply(get_ou_grade)
    df['ml_grade'] = df['ml_confidence'].apply(get_ml_grade)
    print(f"✓ Calculated ou_grade and ml_grade from confidence values")
    
    # Set status to PENDING for all predictions
    df['status'] = 'PENDING'
    print(f"✓ Set status to PENDING for all records")
    
    # Parse and format date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    print(f"✓ Parsed dates from CSV")
    
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    sys.exit(1)

# ==================== HELPER FUNCTION FOR DATABASE OPERATIONS ====================
def insert_to_database(db_config, db_name, df):
    """Insert data to a specific database"""
    print(f"\n{'='*80}")
    print(f"Processing {db_name}")
    print(f"{'='*80}")
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        print(f"✓ Connected to {db_name}")
        print(f"  Host: {db_config['host']}")
        print(f"  Database: {db_config['database']}")
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

    # ==================== CHECK FOR EXISTING RECORDS ====================
    print(f"\n[3/4] Checking for existing records in {db_name}...")
    try:
        cursor.execute(sql.SQL("SELECT match_id FROM {}").format(
            sql.Identifier(TABLE_NAME)
        ))
        existing_ids = set([row[0] for row in cursor.fetchall()])
        print(f"✓ Found {len(existing_ids)} existing records in {db_name}")
    except Exception as e:
        print(f"✗ Error querying existing records: {e}")
        cursor.close()
        conn.close()
        return False

    # Filter out existing records
    new_data = df[~df['match_id'].isin(existing_ids)].copy()
    duplicate_count = len(df) - len(new_data)

    print(f"\n  Records breakdown:")
    print(f"    • Total in CSV: {len(df)}")
    print(f"    • Already in DB: {duplicate_count}")
    print(f"    • New to insert: {len(new_data)}")

    if len(new_data) == 0:
        print(f"\n✓ All records already exist in {db_name}. Nothing to insert.")
        cursor.close()
        conn.close()
        return True

    # ==================== INSERT NEW RECORDS ====================
    print(f"\n[4/4] Inserting {len(new_data)} new records to {db_name}...")

    insert_query = sql.SQL("""
        INSERT INTO {} (
            match_id, home_id, away_id, league_id, league, date,
            home_team, away_team, ou_prediction, ou_probability,
            over_2_5_odds, under_2_5_odds, ml_prediction, ml_probability,
            home_win_odds, away_win_odds, ou_confidence, ml_confidence,
            ou_confidence_level, ml_confidence_level, ou_grade, ml_grade, status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s
        )
    """).format(sql.Identifier(TABLE_NAME))

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
                row['league_name'],
                row['date'],
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
                row['ml_confidence_level'],
                row['ou_grade'],
                row['ml_grade'],
                row['status']
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
        print(f"\n✓ All records committed to {db_name}")
    except Exception as e:
        print(f"\n✗ Error committing: {e}")
        conn.rollback()

    # ==================== SUMMARY ====================
    print(f"\n" + "="*80)
    print(f"INSERTION SUMMARY - {db_name}")
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
    try:
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(
            sql.Identifier(TABLE_NAME)
        ))
        total = cursor.fetchone()[0]
        print(f"\n✓ Total records in {db_name}: {total}")
        
    except Exception as e:
        print(f"⚠ Could not retrieve count: {e}")

    # Close connection
    cursor.close()
    conn.close()
    print(f"✓ {db_name} connection closed")
    
    return True

# ==================== EXECUTE INSERTS TO BOTH DATABASES ====================
print("\n" + "="*80)
print("DUAL DATABASE INSERT PROCESS")
print("="*80)

success_primary = insert_to_database(DB_CONFIG, "PRIMARY (Old Credentials)", df)
success_winbets = insert_to_database(DB_CONFIG_WINBETS, "WINBETS (New Credentials)", df)

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
if success_primary and success_winbets:
    print("✅ SUCCESS! Data saved to BOTH databases")
    print("  ✓ Primary database (old credentials)")
    print("  ✓ WINBETS database (new credentials)")
elif success_primary:
    print("⚠️ PRIMARY database OK, but WINBETS database FAILED")
elif success_winbets:
    print("⚠️ WINBETS database OK, but PRIMARY database FAILED")
else:
    print("❌ Both databases FAILED")

print("\n" + "="*80)
print("✅ SAVE COMPLETE!")
print("="*80)

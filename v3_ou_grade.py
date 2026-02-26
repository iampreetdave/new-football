import psycopg2
from psycopg2 import sql
import os
from datetime import datetime

# ==================== DATABASE CONFIGURATION ====================
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_DATABASE'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

TABLE_NAME = 'predictions_soccer_v3_ourmodel'

print("\n" + "="*60)
print("OU GRADE UPDATE SCRIPT")
print("="*60)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ==================== GRADING LOGIC ====================
def get_ou_grade(confidence):
    if confidence is None:
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

# ==================== CONNECT & FETCH ====================
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✓ Connected to PRIMARY database")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Fetch rows where ou_grade is NULL
cursor.execute(sql.SQL("""
    SELECT match_id, ou_confidence
    FROM {}
    WHERE ou_grade IS NULL
""").format(sql.Identifier(TABLE_NAME)))

rows = cursor.fetchall()
print(f"✓ Found {len(rows)} rows with NULL ou_grade")

if len(rows) == 0:
    print("ℹ Nothing to update.")
    conn.close()
    exit(0)

# ==================== UPDATE GRADES ====================
updated = 0
grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

for match_id, ou_confidence in rows:
    grade = get_ou_grade(ou_confidence)
    grade_counts[grade] += 1

    try:
        cursor.execute(sql.SQL("""
            UPDATE {}
            SET ou_grade = %s
            WHERE match_id = %s
        """).format(sql.Identifier(TABLE_NAME)), (grade, match_id))
        updated += 1
    except Exception as e:
        print(f"⚠ Error updating {match_id}: {e}")
        conn.rollback()

conn.commit()

# ==================== SUMMARY ====================
print(f"\n✓ Updated {updated}/{len(rows)} rows")
print(f"  A (≥78.0): {grade_counts['A']}")
print(f"  B (≥65.7): {grade_counts['B']}")
print(f"  D (≥35.7): {grade_counts['D']}")
print(f"  C (below):  {grade_counts['C']}")

cursor.close()
conn.close()
print("\n✅ Done!")

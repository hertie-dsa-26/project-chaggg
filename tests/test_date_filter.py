"""
Test script to verify the date filter works correctly.
Tests the API date filter without downloading the full dataset.

Usage:
    python test_date_filter.py
"""
import requests
import sys
import os

# Add parent directory to path so we can import from scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scripts.config import APP_TOKEN, BASE_URL
except ImportError:
    print("Error: Could not import from scripts.config")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

END_DATE = "2025-12-31T23:59:59"

def test_recent_records():
    """Test 1: Fetch most recent records and verify they're before cutoff."""
    print("=" * 70)
    print("TEST 1: Checking most recent records")
    print("=" * 70)
    
    params = {
        "$limit": 10,
        "$offset": 0,
        "$order": "date DESC",  # Get most recent records
        "$where": f"date <= '{END_DATE}'",
        "$$app_token": APP_TOKEN
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("⚠️  No records returned")
            return False
        
        print(f"✓ Fetched {len(data)} records")
        print(f"  Most recent date: {data[0]['date']}")
        print(f"  Oldest in batch:  {data[-1]['date']}")
        
        # Check if any records are after 2025-12-31
        violations = []
        for record in data:
            if record['date'] > "2026-01-01T00:00:00.000":
                violations.append(record['date'])
        
        if violations:
            print(f"❌ FAILED: Found {len(violations)} records after 2025-12-31:")
            for date in violations:
                print(f"   - {date}")
            return False
        else:
            print("✓ PASSED: All records are on or before 2025-12-31")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_record_counts():
    """Test 2: Compare total records with and without filter."""
    print("\n" + "=" * 70)
    print("TEST 2: Comparing record counts")
    print("=" * 70)
    
    try:
        # Count without filter
        print("Counting all records...")
        r1 = requests.get(BASE_URL, params={
            "$select": "COUNT(*)",
            "$$app_token": APP_TOKEN
        }, timeout=60)
        r1.raise_for_status()
        total_all = int(r1.json()[0]['COUNT'])
        
        # Count with filter
        print("Counting records ≤ 2025-12-31...")
        r2 = requests.get(BASE_URL, params={
            "$select": "COUNT(*)",
            "$where": f"date <= '{END_DATE}'",
            "$$app_token": APP_TOKEN
        }, timeout=60)
        r2.raise_for_status()
        total_filtered = int(r2.json()[0]['COUNT'])
        
        print(f"\n  Total records (all time):      {total_all:,}")
        print(f"  Total records (≤ 2025-12-31):  {total_filtered:,}")
        print(f"  Difference (2026+ records):    {total_all - total_filtered:,}")
        
        if total_filtered < total_all:
            print(f"\n✓ PASSED: Filter is working ({total_all - total_filtered:,} records excluded)")
            return True
        elif total_filtered == total_all:
            print("\n⚠️  WARNING: No 2026 records found (database may not have 2026 data yet)")
            return True
        else:
            print("\n❌ FAILED: Filtered count is higher than total (shouldn't happen)")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_sample_batch():
    """Test 3: Fetch a sample batch as the fetch.py script would."""
    print("\n" + "=" * 70)
    print("TEST 3: Testing actual fetch_batch behavior")
    print("=" * 70)
    
    params = {
        "$limit": 100,  # Small batch for testing
        "$offset": 0,
        "$order": "date ASC",  # Same as fetch.py
        "$where": f"date <= '{END_DATE}'",
        "$$app_token": APP_TOKEN
    }
    
    try:
        print("Fetching sample batch (100 records)...")
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("⚠️  No records returned")
            return False
        
        dates = [r['date'] for r in data]
        print(f"✓ Fetched {len(data)} records")
        print(f"  Date range: {min(dates)} to {max(dates)}")
        
        # Check for violations
        violations = [d for d in dates if d > '2025-12-31']
        
        if violations:
            print(f"❌ FAILED: Found {len(violations)} records after 2025-12-31")
            return False
        else:
            print("✓ PASSED: All records within date range")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CHICAGO CRIME DATA - DATE FILTER TEST")
    print("=" * 70)
    print(f"Testing filter: date <= '{END_DATE}'")
    print()
    
    if APP_TOKEN == "YOUR_APP_TOKEN_HERE":
        print("⚠️  WARNING: Using default APP_TOKEN")
        print("   The API may work without a token, but it's better to set one.")
        print()
    
    results = []
    
    # Run all tests
    results.append(("Recent Records Test", test_recent_records()))
    results.append(("Record Count Test", test_record_counts()))
    results.append(("Sample Batch Test", test_sample_batch()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status:10} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("🎉 All tests passed! The date filter is working correctly.")
        print("   You can now run fetch.py with the updated filter.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("   You may need to adjust the date filter format.")
    
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

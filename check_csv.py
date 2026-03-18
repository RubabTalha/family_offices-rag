import pandas as pd

# Try to read the CSV
try:
    df = pd.read_csv('family_offices.csv')
    print(f"✅ CSV loaded successfully!")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. '{col}'")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
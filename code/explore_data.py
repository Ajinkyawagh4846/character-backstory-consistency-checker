import pandas as pd
import os
from pathlib import Path

# Set up paths
TRAIN_CSV = '../train.csv'
TEST_CSV = '../test.csv'
BOOKS_DIR = '../books'
RESULTS_DIR = '../results'
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'dataset_summary.txt')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize summary content
summary_lines = []

def print_and_save(text, emoji=''):
    """Print to console and save to summary file"""
    output = f"{emoji} {text}" if emoji else text
    print(output)
    summary_lines.append(output)

print_and_save("=" * 80)
print_and_save("📊 DATASET EXPLORATION REPORT", "📊")
print_and_save("=" * 80)
print_and_save("")

# ============================================================================
# 1. Load CSV files
# ============================================================================
print_and_save("=" * 80)
print_and_save("1️⃣  LOADING DATASETS", "1️⃣")
print_and_save("=" * 80)

try:
    train_df = pd.read_csv(TRAIN_CSV)
    print_and_save(f"✅ Successfully loaded train.csv")
except Exception as e:
    print_and_save(f"❌ Error loading train.csv: {e}")
    train_df = None

try:
    test_df = pd.read_csv(TEST_CSV)
    print_and_save(f"✅ Successfully loaded test.csv")
except Exception as e:
    print_and_save(f"❌ Error loading test.csv: {e}")
    test_df = None

print_and_save("")

# ============================================================================
# 2. Dataset Statistics
# ============================================================================
print_and_save("=" * 80)
print_and_save("2️⃣  DATASET STATISTICS", "2️⃣")
print_and_save("=" * 80)

if train_df is not None:
    print_and_save("")
    print_and_save("📋 TRAIN.CSV", "📋")
    print_and_save("-" * 80)
    print_and_save(f"Number of rows: {len(train_df):,}")
    print_and_save(f"Number of columns: {len(train_df.columns)}")
    print_and_save("")
    
    print_and_save("Column names:")
    for i, col in enumerate(train_df.columns, 1):
        print_and_save(f"  {i}. {col}")
    print_and_save("")
    
    print_and_save("Data types:")
    for col, dtype in train_df.dtypes.items():
        print_and_save(f"  {col}: {dtype}")
    print_and_save("")
    
    print_and_save("Missing values:")
    missing = train_df.isnull().sum()
    if missing.sum() == 0:
        print_and_save("  ✅ No missing values found!")
    else:
        for col, count in missing.items():
            if count > 0:
                print_and_save(f"  {col}: {count} ({count/len(train_df)*100:.2f}%)")
    print_and_save("")
    
    print_and_save("First 5 rows:")
    print_and_save("-" * 80)
    df_head_str = train_df.head().to_string()
    print(df_head_str)
    summary_lines.append(df_head_str)
    print_and_save("")

if test_df is not None:
    print_and_save("")
    print_and_save("📋 TEST.CSV", "📋")
    print_and_save("-" * 80)
    print_and_save(f"Number of rows: {len(test_df):,}")
    print_and_save(f"Number of columns: {len(test_df.columns)}")
    print_and_save("")
    
    print_and_save("Column names:")
    for i, col in enumerate(test_df.columns, 1):
        print_and_save(f"  {i}. {col}")
    print_and_save("")
    
    print_and_save("Data types:")
    for col, dtype in test_df.dtypes.items():
        print_and_save(f"  {col}: {dtype}")
    print_and_save("")
    
    print_and_save("Missing values:")
    missing = test_df.isnull().sum()
    if missing.sum() == 0:
        print_and_save("  ✅ No missing values found!")
    else:
        for col, count in missing.items():
            if count > 0:
                print_and_save(f"  {col}: {count} ({count/len(test_df)*100:.2f}%)")
    print_and_save("")
    
    print_and_save("First 5 rows:")
    print_and_save("-" * 80)
    df_head_str = test_df.head().to_string()
    print(df_head_str)
    summary_lines.append(df_head_str)
    print_and_save("")

# ============================================================================
# 3. Books Information
# ============================================================================
print_and_save("=" * 80)
print_and_save("3️⃣  BOOKS IN BOOKS/ FOLDER", "3️⃣")
print_and_save("=" * 80)

books_info = []
if os.path.exists(BOOKS_DIR):
    book_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith('.txt')]
    book_files.sort()
    
    print_and_save(f"\nFound {len(book_files)} book file(s):\n")
    
    for book_file in book_files:
        file_path = os.path.join(BOOKS_DIR, book_file)
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        file_size_kb = file_size / 1024
        
        if file_size_mb >= 1:
            size_str = f"{file_size_mb:.2f} MB"
        else:
            size_str = f"{file_size_kb:.2f} KB"
        
        books_info.append({
            'filename': book_file,
            'size_bytes': file_size,
            'size_display': size_str
        })
        
        print_and_save(f"  📚 {book_file}")
        print_and_save(f"     Size: {size_str} ({file_size:,} bytes)")
        print_and_save("")
else:
    print_and_save(f"❌ Books directory not found: {BOOKS_DIR}")

# ============================================================================
# 4. Book Usage Mapping
# ============================================================================
print_and_save("=" * 80)
print_and_save("4️⃣  BOOK USAGE MAPPING", "4️⃣")
print_and_save("=" * 80)

if train_df is not None and test_df is not None:
    print_and_save("")
    
    # Get unique books from each dataset
    train_books = set(train_df['book_name'].unique()) if 'book_name' in train_df.columns else set()
    test_books = set(test_df['book_name'].unique()) if 'book_name' in test_df.columns else set()
    
    all_books = train_books.union(test_books)
    
    print_and_save("Book distribution:")
    print_and_save("-" * 80)
    
    for book in sorted(all_books):
        in_train = "✅" if book in train_books else "❌"
        in_test = "✅" if book in test_books else "❌"
        
        train_count = len(train_df[train_df['book_name'] == book]) if book in train_books else 0
        test_count = len(test_df[test_df['book_name'] == book]) if book in test_books else 0
        
        print_and_save(f"  📖 {book}")
        print_and_save(f"     Train: {in_train} ({train_count:,} rows)")
        print_and_save(f"     Test:  {in_test} ({test_count:,} rows)")
        print_and_save("")
    
    # Summary statistics
    print_and_save("Summary:")
    print_and_save("-" * 80)
    print_and_save(f"  Total unique books: {len(all_books)}")
    print_and_save(f"  Books in train only: {len(train_books - test_books)}")
    print_and_save(f"  Books in test only: {len(test_books - train_books)}")
    print_and_save(f"  Books in both: {len(train_books & test_books)}")
    print_and_save("")
    
    if train_books - test_books:
        print_and_save("  Books only in train:")
        for book in sorted(train_books - test_books):
            print_and_save(f"    - {book}")
        print_and_save("")
    
    if test_books - train_books:
        print_and_save("  Books only in test:")
        for book in sorted(test_books - train_books):
            print_and_save(f"    - {book}")
        print_and_save("")
    
    if train_books & test_books:
        print_and_save("  Books in both train and test:")
        for book in sorted(train_books & test_books):
            print_and_save(f"    - {book}")
        print_and_save("")

# ============================================================================
# 5. Additional Statistics
# ============================================================================
print_and_save("=" * 80)
print_and_save("5️⃣  ADDITIONAL STATISTICS", "5️⃣")
print_and_save("=" * 80)

if train_df is not None:
    print_and_save("")
    print_and_save("Train.csv - Label Distribution:")
    print_and_save("-" * 80)
    if 'label' in train_df.columns:
        label_counts = train_df['label'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(train_df) * 100
            print_and_save(f"  {label}: {count:,} ({pct:.2f}%)")
    print_and_save("")
    
    print_and_save("Train.csv - Character Distribution:")
    print_and_save("-" * 80)
    if 'char' in train_df.columns:
        char_counts = train_df['char'].value_counts()
        print_and_save(f"  Total unique characters: {len(char_counts)}")
        print_and_save("  Top 10 characters:")
        for char, count in char_counts.head(10).items():
            print_and_save(f"    {char}: {count}")
    print_and_save("")

if test_df is not None:
    print_and_save("")
    print_and_save("Test.csv - Character Distribution:")
    print_and_save("-" * 80)
    if 'char' in test_df.columns:
        char_counts = test_df['char'].value_counts()
        print_and_save(f"  Total unique characters: {len(char_counts)}")
        print_and_save("  Top 10 characters:")
        for char, count in char_counts.head(10).items():
            print_and_save(f"    {char}: {count}")
    print_and_save("")

# ============================================================================
# 6. Save Summary to File
# ============================================================================
print_and_save("=" * 80)
print_and_save("6️⃣  SAVING SUMMARY", "6️⃣")
print_and_save("=" * 80)

try:
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print_and_save(f"\n✅ Summary saved to: {SUMMARY_FILE}")
except Exception as e:
    print_and_save(f"\n❌ Error saving summary: {e}")

print_and_save("")
print_and_save("=" * 80)
print_and_save("✨ EXPLORATION COMPLETE!", "✨")
print_and_save("=" * 80)


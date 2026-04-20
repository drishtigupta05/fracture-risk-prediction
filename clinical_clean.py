import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("FinallDATA/tabulardata.xlsx")

# -------------------------------
# 🔹 Helper function for ranges
# -------------------------------
def range_to_avg(val):
    if pd.isna(val):
        return np.nan
    
    val = str(val).strip()
    
    # Handle ranges like "40-45"
    if '-' in val:
        try:
            low, high = val.split('-')
            return (float(low) + float(high)) / 2
        except:
            return np.nan
    
    # Handle single numeric values
    try:
        return float(val)
    except:
        return np.nan


# -------------------------------
# 🔹 1. AGE_CATEGORY → numeric
# -------------------------------
df['AGE'] = df['AGE_CATEGORY'].apply(range_to_avg)


# -------------------------------
# 🔹 2. MENOPAUSE_YEAR cleaning
# -------------------------------
def clean_menopause(val):
    if pd.isna(val):
        return np.nan
    
    val = str(val).strip()
    
    if val.lower() in ["not applicable", "na", "none", ""]:
        return np.nan
    
    return range_to_avg(val)

df['MENOPAUSE_YEAR_CLEAN'] = df['MENOPAUSE_YEAR'].apply(clean_menopause)

# Optional binary feature
df['IS_MENOPAUSE'] = df['MENOPAUSE_YEAR_CLEAN'].notnull().astype(int)


# -------------------------------
# 🔹 3. BIRTHDATE → numeric year
# -------------------------------
def extract_birth_year(val):
    if pd.isna(val):
        return np.nan
    
    val = str(val).strip()
    
    # Handle range like "1970-1975"
    if '-' in val:
        try:
            low, high = val.split('-')
            return (int(low) + int(high)) / 2
        except:
            return np.nan
    
    # Handle single year
    try:
        return float(val)
    except:
        return np.nan

df['BIRTH_YEAR'] = df['BIRTHDATE'].apply(extract_birth_year)


# -------------------------------
# 🔹 4. Convert scan dates (YYYY-MM)
# -------------------------------
def parse_scan_date(val):
    if pd.isna(val):
        return (np.nan, np.nan)
    
    val = str(val).strip()
    
    try:
        parts = val.split('-')
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else np.nan
        return (year, month)
    except:
        return (np.nan, np.nan)


# Apply for each scan column
for col in ['SPINE_SCANDATE', 'HIP_SCANDATE', 'HIPNECK_SCANDATE']:
    df[[col + '_YEAR', col + '_MONTH']] = df[col].apply(
        lambda x: pd.Series(parse_scan_date(x))
    )


# -------------------------------
# 🔹 5. Handle missing numeric values
# -------------------------------
df = df.fillna(df.median(numeric_only=True))


# -------------------------------
# 🔹 6. Drop old messy columns (optional)
# -------------------------------
df = df.drop(columns=[
    'AGE_CATEGORY',
    'MENOPAUSE_YEAR',
    'BIRTHDATE',
    'SPINE_SCANDATE',
    'HIP_SCANDATE',
    'HIPNECK_SCANDATE'
])


# -------------------------------
# 🔹 7. Save cleaned dataset
# -------------------------------
df.to_csv("cleaned_clinical_data.csv", index=False)

print("✅ Data cleaned and saved successfully!")
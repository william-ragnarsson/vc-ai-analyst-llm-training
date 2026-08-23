import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "Plug and Play-Raw Data (1).csv")

# Output files to generate
OUTPUT_FILES = [
    os.path.join(PROJECT_ROOT, "data", "Plug and Play-Cleaned.csv"),
    os.path.join(PROJECT_ROOT, "data", "Plug and Play-Raw Data (1) Copy.csv"),
    os.path.join(PROJECT_ROOT, "data", "startups_data.csv"),
]

def clean_csv():
    with open(INPUT_CSV, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    # Columns of interest mapping from raw CSV headers
    col_indices = {
        "Name": headers.index("Name"),
        "Funding amount": headers.index("Funding amount"),
        "Team": headers.index("Team"),
        "Technology": headers.index("Technology"),
        "Market": headers.index("Market"),
        "Value Proposition": headers.index("Value Proposition"),
        "Competitive Advantage": headers.index("Competitive Advantage"),
        "Socially Impactful": headers.index("Socially Impactful"),
        "Invest William": headers.index("Invest William"),
    }

    output_headers = [
        "Name",
        "Funding amount",
        "Team",
        "Technology",
        "Market",
        "Value Proposition",
        "Competitive Advantage",
        "Socially Impactful",
        "Invest William",
    ]

    cleaned_data = [output_headers]

    for r in rows:
        row_out = []
        
        # 1. Name: normalize whitespace and newlines
        raw_name = r[col_indices["Name"]].strip() if len(r) > col_indices["Name"] else ""
        name_clean = " ".join(raw_name.split()) if raw_name else "0"
        row_out.append(name_clean)

        # 2. Funding amount: strip '$', remove decimals (e.g. '$300000.000' -> '300000')
        raw_funding = r[col_indices["Funding amount"]].strip() if len(r) > col_indices["Funding amount"] else ""
        funding_clean = raw_funding.replace("$", "").strip()
        funding_num = funding_clean.split(".")[0] if funding_clean else "0"
        row_out.append(funding_num if funding_num else "0")

        # 3. Numeric rating criteria columns
        criteria_cols = [
            "Team",
            "Technology",
            "Market",
            "Value Proposition",
            "Competitive Advantage",
            "Socially Impactful",
        ]
        for col in criteria_cols:
            idx = col_indices[col]
            val = r[idx].strip() if len(r) > idx else ""
            row_out.append(val if val else "0")

        # 4. Invest William
        idx_iw = col_indices["Invest William"]
        iw_val = r[idx_iw].strip() if len(r) > idx_iw else ""
        row_out.append(iw_val if iw_val else "0")

        cleaned_data.append(row_out)

    # Write output to all specified output paths
    for out_path in OUTPUT_FILES:
        with open(out_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(cleaned_data)
        print(f"Successfully created: {out_path} ({len(cleaned_data)-1} rows)")

if __name__ == "__main__":
    clean_csv()

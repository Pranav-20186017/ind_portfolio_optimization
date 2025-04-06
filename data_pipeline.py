import os
import pandas as pd
import json

def process_bse_csv(file_path):
    """
    Processes the BSE CSV file with fixed column mappings.
    Expects:
      - Security Id      -> Symbol
      - Security Name    -> Name
      - ISIN No          -> ISIN
    """
    df = pd.read_csv(file_path)
    required_columns = {
        "Security Id": "Symbol",
        "Security Name": "Name",
        "ISIN No": "ISIN",
    }
    
    # Check if all required columns exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"BSE file is missing columns: {missing}. Available columns: {list(df.columns)}"
        )
    
    df = df.rename(columns=required_columns)
    df["exchange"] = "BSE"
    return df[["Symbol", "Name", "ISIN", "exchange"]]

def process_nse_csv(file_path):
    """
    Processes the NSE CSV file with fixed column mappings.
    Expects:
      - SYMBOL            -> Symbol
      - NAME OF COMPANY   -> Name
      - ISIN NUMBER       -> ISIN
    """
    df = pd.read_csv(file_path)
    required_columns = {
        "SYMBOL": "Symbol",
        "NAME OF COMPANY": "Name",
        "ISIN NUMBER": "ISIN",
    }
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"NSE file is missing columns: {missing}. Available columns: {list(df.columns)}"
        )
    
    df = df.rename(columns=required_columns)
    df["exchange"] = "NSE"
    return df[["Symbol", "Name", "ISIN", "exchange"]]

def process_nse_sme_csv(file_path):
    """
    Processes the NSE SME CSV file with fixed column mappings.
    Expects:
      - SYMBOL           -> Symbol
      - NAME_OF_COMPANY  -> Name
      - ISIN_NUMBER      -> ISIN
    """
    df = pd.read_csv(file_path)
    required_columns = {
        "SYMBOL": "Symbol",
        "NAME_OF_COMPANY": "Name",
        "ISIN_NUMBER": "ISIN",
    }
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"NSE SME file is missing columns: {missing}. Available columns: {list(df.columns)}"
        )
    
    df = df.rename(columns=required_columns)
    df["exchange"] = "NSE"
    return df[["Symbol", "Name", "ISIN", "exchange"]]

def main():
    # Adjust paths if needed
    bse_file = "BSE_ISIN.csv"
    nse_file = "NSE_ISIN.csv"
    nse_sme_file = "NSE_SME.csv"
    
    # Process each CSV
    df_bse = process_bse_csv(bse_file)
    df_nse = process_nse_csv(nse_file)
    df_nse_sme = process_nse_sme_csv(nse_sme_file)
    
    # Combine all data
    df_all = pd.concat([df_bse, df_nse, df_nse_sme], ignore_index=True)
    
    # Build JSON structure grouped by 'Symbol'
    result = {}
    for symbol, group in df_all.groupby('Symbol'):
        # Each group is a list of dicts with keys: Name, ISIN, exchange.
        # Convert keys to lower-case.
        records = [
            {k.lower(): v for k, v in record.items()}
            for record in group[['Name', 'ISIN', 'exchange']].to_dict(orient='records')
        ]
        result[symbol] = records
    
    # Write JSON to .\stock-search-app\public\stock_data.json
    output_dir = os.path.join('.', 'stock-search-app', 'public')
    os.makedirs(output_dir, exist_ok=True)
    json_output_file = os.path.join(output_dir, 'stock_data.json')
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"JSON file created at: {json_output_file}")
    
    # Also write a master CSV in the current directory
    master_csv_file = "master.csv"
    df_all.to_csv(master_csv_file, index=False)
    print(f"Master CSV file created at: {os.path.abspath(master_csv_file)}")

if __name__ == "__main__":
    main()

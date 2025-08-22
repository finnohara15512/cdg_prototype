#!/usr/bin/env python3
"""
Script to extract rows with encounter_id "7877228" from cdg_mvp_data.semi.processed_documents
and save as CSV artifact.

Usage:
1. Ensure Databricks connection is configured with profile "foha-auge"
2. Run: python extract_encounter_data.py
"""

import pandas as pd
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize Databricks connection
    spark = DatabricksSession.builder.profile("foha-auge").getOrCreate()
    
    print("Connecting to Databricks and querying data...")
    
    # Query the table for rows with encounter_id "7877228"
    query = """
    SELECT * 
    FROM cdg_mvp_data.semi.processed_documents 
    WHERE encounter_id = '7877228'
    """
    
    try:
        # Execute query
        df = spark.sql(query)
        
        # Convert to pandas DataFrame
        pandas_df = df.toPandas()
        
        # Check if any rows were found
        if pandas_df.empty:
            print("No rows found with encounter_id '7877228'")
        else:
            print(f"Found {len(pandas_df)} rows with encounter_id '7877228'")
            
            # Save as CSV
            output_file = "encounter_7877228_data.csv"
            pandas_df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            
            # Display basic info about the data
            print(f"\nColumns: {list(pandas_df.columns)}")
            print(f"Data shape: {pandas_df.shape}")
            
    except Exception as e:
        print(f"Error executing query: {str(e)}")

if __name__ == "__main__":
    main()
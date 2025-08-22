#!/usr/bin/env python3
"""
Process healthcare records CSV for encounter 7877228 to extract timestamps
and create chronologically organized output.
"""

import pandas as pd
import re
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import ast

def extract_timestamps_from_content(content_list: List[str]) -> List[Tuple[Optional[str], str]]:
    """
    Extract timestamps from document content and partition content by timestamps.
    Returns list of (timestamp, content_chunk) tuples.
    """
    timestamps_and_content = []
    
    # Join all content for processing
    full_content = ' '.join(content_list)
    
    # Comprehensive timestamp patterns - exclude birth dates and focus on 2024 medical events
    timestamp_patterns = [
        # Date with time patterns - prioritize these
        r'Date\s*:\s*(2024-\d{1,2}-\d{1,2})\s+Time\s*:\s*(\d{1,2}:\d{1,2}:?\d{0,2})',
        r'(2024/\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)',
        r'(\d{1,2}/\d{1,2}/2024)\s+(\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\s*[AP]M)?)',
        
        # Specific medical record patterns with context
        r'Time of Admission\s*:\s*(\d{1,2}:\d{1,2}\s*[AP]M)',
        r'Date of Admission\s*:\s*(2024/\d{1,2}/\d{1,2})',
        r'Date of Discharge\s*:\s*(2024/\d{1,2}/\d{1,2})',
        r'Collection:\s*(\d{1,2}/\d{1,2}/2024)\s+(\d{1,2}:\d{1,2}[AP]M)',
        r'Order Date\s+(\d{1,2}-\d{1,2}-2024-\d{1,2}:\d{1,2})',
        r'Validation:\s*(\d{1,2}/\d{1,2}/2024)\s+(\d{1,2}:\d{1,2})',
        r'Date of visit[^:]*:\s*(\d{1,2}/\d{1,2}/2024)',
        r'Receiving\s*:\s*(\d{1,2}/\d{1,2}/2024)\s+(\d{1,2}:\d{1,2}:\d{1,2}[AP]M)',
        
        # Date only patterns - only 2024 dates
        r'Date\s*:\s*(2024-\d{1,2}-\d{1,2})',
        r'(2024/\d{1,2}/\d{1,2})(?!\s*Age)',  # Exclude if followed by Age
        r'(\d{1,2}/\d{1,2}/2024)(?!\s*Age)',  # Exclude if followed by Age
        r'(\d{1,2}-\d{1,2}-2024)(?!\s*Age)',  # Exclude if followed by Age
        
        # Time only patterns (when 2024 date context exists in document)
        r'Time\s*:\s*(\d{1,2}:\d{1,2}:?\d{0,2})',
        r'(\d{1,2}:\d{1,2}\s*[AP]M)',
    ]
    
    # Extract all timestamps
    found_timestamps = []
    for pattern in timestamp_patterns:
        matches = re.finditer(pattern, full_content, re.IGNORECASE)
        for match in matches:
            # Combine date and time components
            groups = match.groups()
            if len(groups) == 2:
                # Date and time components
                timestamp_str = f"{groups[0]} {groups[1]}"
            else:
                # Single component
                timestamp_str = groups[0]
            
            # Normalize timestamp format
            normalized_ts = normalize_timestamp(timestamp_str)
            if normalized_ts:
                found_timestamps.append((normalized_ts, match.start(), match.end()))
    
    # If no timestamps found, try to infer from context or return null
    if not found_timestamps:
        # Look for any year 2024 references as fallback
        year_pattern = r'2024'
        if re.search(year_pattern, full_content):
            # Return content with null timestamp - very rare case
            return [(None, full_content)]
        else:
            return [(None, full_content)]
    
    # Sort by position in document
    found_timestamps.sort(key=lambda x: x[1])
    
    # If only one timestamp, return entire content with that timestamp
    if len(found_timestamps) == 1:
        return [(found_timestamps[0][0], full_content)]
    
    # Multiple timestamps - partition content
    partitioned_content = []
    content_parts = []
    
    last_end = 0
    for i, (timestamp, start, end) in enumerate(found_timestamps):
        # Add content before this timestamp to previous partition
        if i == 0:
            # First timestamp - content before belongs to this timestamp
            content_before = full_content[:start]
            content_after_start = start
        else:
            # Subsequent timestamps - split content
            content_before = full_content[last_end:start]
            if content_parts:
                # Add to previous partition
                partitioned_content.append((found_timestamps[i-1][0], ' '.join(content_parts) + ' ' + content_before))
                content_parts = []
            content_after_start = start
        
        # Prepare for next iteration
        last_end = end
        
        # If this is the last timestamp, add all remaining content
        if i == len(found_timestamps) - 1:
            remaining_content = full_content[content_after_start:]
            partitioned_content.append((timestamp, remaining_content))
    
    # Handle edge case where partitioning didn't work well
    if not partitioned_content:
        # Fall back to first timestamp with full content
        return [(found_timestamps[0][0], full_content)]
    
    return partitioned_content if partitioned_content else [(found_timestamps[0][0], full_content)]


def normalize_timestamp(timestamp_str: str) -> Optional[str]:
    """
    Normalize various timestamp formats to ISO format.
    Returns None if parsing fails.
    """
    timestamp_str = timestamp_str.strip()
    
    # Define patterns and their corresponding strptime formats
    patterns = [
        # Date with time
        (r'^(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})$', '%Y-%m-%d %H:%M:%S'),
        (r'^(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})$', '%Y-%m-%d %H:%M'),
        (r'^(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.\d+)?$', '%Y/%m/%d %H:%M:%S'),
        (r'^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{1,2})[AP]M$', '%d/%m/%Y %I:%M%p'),
        
        # Date only
        (r'^(\d{4})-(\d{1,2})-(\d{1,2})$', '%Y-%m-%d'),
        (r'^(\d{4})/(\d{1,2})/(\d{1,2})$', '%Y/%m/%d'),
        (r'^(\d{1,2})/(\d{1,2})/(\d{4})$', '%d/%m/%Y'),
        (r'^(\d{1,2})-(\d{1,2})-(\d{4})$', '%d-%m-%Y'),
        
        # Time only - needs date context
        (r'^(\d{1,2}):(\d{1,2}):(\d{1,2})$', '%H:%M:%S'),
        (r'^(\d{1,2}):(\d{1,2})\s*([AP]M)$', '%I:%M %p'),
        
        # Special formats
        (r'^(\d{1,2})-(\d{1,2})-(\d{4})-(\d{1,2}):(\d{1,2})$', None),  # Special handling needed
    ]
    
    for pattern, strptime_format in patterns:
        match = re.match(pattern, timestamp_str, re.IGNORECASE)
        if match:
            try:
                if strptime_format:
                    dt = datetime.strptime(timestamp_str, strptime_format)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Special handling for order date format: DD-MM-YYYY-HH:MM
                    if pattern.startswith(r'^\d{1,2}-\d{1,2}-\d{4}-'):
                        groups = match.groups()
                        day, month, year, hour, minute = groups
                        dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
                        return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
    
    # If no pattern matched, try a more flexible approach
    try:
        # Remove common prefixes/suffixes
        clean_str = re.sub(r'^(Date|Time)\s*:\s*', '', timestamp_str, flags=re.IGNORECASE)
        clean_str = re.sub(r'\s*(AM|PM)\s*$', r' \1', clean_str, flags=re.IGNORECASE)
        
        # Try pandas to_datetime as fallback
        dt = pd.to_datetime(clean_str, errors='coerce', dayfirst=True)
        if pd.notna(dt):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass
    
    return None


def load_csv_with_multiline_handling(input_file):
    """Custom CSV loader that handles multiline entries properly"""
    import csv
    
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    return pd.DataFrame(records)


def main():
    """Main processing function"""
    input_file = "/Users/finnohara/Documents/cdg_mvp_data/cdg_prototype/original_docs/encounter_7877228_data.csv"
    output_file = "/Users/finnohara/Documents/cdg_mvp_data/cdg_prototype/chronology/encounter_7877228_chronological.csv"
    
    print("Loading CSV file with custom multiline handling...")
    df = load_csv_with_multiline_handling(input_file)
    
    print(f"Processing {len(df)} rows...")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows to understand structure
    if len(df) > 0:
        print(f"First row content preview: {df.iloc[0]['content'][:200]}...")
    
    chronological_records = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        try:
            # Parse the content list (it's stored as a string representation of a list)
            content_str = row['content']
            if content_str.startswith('[') and content_str.endswith(']'):
                # Parse the list string
                content_list = ast.literal_eval(content_str)
            else:
                # Single string content
                content_list = [content_str]
            
            # Extract timestamps and partition content
            timestamp_content_pairs = extract_timestamps_from_content(content_list)
            
            # Create records for each timestamp-content pair
            for timestamp, content in timestamp_content_pairs:
                chronological_records.append({
                    'document_id': row['document_id'],
                    'mrn': row['mrn'],
                    'encounter_id': row['encounter_id'],
                    'timestamp': timestamp,
                    'content': content
                })
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Add record with null timestamp as fallback
            chronological_records.append({
                'document_id': row['document_id'],
                'mrn': row['mrn'],
                'encounter_id': row['encounter_id'],
                'timestamp': None,
                'content': row['content']
            })
    
    # Create output DataFrame
    output_df = pd.DataFrame(chronological_records)
    
    # Sort by timestamp (nulls last)
    output_df['timestamp_sort'] = pd.to_datetime(output_df['timestamp'], errors='coerce')
    output_df = output_df.sort_values(['timestamp_sort', 'document_id'], na_position='last')
    output_df = output_df.drop('timestamp_sort', axis=1)
    
    # Save to CSV
    print(f"Saving {len(output_df)} chronological records to {output_file}")
    output_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    total_records = len(output_df)
    records_with_timestamps = len(output_df[output_df['timestamp'].notna()])
    records_without_timestamps = total_records - records_with_timestamps
    
    print(f"\nSummary:")
    print(f"Total records: {total_records}")
    print(f"Records with timestamps: {records_with_timestamps} ({records_with_timestamps/total_records*100:.1f}%)")
    print(f"Records without timestamps: {records_without_timestamps} ({records_without_timestamps/total_records*100:.1f}%)")
    
    # Show timestamp range
    if records_with_timestamps > 0:
        timestamp_df = output_df[output_df['timestamp'].notna()].copy()
        try:
            timestamp_df['timestamp_dt'] = pd.to_datetime(timestamp_df['timestamp'], errors='coerce')
            valid_timestamps = timestamp_df[timestamp_df['timestamp_dt'].notna()]
            if len(valid_timestamps) > 0:
                min_ts = valid_timestamps['timestamp_dt'].min()
                max_ts = valid_timestamps['timestamp_dt'].max()
                print(f"Timestamp range: {min_ts} to {max_ts}")
            else:
                print("No valid timestamps found for range calculation")
        except Exception as e:
            print(f"Error calculating timestamp range: {e}")


if __name__ == "__main__":
    main()
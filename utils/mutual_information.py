#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
import json
import itertools
import os


def cramers_v(x, y):
    # Create contingency table.
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.values.sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    # Bias correction.
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def calculate_all_mutual_info(df, discrete_columns):
    """
    Calculate mutual information between all pairs of columns in the dataset.
    """
    # Make a copy of the dataframe to avoid modifying the original dataframe
    df = df.copy()
    
    # Create a copy of discrete_columns to avoid in-place mutation
    discrete_cols = set(discrete_columns)
    
    results = {}
    columns = df.columns
    
    # Convert discrete columns to category and continuous columns to numeric
    for col in columns:
        if col in discrete_cols:
            df[col] = df[col].astype('category')
        else:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                print(f"Warning: Could not convert {col} to numeric. Treating as discrete.")
                df[col] = df[col].astype('category')
                discrete_cols.add(col)
    
    # Calculate mutual information for all column pairs
    for col1, col2 in itertools.combinations(columns, 2):
        # Skip if the columns are the same
        if col1 == col2:
            continue
        
        # Handle missing values by dropping rows where either column has NaN
        sub_df = df[[col1, col2]].dropna()
        if len(sub_df) == 0:
            print(f"Warning: No valid data points between {col1} and {col2} after removing NaNs")
            continue
            
        X, y = sub_df[col1], sub_df[col2]
        
        # Case 1: Both columns are continuous
        if col1 not in discrete_cols and col2 not in discrete_cols:
            try:
                # Calculate raw MI
                mi = mutual_info_regression(X.values.reshape(-1, 1), y.values, random_state=0)
                mi_value = mi[0]
                
                results[(col1, col2)] = {
                    'value': mi_value,  # Store raw value without rounding for sorting
                    'type': 'mutual_info_regression',
                    'col1_type': 'continuous',
                    'col2_type': 'continuous',
                    'column_types': 'cont,cont'
                }
            except Exception as e:
                print(f"Error calculating mutual_info_regression for {col1} and {col2}: {e}")
                
        # Case 2: Both columns are discrete
        elif col1 in discrete_cols and col2 in discrete_cols:
            try:
                # Convert to string to ensure compatibility
                values1 = X.astype(str)
                values2 = y.astype(str)
                
                # Calculate raw MI
                mi_value = mutual_info_score(values1, values2)
                
                results[(col1, col2)] = {
                    'value': mi_value,  # Store raw value without rounding for sorting
                    'type': 'mutual_info_score',
                    'col1_type': 'discrete',
                    'col2_type': 'discrete',
                    'column_types': 'disc,disc'
                }
            except Exception as e:
                print(f"Error calculating mutual_info_score for {col1} and {col2}: {e}")
                
        # Case 3: Mixed types - one continuous, one discrete
        else:
            # Identify which column is discrete and which is continuous
            if col1 in discrete_cols:
                discrete_col = col1
                continuous_col = col2
                discrete_val = X
                continuous_val = y
                col1_type = 'discrete'
                col2_type = 'continuous'
                column_types = 'disc,cont'
            else:
                discrete_col = col2
                continuous_col = col1
                discrete_val = y
                continuous_val = X
                col1_type = 'continuous'
                col2_type = 'discrete'
                column_types = 'cont,disc'
            
            try:
                # Calculate raw MI
                mi = mutual_info_classif(continuous_val.values.reshape(-1, 1), discrete_val.astype(str), random_state=0)
                mi_value = mi[0]
                
                results[(col1, col2)] = {
                    'value': mi_value,  # Store raw value without rounding for sorting
                    'type': 'mutual_info_classif',
                    'col1_type': col1_type,
                    'col2_type': col2_type,
                    'column_types': column_types
                }
            except Exception as e:
                print(f"Error calculating mutual_info_classif for {col1} and {col2}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="""
        Compute mutual information between all column pairs in a CSV file.

        All columns listed in the JSON array are treated as discrete, and all others as continuous.
        python correlation_contingency_mutual_info.py --discrete '["sex", "race", "c_charge_degree"]' --output results.csv ../dataset/compas/compas_all.csv
        """
    )
    parser.add_argument("csv_file", type=str, help="Path to CSV file")
    parser.add_argument("--discrete", required=True,
                        help="JSON list of discrete column names, e.g., '[\"col1\", \"col2\"]'")
    parser.add_argument("--drop", type=str, default=None,
                        help="Single column name to drop from processing, e.g., --drop column_name")
    parser.add_argument("--output", type=str, help="Path to save results as CSV file")

    args = parser.parse_args()
    
    # Load CSV file
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Drop specified column
    if args.drop:
        if args.drop in df.columns:
            df = df.drop(columns=[args.drop])
            print(f"Dropped column: {args.drop}")
        else:
            print(f"Warning: Column '{args.drop}' not found in dataset")
    
    # Parse discrete columns
    try:
        discrete_columns = json.loads(args.discrete)
    except json.JSONDecodeError:
        print("Error: Could not parse discrete columns list. Please provide a valid JSON list.")
        sys.exit(1)
    
    # Calculate mutual information for all columns
    results = calculate_all_mutual_info(df, discrete_columns)
    
    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame([
        {
            'Column_1': col1,
            'Column_2': col2,
            'MI_Value': round(info['value'], 2),  # Round only for display
            'Type': info['type'],
            'Column_Types': info['column_types']
        }
        for (col1, col2), info in results.items()
    ])
    
    # Sort by mutual information value (descending)
    results_df = results_df.sort_values('MI_Value', ascending=False)
    
    # Print results in a formatted way
    print("\nMutual Information between all column pairs:")
    print(f"{'#':<4} {'Column 1':<20} {'Column 2':<20} {'MI':<10} {'Column Types':<15} {'Type':<25}")
    
    for idx, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{idx:<4} {row['Column_1']:<20} {row['Column_2']:<20} {row['MI_Value']:.2f}   {row['Column_Types']:<15} {row['Type']}")
    
    print("\nScore Range:")
    print("- MI: Raw mutual information (unbounded). Higher values indicate stronger association.")
    print("\nNote: Mutual information is a symmetric measure. The value I(X;Y) = I(Y;X) for any variables X and Y.")
    
    # Save to CSV if output path is provided
    if args.output:
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
    main()
    
    
"""
    python correlation_contingency_mutual_info.py --discrete '["sex", "race", "c_charge_degree"]' --output compas_all_results.csv ../dataset/compas/compas_all.csv
    
    
    python correlation_contingency_mutual_info.py --discrete '["sex", "race",
    "education", "occupation", "retirement_status", "spanish_language_home", "asian_language_at_home", "tribal_enrollment", "psa_or_prostate_exam_recent", "mammogram_screened_last_2y", "annual_income_labels"]'   --output fake_dataset_mutual_information.csv  
    ../dataset/fake_dataset_fader_sensitive/fake_dataset_sensitive_all.csv 
"""


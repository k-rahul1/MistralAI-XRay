# Author 
# Rahul Kumar (Northeastern University)

import pandas as pd
import argparse
import re

def main(args):
    """
    Processes a CSV file containing medical reports, handling missing values, filtering specific tokens, and saving the processed data.

    This function reads a CSV file into a pandas DataFrame, replaces any NaN values in the 'findings' column 
    with corresponding values from the 'impression' column, filters out unwanted tokens from 'findings', 
    and saves the processed DataFrame to a new CSV file. It also saves UIDs for which 'findings' are empty into a csv file.

    """
    # read csv file into dataframe
    df = pd.read_csv(args.i_report_path)
    processed_df = df
    # Replace NaN 'findings' with 'impression' if 'findings' is NaN
    processed_df['findings'] = processed_df.apply(lambda row: row['impression'] if pd.isna(row['findings']) else row['findings'], axis=1)
    
    # find uid for empty findings even after replacing with impression
    ids_df = processed_df[processed_df['findings'].isna()]['uid']
    uid_df = pd.DataFrame(ids_df, columns=['uid'])
    uid_df.to_csv(args.o_uid_path, index=False)

    # Drop rows where 'findings' is NaN
    processed_df = processed_df[(~processed_df['findings'].isna())]

    # Iterate over the 'findings' column with index
    for i, findings in processed_df['findings'].items():
        # Split the findings into tokens (words)
        tokens = re.split(r'[ /-]', findings)
        # Filter out the tokens that are not 'XXXX'
        processed_findings = [token for token in tokens if token not in ['XXXX', 'XXXX.', 'XXXX,', '4XXXX', '(XXXX', "XXXX't", "XXXX's", 'XXXX.In', '(4.3 x 2.8 XXXX)', '5XXXX', '(XXXX),', 'XXXX..', "XXXX'", 'XXXX)', 'XXXX;', 'XXXX:']]
        # Join the filtered tokens back into a string and set it in the DataFrame
        processed_df.loc[i, 'findings'] = ' '.join(processed_findings)
        
    # converting the dataframe to csv file
    processed_df.to_csv(args.o_report_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_report_path", type=str, default="../Dataset/Open-i/indiana_reports.csv", help="path to indiana reports csv file")
    parser.add_argument("--o_uid_path", type=str, default="../Dataset/Open-i/uid_empty_findings.csv", help="path to uid csv file")
    parser.add_argument("--o_report_path", type=str, default="../Dataset/Open-i/indiana_reports_processed.csv", help="path to processed reports csv file")
    args = parser.parse_args()

    main(args)

import pandas as pd
import argparse
import re

def main(csvPath):
    """
    Processes a CSV file containing medical reports, handling missing values, filtering specific tokens, and saving the processed data.

    This function reads a CSV file into a pandas DataFrame, replaces any NaN values in the 'findings' column 
    with corresponding values from the 'impression' column, filters out unwanted tokens from 'findings', 
    and saves the processed DataFrame to a new CSV file.

    Parameters:
    - csvPath (str): Path to the input CSV file containing medical report data.
    """
    # read csv file into dataframe
    df = pd.read_csv(csvPath)
    processed_df = df
    # Replace NaN 'findings' with 'impression' if 'findings' is NaN
    processed_df['findings'] = processed_df.apply(lambda row: row['impression'] if pd.isna(row['findings']) else row['findings'], axis=1)
    
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

    #### code to find uid for empty findings
    # ids_df = processed_df[processed_df['findings'].isna()]['uid']
    # # print(processed_df.columns)
    # uid_df = pd.DataFrame(ids_df, columns=['uid'])
    # uid_df.to_csv("uid_remove_images.csv", index=False)
    ####
        
    # converting the dataframe to csv file
    processed_df.to_csv("processed_indiana_reports.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="indiana_reports.csv", help="path to csv file")
    args = parser.parse_args()

    main(args.csv_path)
# Author 
# Rahul Kumar (Northeastern University)

import pandas as pd
import argparse

def main(file_path):
    """
    Calculate and print the mean and standard deviation for specified evaluation metrics from a CSV file.

    This function reads a CSV file containing various evaluation scores for models or algorithms. It then
    calculates and prints the mean and standard deviation for selected columns that measure performance
    (bleu_score, bertscore, semb_score).

    Args:
    file_path (str): The path to the CSV file containing evaluation metric scores.

    Outputs:
    The function prints the mean and standard deviation of the selected columns directly to the console.
    """
    df = pd.read_csv(file_path)

    columns_of_interest = ['bleu_score','bertscore','semb_score']

    averages = df[columns_of_interest].mean()
    std= df[columns_of_interest].std()

    print("Column Averages:")
    print(averages)
    print("Column STD:")
    print(std)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='Training/eval_metric_score_mistral.csv')
    args = parser.parse_args()

    main(args.file_path)

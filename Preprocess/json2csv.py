# Author 
# Rahul Kumar (Northeastern University)

import json
import csv
import pandas as pd


def json2csv(json_file_path,train_report_path):
    """
    Convert a JSON file containing medical data into a CSV format focusing on specific columns.
    This will be required during inference to compare the train report with input image.

    Args:
    json_file_path (str): The file path of the JSON file containing the source data.
    train_report_path (str): The file path where the CSV file will be saved.

    """
    df= pd.read_json(json_file_path)

    df1=df[[ 'uid','findings']]

    df1.to_csv(train_report_path, index=False)


if __name__ == "__main__":
    train_json_file_path = '../Dataset/Open-i/train.json'
    train_report_path = '../Dataset/Open-i/trainReport.csv'
    test_json_file_path = '../Dataset/Open-i/test.json'
    test_report_path = '../Dataset/Open-i/testReport.csv'

    json2csv(train_json_file_path,train_report_path)
    json2csv(test_json_file_path,test_report_path)
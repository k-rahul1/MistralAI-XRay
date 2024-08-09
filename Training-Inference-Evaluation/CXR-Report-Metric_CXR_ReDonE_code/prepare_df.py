# Author 
# Rahul Kumar (Northeastern University)

import pandas as pd
import json
import argparse

def main(fpath, opath, tpath):
    filt = 'findings'
        
    # Read the CSV file
    df = pd.read_csv(fpath)
    
    test = pd.read_csv(tpath)
    
    pred = df[filt]
    pred = pd.concat([test[['uid']], pred], axis=1)
    
    pred.to_csv(opath, index=False)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='../final_prediction_report_mistral.csv')
    parser.add_argument('--opath', default='../final_prediction_report_mistral_with_uid.csv') 
    parser.add_argument('--rpath', default='../../Dataset/Open-i/testReport.csv') 

    args = parser.parse_args()
    
    main(args.fpath, args.opath, args.rpath)

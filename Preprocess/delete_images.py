# Author 
# Rahul Kumar (Northeastern University)

import pandas as pd

# Extracting the UIDs for the images which needs to be deleted before creaing JSON file and training
# as these images does not have corresponding finding section in the report.

file1 = pd.read_csv('../Dataset/Open-i/indiana_projections.csv')

file2 = pd.read_csv('../Dataset/Open-i/uid_empty_findings.csv')

merged_df = pd.merge(file2, file1, on='uid', how='left')


result_df = merged_df[['uid', 'filename']]

result_df.to_csv('../Dataset/Open-i/images_to_be_deleted.csv', index=False)

# Display the result
print(result_df)

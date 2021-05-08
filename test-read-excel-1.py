import pandas as pd
import os

data_dir = 'data'
input_data_filename = 'excel_test1.xls'
df = pd.read_excel(
    os.path.join(data_dir, input_data_filename), 
    header=[3, 4],
    index_col=[1, 2]
)
print(df)
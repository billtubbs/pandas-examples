import pandas as pd
import os

data_dir = 'data'
input_data_filename = 'excel_test2.xls'
df = pd.read_excel(
    os.path.join(data_dir, input_data_filename), 
    header=[3, 4],
    index_col=[0, 1]
)
print(df)
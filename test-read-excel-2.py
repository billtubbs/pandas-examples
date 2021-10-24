import pandas as pd
from platform import python_version
import os

print(f"Python: {python_version()}")
print(f"Pandas: {pd.__version__}")

data_dir = 'data'
input_data_filename = 'excel_test2.xls'
df = pd.read_excel(
    os.path.join(data_dir, input_data_filename), 
    header=[3, 4],
    index_col=[0, 1]
)
print(df)
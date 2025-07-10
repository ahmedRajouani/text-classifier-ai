import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
    'Math': [85, 78, np.nan, 92, 88],
    'Science': [90, 82, 85, 88, 79],
    'English': [88, 85, 90, 87, 84],
    'Class': ['A', 'B', 'A', 'B', 'A']
}


df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
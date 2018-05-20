print("*********************** LOADING DATA ***********************")

print("*********************** USING FOR LOOP ***********************")

# Loading data
# Turning a csv file into a Matrix:

import numpy as np

file = open("C:/Users/surya/Desktop/Python/data_2d.csv")
X = []

# Split using comma
# Convert to float
# Append to an empty List

for line in file:
    row = line.split(',')
    sample = map(float, row)
    X.append(row)

# Converting into an array
X = np.array(X)
print(X)

# Shape
print(X.shape)

print("*********************** USING PANDAS ***********************")

# Pandas

import pandas as pd

file = open("C:/Users/surya/Desktop/Python/data_2d.csv")

x = pd.read_csv(file, header=None)

# Check data type
print(type(x))
# <class 'pandas.core.frame.DataFrame'>
# It's a dataframe

print(x.shape)
x.head()
x.head(10)

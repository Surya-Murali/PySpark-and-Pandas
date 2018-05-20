print("*********************** LOADING DATA ***********************")

print("*********************** CONNECTING TO SQL SERVER USING PYTHON ***********************")

# Connecting to SQL Server Using Python - PERFECT

import pyodbc
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=DESKTOP-748SD4\SQLEXPRESS;"
                      "Database=mydb;"
                      "Trusted_Connection=yes;")

print(cnxn)
cursor = cnxn.cursor()
cursor.execute('SELECT * FROM Data$')

# Creating an empty List
L = []
for row in cursor:
    print(row)
    L.append(row)

# Converting it to an Array
import numpy as np
A = np.array(L)
print(A)
print(A[0,0])

print("*********************** CONNECTING TO MYSQL DATABASE USING PYTHON ***********************")

# Connecting to MySQL Database Using Python

# Fetch a single line:
# Fetch a single row using fetchone() method.
# row = cursor.fetchone ()

# Fetch multiple rows:
# Fetch all of the rows from a query
data = cursor.fetchall ()

# Python script:
# import the MySQLdb and sys modules
import MySQLdb
import sys

# Open a database connection
# Change the host IP address, username, password and database name to match your own
connection = MySQLdb.connect (host = "192.168.1.2", user = "user", passwd = "password, db = "scripting_mysql")

# A cursor object using cursor() method
cursor = connection.cursor ()

# Execute the SQL query using execute() method.
cursor.execute ("SELECT FIRSTNAME, LASTNAME FROM TABLE")

# fetch all of the rows from the query
data = cursor.fetchall()

# Print the rows
for row in data :
print row[0], row[1]

# Close the cursor object
cursor.close()

# Close the connection
connection.close ()
                              
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

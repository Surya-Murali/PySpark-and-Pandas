#Pandas
#The Pandas library is built on NumPy and provides easy-to-use data structures and data analysis tools for the Python 

#Import
import pandas as pd

#Pandas Data Structures

#Series
#1D - A one-dimensional labeled array a capable of holding any data type 
s = pd.Series([2, 5, 6, -92], index=['a', 'b', 'c', 'd'])

#DataFrame
#A two-dimensional labeled data structure with columns of potentially different types
data = {'Country': ['India', 'US'], 'Capital': ['New Delhi', 'Washington, D.C.']}
df = pd.DataFrame(data, columns=['Country', 'Capital'])

Input / Output
#Read and Write to CSV
pd.read_csv('file.csv', header=None, nrows=5)
df.to_csv('mydata.csv')

#Read and Write to Excel
pd.read_excel('file.xlsx')
pd.to_excel('dir/myDataFrame.xlsx', sheet_name='Sheet1')

#Read multiple sheets from the same file
xlsx = pd.ExcelFile('file.xls')
df = pd.read_excel(xlsx, 'Sheet1')

#Help
help(pd.Series.loc)

#Selection

#Getting
#Get one element
s['b'] 
#5

#Get subset of a DataFrame
df[1:] 
#Country  Capital 
#1 India  New Delhi 
#2 US     Washington D.C.

#Selecting, Boolean Indexing & Setting

#By Position
#Select single value by row & column
#df.iloc[[0],[0]] 
df.iat([0],[0])

#By Label
#Select single value by row & column labels
df.loc[[0], ['Country']] 
df.at([0], ['Country']) 

#By Label/Position
#Select single row of subset of rows
df.ix[2] 

#Select a single column of subset of columns
df.ix[:,'Capital'] 

#Select rows and columns
df.ix[1,'Capital'] 

#Boolean Indexing
#Series s where value is not >1
s[~(s > 1)]

#s where value is <-1 or >2
s[(s < -1) | (s > 2)] 

#Use filter to adjust DataFrame
df[df['Population']>1200000000] 

#Setting
#Set index a of Series s to 6
s['a'] = 6 

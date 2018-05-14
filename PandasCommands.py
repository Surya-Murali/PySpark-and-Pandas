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
df[df['Country'] == 'India'] 

#Setting
#Set index a of Series s to 6
s['a'] = 6 

#Read and Write to SQL Query or Database Table
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
pd.read_sql("SELECT * FROM my_table;", engine)
pd.read_sql_table('my_table', engine)
pd.read_sql_query("SELECT * FROM my_table;", engine)

#read_sql()is a convenience wrapper around read_sql_table() and read_sql_query()
pd.to_sql('myDf', engine)

#Dropping
#Drop values from rows (axis=0)
s.drop(['a', 'c']) 
#Drop values from columns(axis=1)
df.drop('Country', axis=1) 

#Sort & Rank
#Sort by labels along an axis
df.sort_index() 

#Sort by the values along an axis
df.sort_values(by='Country')
#Assign ranks to entries
df.rank() 

#Retrieving Series/DataFrame Information
#Basic Information
#(rows,columns)
df.shape 

#Describe index
df.index 

#Describe DataFrame columns
df.columns

#Info on DataFrame
df.info() 

#Number of non-NA values
df.count() 

#Summary
#Sum of values
df.sum() 

#Cummulative sum of values
df.cumsum() 

#Minimum/maximum values
df.min()/df.max() 

#Minimum/Maximum index value
df.idxmin()/df.idxmax() 

#Summary statistics
df.describe() 

#Mean of values
df.mean() 

#Median of values
df.median() 


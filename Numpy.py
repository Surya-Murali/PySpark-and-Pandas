# Bye Bye Cincinnaattttiiiiiiiiti. California - first day - May 17, 2018

# In the terminal, type:
# pip install pandas
# pip install numpy 

# For Python3,
# pip3 install pandas
# pip3 install numpy 

import numpy as np
import pandas as pd

print("Hello World!")
print(np)

# List
L = [1,2,3]
print(L)

# For loop
for e in L:
    print(e)

# Appending List - Method 1
L.append(4)

# Appending List - Method 2
L = L + [5]

# Vector Addition in List
# Creating an empty List
L2 = []

# Adding L to L:
for i in L:
    L2.append(i + i)

# In List, plus sign does concatenation, whereas in Arrays, plus sign does Vector addition.
# For Vector addition in Lists, we use the for loop.
print(L2)

print(2*L)
# [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

# Squaring in Lists
LSquared = []
for i in L2:
    LSquared.append(i*i)
print(LSquared)

print("############################################")

# Arrays
A = np.array([1,2,3])
print(A)

# For loop
for i in A:
    print(i)

# Append cannot be done in Numpy Arrays: Error
# A.append(4)

# Not works
# A= A+[5]
# print(A)

# In List, plus sign does concatenation, whereas in Arrays, plus sign does Vector addition.
# For Vector addition in Arrays, we just use the plus sign

# Vector Addition in Arrays:
A2 = A+A
print(A2)
# [2 4 6]

# Therefore, Arrays does element wise operation.
print(A2*A2)
# [4 16 36]

# Array Cubing / Power:
print(A2**3)
# [  8  64 216]

# Important Lesson:
# Arrays - for vector operations - add, multiply, etc.
# List - convenient for appending/ concatenation. For vector operations, for loop has to be used.
# But for loops are slow in Python. Hence, numpy arrays are recommended in these cases.

print("############################################")

#Dot Product
a = np.array([1,2])
b = np.array([3,4])
dot= 0

print(zip(a,b))
# <zip object at 0x0000014AA47AECC8>
# Python zip(): The zip() function take iterables (can be zero or more), makes iterator that aggregates elements based on the iterables passed, and returns an iterator of tuples.

for i,j in zip(a,b):
    dot = dot + i*j
    print(dot)
# 3
# 11

print(dot)
# 11

# Method 2 for dot product:
print(a*b)
# [3,8]

print(np.sum(a*b))
# 11

# Method 3:
print(np.dot(a,b))
print(a.dot(b))
print(b.dot(a))

# Matrix:

M = np.array([ [1,2], [3,4] ])
print(M)
# [[1 2]
#  [3 4]]

# First Element:
M[0][0]
M[0,0]

# List Again
L = [ [1,2], [3,4] ]
L[0][0]

# Converting into Matrix:
M2 = np.matrix(M)
print(M2)

# Converting into Array:
A = np.array(M)
print(A)

# Transpose - Matrix operations can also be performed in numpy arrays:
print(A.T)

# Lesson Learnt: Matrix is a two dimensional numpy array. And Vector is a 1 D numpy array.
# Matrix is a 2 D Mathematical object that contains numbers, whereas a Vector is a 1 D Mathematical object that contains numbers.

# The official documentation suggests against using a Matrix in Python.
# If you see a matrix, convert to a numpy array.

print("############################################")

# Different ways of generating arrays:

np.array([1,2,3])

# Zeros Array:

z = np.zeros(10)
print(z)

# 10*10 Matrix with all Zeros:
# Only one input which contains a tuple
Z = np.zeros((10,10))
print(Z)

ones = np.ones((10,10))
print(ones)

# Random Array of size 10*10:
# Gives us uniformly distributed numbers between 0 and 1
R = np.random.random((10,10))
print(R)

# For Gaussian distributed functions: (Mean 0 and Variance 1)
# Pass input individually (not as tuples
G = np.random.randn(10,10)

#Calculating Mean and Variance:
print(G.mean())
print(G.var())

print("############################################")

# Word Problem:
# The admission fee at a fair is $2.50 for children and $5 for adults.
# On a certain day, 300 people enter and $1250 is collected.
# How many children and adults attended the fair?

# So, x + y = 300
# And, 2.5*x + 5*y = 1250

a = np.array([[1, 2.5], [2.5, 5]])
b = np.array([[300, 1250]])
# Solve Function
answer = np.linalg.solve(a, b)

print(answer)

print("############################################")

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a, b):
  result = 0
  for e, f in zip(a, b):
    result += e*f
  return result

t0 = datetime.now()
for t in range(T):
  slow_dot_product(a, b)
dt1 = datetime.now() - t0

t0 = datetime.now()
for t in range(T):
  a.dot(b)
dt2 = datetime.now() - t0

print("dt1 / dt2:", dt1.total_seconds() / dt2.total_seconds())

#########################################################################

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a, b):
  result = 0
  for e, f in zip(a, b):
    result += e*f
  return result

t0 = datetime.now()
for t in range(T):
  slow_dot_product(a, b)
dt1 = datetime.now() - t0

t0 = datetime.now()
for t in range(T):
  a.dot(b)
dt2 = datetime.now() - t0

print("dt1 / dt2:", dt1.total_seconds() / dt2.total_seconds())

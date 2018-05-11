#Spark

#PySpark is the Spark Python API that exposes the Spark programming model to Python.

#Initializing Spark
#SparkContext

from pyspark import SparkContext
sc = SparkContext(master = 'local[2]')

#Inspect SparkContext

#Retrieve SparkContext version
sc.version 

#Retrieve Python version
sc.pythonVer 

#Master URL to connect to
sc.master 

#Path where Spark is installed on worker nodes
str(sc.sparkHome) 

#Retrieve name of the Spark User running SparkContext
str(sc.sparkUser()) 

#Return application name
sc.appName 

#Retrieve application ID
sc.applicationId 

#Return default level of parallelism
sc.defaultParallelism 

#Default minimum number of partitions for RDDs
sc.defaultMinPartitions

#Configuration

from pyspark import SparkConf, SparkContext
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

#Using The Shell
#In the PySpark shell, a special interpreter-aware SparkContext is already created in the variable called sc. 
#$ ./bin/spark-shell --master local[2]
#$ ./bin/pyspark --master local[4] --py-files code.py

#Set which master the context connects to with the --master argument, and add Python .zip, .egg or .py files to the runtime path by passing a comma-separated list to --py-files.

#Loading Data
#Parallelized Collections

rdd = sc.parallelize([('a',2),('a',5),('b',6)])
rdd2 = sc.parallelize([('a',1),('c',2),('b',2)])
rdd3 = sc.parallelize(range(100))
rdd4 = sc.parallelize([("a",["x","y","z"]), ("b",["l", "m"])])

#External Data
#Read either one text file from HDFS, a local file system or or any Hadoop-supported file system URI with textFile(), or read in a directory of text files with wholeTextFiles()

textFile = sc.textFile("/my/directory/*.txt")
textFile2 = sc.wholeTextFiles("/my/directory/")

#Retrieving RDD Information 
#Basic Information

#List the number of partitions
rdd.getNumPartitions() 

#Count RDD instances
rdd.count()  
#3

#Count RDD instances by key
rdd.countByKey() 
defaultdict(<type 'int'>,{'a':1,'b':2})

#Count RDD instances by value
rdd.countByValue() 
defaultdict(<type 'int'>,{('b',2):1,('a',5):1,('a',6):1})

#Return (key,value) pairs as a dictionary
rdd.collectAsMap() 

#Sum of RDD elements
rdd3.sum() 

#Check whether RDD is empty
sc.parallelize([]).isEmpty()

#Summary

#Maximum value of RDD elements
rdd3.max()
  
#Minimum value of RDD elements
rdd3.min() 

#Mean value of RDD elements
rdd3.mean()

#Standard Deviation value of RDD elements
rdd3.stdev()

#Variance value of RDD elements
rdd3.variance()

#Compute histogram by bins
rdd3.histogram(3)

#Summary statistics (count, mean, stdev, max & min)
rdd3.stats()

#Applying Functions

#Apply a function to each RDD element
rdd.map(lambda x: x+(x[1],x[0])) 

#Apply a function to each RDD element and flatten the result
rdd5 = rdd.flatMap(lambda x: x+(x[1],x[0]))
rdd5.collect()

#Apply a flatMap function to each (key,value) pair of rdd4 without changing the keys
rdd4.flatMapValues(lambda x: x).collect()

#Selecting Data

#Getting
#Return a list with all RDD elements
rdd.collect()

rdd.take(2)

rdd.first()

rdd.top(2) 
  
#Sampling

rdd3.sample(False, 0.15, 81).collect() 

#Filtering

rdd.filter(lambda x: "a" in x).collect()

rdd5.distinct().collect()
    
rdd.keys().collect() 
      
#Iterating

def g(x): print(x)

rdd.foreach(g) 
      
#Reducing 

rdd.reduceByKey(lambda x,y : x+y).collect()     

rdd.reduce(lambda a, b: a + b) 


  
  
  
  
  

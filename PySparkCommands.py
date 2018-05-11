#SPARK

#PySpark is the Spark Python API that exposes the Spark programming model to Python.

#INITIALIZING SPARK
#SparkContext

from pyspark import SparkContext
sc = SparkContext(master = 'local[2]')

#INSPECT SparkContext

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

#CONFIGURATION

from pyspark import SparkConf, SparkContext
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

#USING THE SHELL
#In the PySpark shell, a special interpreter-aware SparkContext is already created in the variable called sc. 
#$ ./bin/spark-shell --master local[2]
#$ ./bin/pyspark --master local[4] --py-files code.py

#Set which master the context connects to with the --master argument, and add Python .zip, .egg or .py files to the runtime path by passing a comma-separated list to --py-files.

#LOADING DATA
#PARALLELIZED COLLECTIONS

rdd = sc.parallelize([('a',2),('a',5),('b',6)])
rdd2 = sc.parallelize([('a',1),('c',2),('b',2)])
rdd3 = sc.parallelize(range(100))
rdd4 = sc.parallelize([("a",["x","y","z"]), ("b",["l", "m"])])

#EXTERNAL DATA
#Read either one text file from HDFS, a local file system or or any Hadoop-supported file system URI with textFile(), or read in a directory of text files with wholeTextFiles()

textFile = sc.textFile("/my/directory/*.txt")
textFile2 = sc.wholeTextFiles("/my/directory/")

#RETRIEVING RDD INFORMATION
#BASIC INFORMATION

#List the number of partitions
rdd.getNumPartitions() 

#Count RDD instances
rdd.count()

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

#SUMMARY

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

#APPLYING FUNCTIONS

#Apply a function to each RDD element
rdd.map(lambda x: x+(x[1],x[0])) 

#Apply a function to each RDD element and flatten the result
rdd5 = rdd.flatMap(lambda x: x+(x[1],x[0]))
rdd5.collect()

#Apply a flatMap function to each (key,value) pair of rdd4 without changing the keys
rdd4.flatMapValues(lambda x: x).collect()

#SELECTING DATA

#GETTING
#Return a list with all RDD elements
rdd.collect()

#Take first 2 RDD elements
rdd.take(2)

#Take first RDD element
rdd.first()

#Take top 2 RDD elements
rdd.top(2) 
  
#SAMPLING
#Return sampled subset of rdd3
rdd3.sample(False, 0.15, 81).collect() 

#FILTERING
#Filter the RDD 
rdd.filter(lambda x: "a" in x).collect()

#Return distinct RDD values
rdd5.distinct().collect()

#Return (key,value) RDD's keys
rdd.keys().collect() 
      
#ITERATING

def g(x): print(x)
#Apply a function to all RDD elements
rdd.foreach(g) 
#('a', 7)
#('b', 2)
#('a', 2)

#RESHAPING DATA
#REDUCING
#Merge the rdd values for each key
rdd.reduceByKey(lambda x,y : x+y).collect()     

#Merge the rdd values
rdd.reduce(lambda a, b: a + b) 

#GROUPING BY
#Return RDD of grouped values
rdd3.groupBy(lambda x: x % 2).mapValues(list).collect()

#Group rdd by key
rdd.groupByKey().mapValues(list).collect()
#[('a',[7,2]),('b',[2])]
 
#AGGREGATING
seqOp = (lambda x,y: (x[0]+y,x[1]+1))
combOp = (lambda x,y:(x[0]+y[0],x[1]+y[1]))

#Aggregate RDD elements of each partition and then the results
rdd3.aggregate((0,0),seqOp,combOp) 

#Aggregate values of each RDD key
rdd.aggregateByKey((0,0),seqop,combop).collect()
#[('a',(9,2)), ('b',(2,1))]

#Aggregate the elements of each partition, and then the results
rdd3.fold(0,add) 

#Merge the values for each key
rdd.foldByKey(0, add).collect()
#[('a',9),('b',2)]

#Create tuples of RDD elements by applying a function
rdd3.keyBy(lambda x: x+x).collect()

#MATHEMATICAL OPERATIONS

#Return each rdd value not contained in rdd2
rdd.subtract(rdd2).collect()  
#[('b',2),('a',7)]

#Return each (key,value) pair of rdd2  with no matching key in rdd
rdd2.subtractByKey(rdd).collect()  
#[('d', 1)]

#Return the Cartesian product of rdd and rdd2
rdd.cartesian(rdd2).collect()

#SORT
#Sort RDD by given function 
rdd2.sortBy(lambda x: x[1]).collect()
#[('d',1),('b',1),('a',2)]

#Sort (key, value) RDD by key
rdd2.sortByKey().collect()
#[('a',2),('b',1),('d',1)]

#REPARTITIONING 
#New RDD with 4 partitions
rdd.repartition(4) 

#Decrease the number of partitions in the RDD to 1
rdd.coalesce(1) 

#SAVING
rdd.saveAsTextFile("rdd.txt")
rdd.saveAsHadoopFile("hdfs://namenodehost/parent/child", 'org.apache.hadoop.mapred.TextOutputFormat')

#STOPPING SparkContext
sc.stop()

#EXECUTION
#$ ./bin/spark-submit examples/src/main/python/pi.py

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

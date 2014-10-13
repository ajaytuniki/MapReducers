
#Fast Implementation of Topic-Sensitive PageRank using MapReduce

###CSE 591 – Cloud Computing Course Project

------------------------------------------------------------
Ajay Kumar Tuniki

Arizona State University 

atuniki@asu.edu 

----------------------------------------------------------
Anvesh Reddy Koppela

Arizona State University 
    
akoppela@asu.edu 

---------------------------------------------------------
Uma Sampath Mallampalli

Arizona State University 

umallamp@asu.edu 

------------------------------------------------------------

##Table of Contents

- [Introduction](#i-introduction)
- [System Model](#ii-system-model)
- [Project Description](#iii-project-description)
- [Use Case Scenarios](#iv-use-case-scenarios)
- [Conclusion](#v-conclusion)


-------------------------------------------------------------------------------

> ###Abstract:
> Information on the World Wide Web is growing at an exponential rate. 
> Today’s web data consists of more than one trillion web pages and the data keeps
> generating at a much faster rate. A single machine cannot analyze and compute 
> PageRank on this massive set of data. Therefore, there is a need for an efficient 
> distributed system to store and process these massive datasets. In this project, we
> are addressing this issue by implementing a MapReduce programming paradigm to compute a variation of PageRank which assigns scores for web pages with respect to 
> pre-determined topics. MapReduce programming paradigm provides an efficient way of
> parallelizing the page rank computation and can be scaled to larger web graph. The 
> main goal of the project is to calculate the topic sensitive page rank values for
> each web page.
> A summary of the project tasks are as follows: Setting up a runnable Hadoop 
> map-reduce environment in cloud platform, constructing an inverted index and link
> graph for the dataset, determining the different topics using an efficient K-Means 
> clustering algorithm and computing topic sensitive page rank values for every web 
> page with respect to each predetermined topic. The project is estimated to complete 
> the Implementation of K- means clustering to extract topics by the end of midterm and > submitting a runnable application for calculating topic sensitive page ranks as a 
> final reportable.

> Keywords— Information Retrieval, Cloud Computing, PageRank, Topic Specific PageRank, > Inverted Index, Link Graph, K-Means Clustering, HDFS, Hadoop, MapReduce, Web Graph, 
> Wikipedia Dataset, Document Term Matrix, TF-IDF, Initial Centroid, Stopword, 
> Stochastic Matrix, Reset Distribution, Irreducibility, Aperiodicity.

###I.	INTRODUCTION

#### a.	Introduction and background to the problem
Information on the World Wide Web is growing at an exponential rate. As a result, many web pages are created every day. The need for developing efficient and effective methods for analyzing and performing computation on this large scale data is gaining significance. Due to exponential growth of web pages, we need an efficient measure to calculate the relative importance of web pages and provide more accurate and relevant search results to the user query.
PageRank is an effective algorithm used to measure the importance of a web page by analyzing the link structure of the web graph. It is based on a probability distribution which simulates the browsing activity of a web suffer who randomly starts on a web page and follows the links on the page. The page that has higher probability that a random surfer will eventually end has a higher page rank.

PageRank computation assigns a score to every page that represents the relative importance of the page with respect to other pages. These PageRank scores are combined with other relevance scores to determine the position of web page in the search results. However, global computation of page rank values are calculated offline without considering the context of the query.
Taher H. Haveliwala has proposed an algorithm to compute the topic sensitive page rank values in the paper “TopicSensitive PageRank”. Based on his idea, in topic sensitive page rank, we compute page rank scores, for every web page with respect to various topics that were predetermined using K-means algorithm. When a search query is issued, these topic sensitive page rank values are combined with other IR scores to determine the ordering of search results. The final page rank values are used to determine the position or rank of web page in the search results.

####b.	Challenges in these problems
Page rank computation involves performing power iteration over the initial PageRank vector and link graph. Power iteration involves repeated matrix-vector multiplication until the initial PageRank vector is aligned in the direction of the primary Eigen vector of the matrix. On an average, power iteration involves matrix-vector multiplication on the order of 100 times.

Because of the massive scale of today’s web data, performing analysis and computation is not feasible on a single machine. Moreover, performing power iteration on a huge link graph using traditional methods consumes lot of time, CPU usage and main memory. 

####c.	Applied technologies and solutions to address these problems
This project implements an efficient way of parallelizing the computation of topic specific page rank using Hadoop map-reduce programming paradigm. It provide an effective infrastructure to automatically parallelize, and can potentially be scaled to large web scale datasets. The following technologies are used to implement this project.
 - Hadoop Distributed File System (HDFS) – Required to store these massive datasets in a distributed manner.
 - Shell Scripting – Required to setup the Hadoop multimode cluster
 - MapReduce Programming in Java – Required to implement various tasks in the project.
 - Knowledge of Java programming is also required.

####d. Expected outcomes of this project
Upon successful implementation of this project, each web page in the dataset is assigned a PageRank score specific to the topic identified.


###II.	SYSTEM MODEL
####A.	System Model
The core architecture of the application is to set up Hadoop multi node cluster with one master node and three slave nodes.

We are provided with three Linux virtual machines to setup the Hadoop multi node cluster. The architecture diagram has been updated accordingly. In this architecture, the Hadoop cluster consists of three data nodes and a slave node. The Name node acts as both master and slave node and two dedicated machines are assigned for slave node.The client submits a MapReduce job to the master node and the job is executed parallel among the three data nodes. The data required for the job to process is replicated across multiple data nodes.

###System Model
-![alt text](https://lh3.googleusercontent.com/k_RiRa-ufyuMoQnU8A7OuVBycu47BbbUHdcInhBpjR1UzzxkcnkZPzvsc1SaOqwGK5e9cg=w1342-h523)

###Project Flow![alt text](https://lh6.googleusercontent.com/36G3XMMFmTssdxCOq3O0sghCO_mOXUvi8-91tw7GhRjrxjYNS94aEcSxU4HHA-txBH_Sjg=w1342-h523)

###Existing System
-![alt text](https://lh6.googleusercontent.com/RmvWKtHAiytgQk5VPWpNaSFxBjKpXceJFj_l2kOtvmK_6NGWloKcutuuyA75mYFTXD9dLQ=w1342-h448)

###Proposed System![alt text](https://lh3.googleusercontent.com/Yl-NhZq5hV75_DUGJls0jsPB104b8X8f1azsS9pS1b19BbLF5mQSHhfaS0JbJzSapl5W_w=w1342-h448)


####B.	Software
 - Shell Scripting – Required to interact with Hadoop distributed file system [HDFS]. 
 - Operating System - Five Linux virtual machines to setup the Hadoop multi node cluster.
 - Hadoop 2.4.1 – Hadoop API to execute MapReduce jobs.
 - Hadoop Distributed File System (HDFS) – Required to store these massive datasets in a distributed manner.
 - MapReduce Programming in Java – Required to implement various tasks in the project. Knowledge of Java programming is also required.
 - Eclipse IDE – IDE required to develop Map-Reduce programs.
 
###III.	PROJECT DESCRIPTION
The mail goal of this project is to retrieve topic specific page rank values for all the web pages. The critical tasks involved in this project are setting up the Hadoop multi node cluster, creating inverted index, constructing link graph, clustering the web pages to extract topics using K-Means clustering and finally computing topic sensitive page rank values. 


###IV.	USE CASE SCENARIOS
When the user enters a search query, then the system searches for the query term against the total corpus of the web. It will be having global page rank measures for each page, which are pre-computed. It combines similarity measures with global page rank measures, with a specified weightage. The system returns top results sorted according to their combined scores. But here the system is not considering context of the query. So our goal is to modify the existing system in computation of page rank. So we are computing topic specific page rank for each page in the web corpus.

We are adding an extra feature of “Topic Specific page rank to each web page in the web corpus”
The following information specify the main input and output for our project
 - Input – Wikipedia article dataset
 - Output – Topic specific page rank value for each article in the dataset.

####A.	Task:1 Getting adequate knowledge of MapReduce Programming Paradigm
Hadoop is a batch processing technology used on large volumes of data. It is required to gain sufficient knowledge about Hadoop and MapReduce paradigm.


####B.	Task 2: Setting up Hadoop MapReduce Platform
We need to set up Hadoop single node cluster on each of our local machines for coding and testing on limited dataset.

The Hadoop environment with single node cluster has been set up successfully in our local Linux machines to develop and test the project. Initially, the implementation the project was done on Hadoop 1. Later, we upgraded it to Hadoop 2.The current stable version of Hadoop 2 Hadoop 2.4.1 is used. While working on the set up, we have observed significant number of changes in Hadoop versions. 

####C.	Task 3: Understanding and Pre-Processing the wikipedia dataset suitable for Hadoop MapReduce
We are using a publicly available dataset of Wikipedia. The Wikipedia dataset used here consists of text files with one line for each Wikipedia article. Each line contains five fields, separated by tabs like below 

    <article ID> <title> <last modified time> <XML version of article> <plain text version>.
The size of the dataset is 1 GB. The Wikipedia dataset available has to be initially processed to make it suitable for our project. 

Each row in the Wikipedia dataset corresponds to a particular article. The outgoing links of that article are available in the target tags of xml version of the article. It is necessary to extract a link structure in a defined format to compute page rank. The dataset has been processed according to the needs of the project and link structure is obtained in further tasks.

####D.	Task 4: Building Inverted Index over dataset.
Inverted Indexing supports fast searching of a query term against the dataset of web pages. MapReduce programming paradigm will be used to the construct inverted index structure from the given dataset. The inverted index created is used for identifying topics and also to compute topic specific Page Ranks of the pages.

####Inverted Index
Inverted index is an index structure used to store the statistical information about the content such as words in a text document. It contains the following information
 - Word name or id
 - Number of articles the word has appeared in the entire article corpus
 - The ids of the articles in which the word has occurred and corresponding number of times the word has occurred in the article.
 - Position of the word in the article can also be stored but, in our implementation of topic specific this information is not useful.
 
####Implementation

 - SimilarityMetricWritable.Java
 - To capture the structure of inverted index we have implemented a custom writable program called SimilarityMetricWritable.java. This class consists of the structure which is required for inverted index. The structure contains following properties.
 - Term Frequency (T.F): for each word in an article, the term frequency is defined as number of times that word has occurred across the given article. It is normalized using L-Infinity norm. i.e. by dividing with maximum number times that any word occurs in the corresponding article.

```
TF = Number of times the current word has occurred in the document / Maximum number of times any word has occurred in the document
```

 - Inverse Document Frequency (I.D.F): It is defined for each word independently. It gives us the distribution of given word across all the articles. It is calculated as logarithm of number of total articles present in the corpus divided by number of articles in which this particular word is present.

```
IDF = Total number of documents in the input dataset / Number of documents that contain the current word
```

 - Word: The content appeared in the article
 - Article Id: Name of the given article

 - InvertedIndexMapper.Java 
The mapper is executed parallel among all the datanodes and the intermediate results are passed to reducer. The input to mapper is the Wikipedia dataset where each article is separated by new line.
 - This mapper function reads a line from Wikipedia dataset and extracts the plain text content of the article. If the article does not contain the plain text format then it is ignored.
 - The pain text article is tokenized into set of key words.
 - The mapper function emits the word and the document number for the each word appeared in the plain text article.

```
Pseudo code for mapper
For each line in the input Wikipedia dataset
	If the line contains plain text field for article 
Break the line into set of string tokens
Ignore the token if the word is stop word or the word contains unidentified characters
If the word is not a stop word emit the word and complex inverted index object

Emit (Word, Article Id)
```

 - InvertedIndexReducer.Java
It takes the intermediate results from the mapper and generated the final output. The reducer is executed parallel among all the datanodes and the final output is written back to HDFS.
 - The reducer obtains the word and a set of all the document ids in which the word appeared from the mapper, after shuffle process.
 - The reducer counts the number of documents the word has appeared and generates the inverted index record for the word.
 - The reducer function emits the word and the inverted index record generated using SimilarityMetricWritable class.


```
Pseudo code for Reducer
For each record received from mapper after shuffle and sort process
	Count the number of documents the word has appeared 
Count the number of times the word has occurred in each document
For each record construct the inverted index record and emit the word along with the inverted index record

Emit (Word, Inverted Index Record)
```


####Input – Output:
 - Input – Wikipedia article dataset.
 - Output – Inverted Index structure for the entire dataset.


####E.	Task 5: Extracting link structure from the wikipedia Dataset
Page rank of a web page is defined as the probability of a surfer landing on the page. So Page rank algorithm is based on the link structure of the web pages. But, the data does not include specific information about the links. We need to extract the link graph of the web pages. It can be efficiently done by using MapReduce programming paradigm.

The link structure format as needed to compute page rank consists of article name (page), initial page rank, the count of outgoing links in it and the list of outgoing links. The link structure has been established successfully. The link structure obtained here would be the input to the page rank algorithm. 

####Implementation:

 - LinkGraphRecordWritable.class
To capture the structure of link structure a custom writable program called LinkGraphRecordWritable.class has been implemented. The structure consists of the article name, initial page rank, count of outgoing links and list of outgoing links of that article. Initially we assume the page rank is one for all the articles.

 - LinkStructureMapper.class
The mapper is executed in parallel among all the datanodes and the intermediate results are passed to reducer. The input to this implementation is a file containing the Wikipedia Dataset with five fields separated by tabs. The fields ‘article name’ and ‘xml version of article’ are read from the input. The outgoing links of an article are placed between the <target></target> tags of xml version of article. A regular expression is defined to capture the links between the tags.

The functionality of this mapper is: 
 - The mapper function reads each line from the given dataset and extracts the article name and the xml version of article in each row. The line corresponds to a particular article.
 - It defines a java regular expression which can identify the links between the tags <target></target> in the xml version of the article. 
 - The mapper function then emits the article name and the outgoing link for each link between the <target></target> tags in that particular line corresponding to the particular article. 

Emit (Article, Outgoing link)

```
Pseudo Code for LinkStructure Mapper:

For each row in DataSet Articles 
Extract Article Name , Extract OutLink Article List
For each outlink article in Outlink Article List
	Emit ( Article Name, outlink article name)
	```
	
 - LinkStructureReducer.class
The Reducer takes the intermediate outputs from the Mapper task to transform and generate the required link structure. The shuffle process combines the pairs based on the key value. Here, the pairs are combined based on the article names. The outputted link structure is defined in a customized format as needed to compute the page rank. The reducer is executed in parallel among all the data nodes and the final output is written back to HDFS. 
The functionality of this reducer is:
 - The reducer obtains all the <key, value> pairs from the mapper task in the form of <article, outgoing link> pairs.
 - The reducer also counts the number of outgoing links for a particular article.
 - It utilizes the Customized format to output the values. The format is as described in the class LinkGraphRecordWritable.
 - The reducer function emits the article, initial page rank, count of outgoing links and the list of outgoing links.

Emit (Article, LinkStructureRecord)


```
Pseudo Code for LinkStructure Reducer:
Let (Article Name,outlink article list) be the list of input key value pairs to the Reducer
For each (Article Name, outlink article list) pair
Count the number of outlinks for the Article Name		
construct Link Structure Record as <Article Name, count , Outlinks List>
Emit(Article, LinkStructure Record)
```
####Input – Output:
 - Input – Wikipedia article dataset
 - Output – Link Structure

####F.	Task 6: Implementation of k-means clustering to extract topics
In order to identify topics, we perform k-means clustering on Wikipedia dataset. To get the optimum value of k, the clustering algorithm will be run for different values of k and the value of k is selected such that it increases intra cluster tightness and inter cluster separation.

KMeans requires an additional data structure to perform clustering on the Wikipedia dataset. The data structure is termed as Document Term Matrix which is built based on the inverted index.

####Document Term Matrix
This data structure is used to represent the articles as vectors in the space of terms. Each document is represented using the set of key words in the article and the importance of each word is determined by TF-IDF similarity metric. This matrix is used to compare the similarity between two documents.


Implementation of Document Term Matrix

 - DocumentTermMatrixMapper.Java
The mapper reads the output of Inverted Index and for each record in inverted index, the mapper emits the document id and the word (SimilarityMetricWritble object). Additionally, the mapper calculates the IDF value for each word and writes to HDFS file system.

```
Pseudo code for mapper
For each line in the inverted index file
Break the line into key (word) value (documents) pair
Calculate the Inverse Document Frequency for each word based on the formula
IDF = Total number of documents in the input dataset / Number of documents that contain the current word

Emit the DocumentId and the Complex Similarity metric object  

Emit (DocumentId, Word)
```

 - DocumentTermMatrixReducer.Java
The reducer reads the intermediate results from mapper after shuffle process and constructs the record for document term matrix. Moreover, the reducer computes the TF values for each word before writing on the HDFS file system. Reducer emits the document id and the document term matrix record (List of SimilarityMetricWritable objects) associated with that document.

```
Pseudo code for Reducer
For each record received from mapper after shuffle and sort process
	Find the maximum word occurred in the document
 		Calculate the term frequency for the word based on the formula
TF = Number of times the current word has occurred in the document / Maximum number of time any word has occurred in the document
For each line construct the document term matrix record and emit the document along with the record
Emit (DocumentId, Document Term Matrix Record)

```

 - Input – Inverted Index generated in task 4
 - Output – Document Term Matrix


####KMeans

KMeans is used to cluster Wikipedia articles to extract the articles that are more similar to each other. Documents with higher similarity form a cluster. Topics can be extracted from these clusters and will be used for computing the topic sensitive page rank.

####Implementation of KMeans

 - KMeansInitialClusters.Java
To perform clustering, initial seed centroids should be selected from the Wikipedia article corpus. This java program reads the document term matrix from the HDFS file system and randomly selects K documents as initial clusters for the computation. This program writes the vector representation of the cluster centroids back to HDFS.

```
Pseudo code for Initial Clusters Centroids
From the document term matrix file
	Load all the documents and matrix terms into memory
	Randomly select K initial records as initial clusters
	Write the selected records into HDFS
```
 


 - KMeansMapper.Java
This mapper function reads cluster centroids and the document term matrix from the file system and for each document, the mapper function computes the document similarity between the document and the cluster centroid. Based on the similarity measures, the mapper identifies the cluster the document belongs to and emits the cluster id and the article id (object of SimilarityMetricWritable).

```
Pseudo code for Mapper
Prerequisite – Compute the TF-IDF similarity metric for each word in the document and Load the cluster documents into memory
For each document in document term matrix do
	Compute the similarity between all the K cluster centroids and the current document
	Compare the similarity between all the cluster centroids and the current document
Assign the document to the cluster that has highest similarity
Emit the cluster id the document is assigned to and the article id from mapper 

Emit (ClusterId, ArticleId)
```

 - KMeansReducer.Java
The reducer function re-computes the cluster centroids based on the results from mapper. The reducer writes the cluster centroids back to HDFS. It emits the cluster id and article id (object of SimilarityMetricWritable). 


```
Pseudo code for Reducer
For each record received from mapper after shuffle and sort process do
	Count the number of documents assigned to the cluster
Compute the average TF and IDF scores for each word in the document present in the cluster.
Construct the updated cluster centroid record and write to HDFS
Emit the cluster id and the article id from the reducer

Emit (ClusterId, ArticleId)

```

The process is repeated until the cluster centroids does not change.

 - Input – Initial Cluster Centroids, Document Term Matrix
 - Output – Clusters

####G.	Task 7: Implementing Page Rank algorithm to get  topic specific page rank for each web page.
To get topic specific page rank for all the web pages, “Stochastic matrix” and “Reset Distribution” are adjusted accordingly to give more priority to the pages corresponding to specific topic. We will implement a MapReduce program to perform this task.

Page Rank uses link structure extracted from Wikipedia data set in “Task5” and computes page rank values for each page in the dataset. The algorithm exploits sparsely presented links among the web articles to efficiently compute the page rank for each page.
  
```
The Page Rank algorithm :  
 - Let P be the set of initial page rank values for each article in Wikipedia data set. Let M be the stochastic matrix and R be the Reset distribution.
 - Final page rank values can be obtained from the initial page rank by repeatedly multiplying Page rank vector P with a matrix M. 
 - Matrix M can be obtained by adding two matrices Stochastic matrix and Reset Distribution with a weightage of alpha.
 - Stochastic Matrix represents the original link structure of the Wikipedia articles and Reset Distribution represents that the “when the user goes randomly to a page other than the link structure. 
- The Reset Distribution can be adjusted in such a way that, user goes to a topic specific page instead of a random page. The below example describes the terminology. 
 - Matrix A is called as transition matrix and if we normalize each column in the above matrix, we get the Stochastic Matrix.
 - All values in the Reset Distribution (of same dimensionality as that of stochastic matrix) are adjusted to 1/n (where n is the number of articles), assuming user randomly selects one page out of all pages.
 - The Reset Distribution and Stochastic Matrices are added together to make the Matrix M, which represents the link graph, irreducible and aperiodic, which essentially means eliminating the unique cycles in the web graph.
 - This modification helps in convergence of the page rank values to unique page rank values.
 - To get the topic specific page rank, instead of making all entries in the reset distribution to 1/n, we can give high values to the articles, which are in the selected topic.
 ```
 
####Implementation of Page Rank:

 PageRankMapper.java:

The mapper is executed in parallel among all the datanodes and the intermediate results are passed to reducer. The input to this implementation is a file containing the Wikipedia Dataset transformed to a well-defined link structure. The link structure input would be in the following format:

The functionality of this mapper is: 
 - This mapper function reads each line from the link structure and emits two kinds of output <key, value> pairs.
 - For source article, it emits a <key, value> pair as : 
<source, (list of articles which has a link from the source)>
 - For each of destination articles it emits a <key, value> pair as:
<article name, (number of out links from the source articles)>

Emit (Source Article, The Outgoing links)
Emit (Destination Article, <Pagerank and Number of Outlinks from corresponding source article>)


```sh
Pseudo code for the Mapper:

For each line in the link Structure file
	Split files to get source article, page rank value, out links
      	For each out link from the source article
Emit <destination article(key),( page rank, number of links from the source article)(value)>
  		For the source article emit <source article(key),(articles pointed by this source article) >
For the source article emit<source article (key), (page rank, number of links from the source node) (value)>
```

The Mapper outputs the two kinds of <Key, Value> Pairs as following:

 - PageRankReducer.java:

The Reducer takes the intermediate outputs from the Mapper task to transform and generate the required PageRanks. The shuffle process combines the pairs based on the key value. Here, the pairs are combined based on the article names. The reducer is executed in parallel among all the datanodes and the final output is written back to HDFS. 

The functionality of this reducer is:
 - 	The Reducer gets the data from Mapper grouped according their keys.
 - The input records to the reducer would be of two types.
 - One type of record would be <Source Article, set of destination articles>. This record is processed to get the out links for the given source article. 
 - The other type of record would be <Destination article, List of corresponding values>.
 - For each value of the record of type destination, page rank is calculated by dividing the current PageRank by number of out links. All the values in the values list corresponding to a particular destination record are summed up to get a new page rank.

Emit (Article, PageRank)

```sh
Pseudo Code for the Reducer:
 
 For each record in the input
	If it is of type 1 (<source node (key), list of out links (values)>)
		Emit<source node, page rank, count, list out link articles>
	Else if it is of second type (<destination node, page rank, count of out links>)
		Get the new page rank value by adjusting value of alpha and reset distribution
		Alpha value is set to 0.9	
		Emit <destination node, new page rank, count of out links>
		
```

This process is continued until the Page Rank values for each page converges. This essentially means that, for each page in the Wikipedia dataset, page rank value in the previous iteration and current iteration would not differ by more than some threshold value.

 - Input: Link Structure of the Wikipedia Dataset articles
 - Output: Page Rank values for each article in the Wikipedia Dataset

H.	Task 8: Running and Testing locally on limited dataset size
The MapReduce job will be run locally by setting up a Hadoop single node cluster. The Dataset used here will be of the size 1 GB. 
We have successfully accomplished the task of running the complete set up in local environment using limited Dataset.
The Hadoop 2.4.1 environment single node clusters have been set up in our local Linux machines by each of us. This has been used to implement the project and also to test. Each of the MapReduce job is defined using a Driver class.

We have successfully tested our MapReduce jobs on a single node cluster. The following screen shot specifies the summary of the jobs ran on a single node cluster.

####I.	Task 9: Setting up and Running Hadoop multi node cluster in the cloud environment
The Cloud set up requires multi node cluster consisting of five virtual machines. One of which will be used as master node and the remaining four as slave nodes.
We have followed up on establishing a multi node cluster using the virtual machines. Due to limited availability, only three virtual machines have been allocated. One of which will be used as master node and the other two as slave nodes. Considering the huge data set, the virtual machines might be scaled depending on the availability. The following screen shots are taken from the remote machines that were provided to setup the Hadoop cluster.

A dedicated user (hduser) is setup to run Hadoop related jobs. The cluster description is given below.
 - Namenode – The machine HNamenode acts as name node in the Hadoop cluster setup.
 - Datanode – HDatanode1 and HDatanode2 are dedicated servers and HNamenode also acts as slave node in the cluster setup.
 - Secondary Namenode – HNamenode acts as secondary namenode.
 - Resource Manager – To monitor the status of the datanodes, the daemon Resource manager runs on HNamenode.
 - Node Manager – To monitor the status of Hadoop task on data nodes, the node manager daemon runs on HNamenode, HDatanode1 and HDatanode2.

####J.	Task 10: Deploying and Testing the functionality of the application on cloud
The application will be deployed on vlab resources and end to end testing is performed on the application.
We have successfully accomplished the task of running the complete set up in cloud environment using complete Dataset.
The Hadoop 2.4.1 environment multi node clusters have been set up on the cloud environment. This has been used to deploy the project and also to test. Each of the MapReduce job is defined using a Driver class.

We have successfully tested our MapReduce jobs on a multi node cluster. The following screen shots specify the summary of the jobs ran on a multi node cluster.

#### K.	Deliverables
The goal of the project is to submit a java application implemented in Hadoop MapReduce environment that computes the topic sensitive page rank score for each web page. The project deliverables include a working environment of Hadoop cluster, a MapReduce algorithm for topic sensitive page rank.
The following deliverables have been identified at the end of the project
 - Code for the project available in mobisphere cloud
 - TSPR.jar – an executable jar file to run the project
 - Mini.tsv – an input dataset with Wikipedia articles
 - StopWords.rtf – set of stop words in the English dictionary, to be eliminated in the project
 

####M.	Execution Steps to be followed
Steps to run the jobs on Hadoop Multinode Cluster:
 - Copy the given jar file into any folder
 - Place the input dataset mini.tsv and StopWords.rtf in the same folder where the jar folder is placed
 - Execute the following commands in a new terminal window
 - Hadoop jar TSPR.jar mini.tsv StopWords.rtf
 - Observe the status of the jobs in the following url  
  HNameNode:19888/jobhistory.app 

###V.	CONCLUSION
The main aim of this proposal is to create a faster implementation for computing topic sensitive PageRank using Hadoop MapReduce environment. As a future work, we can use more efficient clustering algorithms combined with hyperlink structure and co-citation relations to automatically retrieve more accurate topic.

# Quora-question-pair-similarity
## Business Problem
### Description

Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

    Credits: Kaggle

### Problem Statement

    Identify which questions asked on Quora are duplicates of questions that have already been asked.
    This could be useful to instantly provide answers to questions that have already been answered.
    We are tasked with predicting whether a pair of questions are duplicates or not.

### 1.3 Real world/Business Objectives and Constraints

    The cost of a mis-classification can be very high.
    You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
    No strict latency concerns.
    Interpretability is partially important.

# Machine Learning Probelm
##  Data
###  Data Overview

- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290
###  Example Data point

"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"

##  Mapping the real world problem to an ML problem
###  Type of Machine Leaning Problem

It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.
### Performance Metric

Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s):

    log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
    Binary Confusion Matrix

###  Train and Test Construction

We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.
#  Exploratory Data Analysis 
##  Reading data and basic stats 


We are given a minimal number of data fields here, consisting of:

    id: Looks like a simple rowID
    qid{1, 2}: The unique ID of each question in the pair
    question{1, 2}: The actual textual contents of the questions.
    is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

###  Distribution of data points among output classes

    Number of duplicate(smilar) and non-duplicate(non similar) questions


##  Basic Feature Extraction (before cleaning)

Let us now construct a few features like:

    freq_qid1 = Frequency of qid1's
    freq_qid2 = Frequency of qid2's
    q1len = Length of q1
    q2len = Length of q2
    q1_n_words = Number of words in Question 1
    q2_n_words = Number of words in Question 2
    word_Common = (Number of common unique words in Question 1 and Question 2)
    word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
    word_share = (word_common)/(word_Total)
    freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
    freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2

###  Analysis of some of the extracted features 
####  Feature: word_share 


    The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
    The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)

#### Feature: word_Common
The distributions of the word_Common feature in similar and non-similar questions are highly overlapping
##  Featurizing text data with tfidf weighted word-vectors/tfidf word-vectors

#  Machine Learning Models


# Conclusion
Model               |       Tokenizer          |  Train Log loss     | Test Log loss
--------------------|--------------------------|---------------------|--------------
Logistic Regression |        TFIDF             |   0.44              |  0.41        
Linear SVM          |        TFIDF             |   0.45              |  0.45        
xgboost             |        TFIDF             |   0.36              |  0.36        
Logistic Regression |TFIDF  weighted word2Vec  |   0.51              |  0.52        
Linear SVM          |TFIDF weighted word2Vec   |   0.47              |  0.48        
xgboost             |TFIDF weighted word2Vec   |   n/a               |  0.35      





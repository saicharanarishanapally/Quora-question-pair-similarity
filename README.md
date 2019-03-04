# Quora-question-pair-similarity
## 1. Business Problem
### 1.1 Description

Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

    Credits: Kaggle

### Problem Statement

    Identify which questions asked on Quora are duplicates of questions that have already been asked.
    This could be useful to instantly provide answers to questions that have already been answered.
    We are tasked with predicting whether a pair of questions are duplicates or not.

### 1.2 Sources/Useful Links

    Source : https://www.kaggle.com/c/quora-question-pairs

    Useful Links
    Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
    Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
    Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
    Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

### 1.3 Real world/Business Objectives and Constraints

    The cost of a mis-classification can be very high.
    You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
    No strict latency concerns.
    Interpretability is partially important.

# 2. Machine Learning Probelm
## 2.1 Data
### 2.1.1 Data Overview

- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290
### 2.1.2 Example Data point

"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"

## 2.2 Mapping the real world problem to an ML problem
### 2.2.1 Type of Machine Leaning Problem

It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.
### 2.2.2 Performance Metric

Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s):

    log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
    Binary Confusion Matrix

### 2.3 Train and Test Construction

We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.
# 3. Exploratory Data Analysis 
## 3.1 Reading data and basic stats 


We are given a minimal number of data fields here, consisting of:

    id: Looks like a simple rowID
    qid{1, 2}: The unique ID of each question in the pair
    question{1, 2}: The actual textual contents of the questions.
    is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

### 3.2.1 Distribution of data points among output classes

    Number of duplicate(smilar) and non-duplicate(non similar) questions


## 3.3 Basic Feature Extraction (before cleaning)

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

### 3.3.1 Analysis of some of the extracted features 
#### 3.3.1.1 Feature: word_share 


    The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
    The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)

#### 3.3.1.2 Feature: word_Common
The distributions of the word_Common feature in similar and non-similar questions are highly overlapping
## 3.6 Featurizing text data with tfidf weighted word-vectors/tfidf word-vectors

# 4. Machine Learning Models
## 4.4 Building a random model (Finding worst-case log-loss)
TD_IDF weighted word2Vec : Log loss on Test Data using Random Model 0.887242646958
TF-IDF vectors           :Log loss on Test Data using Random Model 0.8844640634051439

## 4.4 Logistic Regression with hyperparameter tuning
### TD_IDF weighted word2Vec
For values of best alpha =  1 The train log loss is: 0.513842874233
For values of best alpha =  1 The test log loss is: 0.520035530431
### TF-IDF vectors 
For values of best alpha =  0.001 The train log loss is: 0.44096373727997223
For values of best alpha =  0.001 The test log loss is: 0.44145341577157887
## 4.5 Linear SVM with hyperparameter tuning 
### TD_IDF weighted word2Vec
For values of best alpha =  0.0001 The train log loss is: 0.478054677285
For values of best alpha =  0.0001 The test log loss is: 0.489669093534
### TF-IDF vectors
For values of best alpha =  1e-05 The train log loss is: 0.45425654847111024
For values of best alpha =  1e-05 The test log loss is: 0.45423679091520236
## 4.6 XGBoost 
### TD_IDF weighted word2Vec
The test log loss is: 0.357054433715
### TF-IDF vectors
For values of best alpha =  1e-05 The train log loss is: 0.3670812049007482
For values of best alpha =  1e-05 The test log loss is: 0.36926796350167773

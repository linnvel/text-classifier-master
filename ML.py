#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:06:45 2020

@author: xchen
"""
## required packages

# system imports
import os
import sys
from termcolor import colored
from colorama import init

# data manipulation and data clean
from nltk.corpus import stopwords

# sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# self-defined
import pipeline


# default data path
DATA_PATH = '../data'
GLOVE_PATH = '../glove.6B'


# default parameters
stop_words = stopwords.words('english')
stop_words = stop_words + ['would','could','may','also', 'one', 'two', 'three', 
                           'first', 'second' ,'third',
                           'someone', 'anyone', 'something', 'anything', 
                           'subject', 'organization', 'lines',
                           'article', 'writes', 'wrote']
tokenize_regex1 = r"\w+|\$[\d\.]+"
tokenize_regex2 = r"[a-zA-Z_]+"


def main_test(path):
    
    dir_path = path or DATA_PATH
    TRAIN_DIR = os.path.join(dir_path, "train")
    TEST_DIR = os.path.join(dir_path, "test")

    # load data
    print (colored('Loading files into memory', 'green', attrs=['bold']))
    
    train_path_list, ylabel_train = pipeline.parse_files(TRAIN_DIR)
    test_path_list, ylabel_test = pipeline.parse_files(TEST_DIR)
    
    train_documents = [pipeline.load_document(path = path, label = y) for \
                       path, y in zip(train_path_list, ylabel_train)]
    test_documents = [pipeline.load_document(path = path, label = y) for \
                      path, y in zip(test_path_list, ylabel_test)]
    
    # clean all documents
    print (colored('Cleaning all files', 'green', attrs=['bold']))
    pipeline.clean_all_documents(train_documents, 
                                 word_split_regex = tokenize_regex1,
                                 stop_words = stop_words,
                                 contraction_dict = 'default')
    pipeline.clean_all_documents(test_documents,
                                 word_split_regex = tokenize_regex1,
                                 stop_words = stop_words,
                                 contraction_dict = 'default')
    
    
    # encode labels
    print (colored('Encoding labels', 'green', attrs=['bold']))
    y_train, y_test, category = pipeline.label_encoder(ylabel_train, ylabel_test, 'ordinal')
    
    
    ## *************************** machine learning ***************************
    
    # calculate the BOW representation
    print (colored('Calculating BOW', 'green', attrs=['bold']))
    X_train_bow = pipeline.BagOfWord.fit_transform(train_documents)
    X_test_bow = pipeline.BagOfWord.transform(test_documents)
    print ("The shape of X after processing is: \ntrain: %s, test: %s"%(X_train_bow.shape, X_test_bow.shape))
    
    # calculate the tf-idf representation
    print (colored('Calculating Tf-idf', 'green', attrs=['bold']))
    X_train_tfidf = pipeline.Tfidf.fit_transform(train_documents)
    X_test_tfidf = pipeline.Tfidf.transform(test_documents)
    print ("The shape of X after processing is: \ntrain: %s, test: %s"%(X_train_tfidf.shape, X_test_tfidf.shape))
    
    # scale
    scaler = preprocessing.Normalizer()
    X_train_scaled = scaler.fit_transform(X_train_bow)     
    X_test_scaled = scaler.transform(X_test_bow)
    
    ## models
    # naive bayes
    clf_nb = MultinomialNB()
    # logistic regression
    clr_lr = LogisticRegression(penalty='l2', C=12, solver='lbfgs', max_iter=500, random_state=42)
    # svm
    clf_svm = SGDClassifier(penalty = 'l2',alpha = 5e-5, random_state=42)
    
    # model selection
    print (colored('Selecting model using 10-fold cross validation', 'magenta', attrs=['bold']))
    clf_list = [clf_nb, clr_lr, clf_svm]
    clf_optimal, clf_f1 = pipeline.model_selection(X_train_tfidf, y_train, clf_list, cv=5, scoring='f1_macro')
    
    # test the optimal classifier with train-test-split
    print (colored('Testing the optimal classifier with train-test split', 'magenta', attrs=['bold']))
    f1 = pipeline.test_classifier(X_train_tfidf, y_train, clf_optimal, test_size=0.2, y_names=category, confusion=True)
    print('Train score (macro f1):%.4f, test score (macro f1):%.4f'%(f1[1],f1[0]))
    
    # predict test set
    print (colored('Predicting test dataset', 'magenta', attrs=['bold']))
    y_pred_ml = pipeline.model_prediction(clf_optimal, X_train_tfidf, y_train, X_test_tfidf)
    pipeline.model_report(y_test, y_pred_ml, y_names=category, confusion=True)




def main():
    init()

    # get the dataset
    print (colored("Where is the dataset?", 'cyan', attrs=['bold']))
    print (colored('Press return with default path', 'yellow'))
    ans = sys.stdin.readline()
    # remove any newlines or spaces at the end of the input
    path = ans.strip('\n')
    if path.endswith(' '):
        path = path.rstrip(' ')

    print ('\n\n')

    # do the main test
    main_test(path)
    
if __name__ == '__main__':
    main()
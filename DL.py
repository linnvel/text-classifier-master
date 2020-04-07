#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 07:37:34 2020

@author: xchen
"""

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
import numpy as np
from nltk.corpus import stopwords

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
                                 word_split_regex = tokenize_regex2,
                                 stop_words = stop_words,
                                 contraction_dict = 'default')
    pipeline.clean_all_documents(test_documents,
                                 word_split_regex = tokenize_regex2,
                                 stop_words = stop_words,
                                 contraction_dict = 'default')
    
    
    # encode labels
    print (colored('Encoding labels', 'green', attrs=['bold']))
    y_train, y_test, category = pipeline.label_encoder(ylabel_train, ylabel_test, 'ordinal')
    
    
    ## *************************** deep learning ***************************
    
    # calculate word embeddings
    print (colored('Calculating word index representation', 'green', attrs=['bold']))
    
    # parameter
    max_sequence_length = 300   # max words to consider in a document
    max_num_words = 20000       # max words to include in the vocabulary 
    embedding_dim = 100
    
    X_train_index, X_test_index, word_index = pipeline.DocumentToWordTndex(train_documents, test_documents,
                                                 max_sequence_length, max_num_words)
    print(f'Found {len(word_index)} unique tokens/words')
    print(f'The maximum word index is {X_train_index.max()}')
    print ("The shape of X after processing is: \ntrain: %s, test: %s"\
           %(X_train_index.shape, X_test_index.shape))
    
    print (colored('Matching with GloVe vectors', 'green', attrs=['bold']))
    embedding_matrix, unmathced = pipeline.construct_embedding_matrix(GLOVE_PATH, word_index, max_num_words, 
                                                  embedding_dim, verbose=6)
       
    
    # train a CNN network
    print(colored("How many epochs?", 'cyan', attrs=['bold']))
    ans = sys.stdin.readline()
    input_str = ans.strip('\n')
    if input_str.endswith(' '):
        input_str = input_str.rstrip(' ')

    epochs = int(input_str or '10')
    
    print(colored("\nWhat is the batch size?", 'cyan', attrs=['bold']))
    ans = sys.stdin.readline()
    input_str = ans.strip('\n')
    batch_size = int(input_str or '16')#8, 16, 32, 64, 128, 256
    
    num_classes = np.unique(y_train).shape[0]
    
    # build model
    print (colored('Building deep learning model...', 'magenta', attrs=['bold']))
    model = pipeline.get_cnn_model(embedding_matrix, max_sequence_length, num_classes)
    
    # train model
    print (colored('\n\nTraining model...', 'magenta', attrs=['bold']))
    f1_train, f1_test, f1_list = pipeline.train_dl_classifier(X_train_index, y_train, model, test_size=0.2, 
                                                   epochs=epochs, batch_size=batch_size, 
                                                   verbose=1, y_names=category, 
                                                   confusion=False, random_state=42)
    print('The training result is... train f1-macro = %.4f, val f1-macro = %.4f'%(f1_train, f1_test))
    
    # save model
    name = 'cnn_batch_%d_epoch_%d'%(batch_size,epochs)
    print (colored("\n\nSaving model to as %s.h"%(name), 'magenta', attrs=['bold']))
    pipeline.model_save(model, name)
    
    # predict test set
    #model = pipeline.model_load('models/cnn_batch_16_epoch_10.h5')
    print (colored('Predicting test dataset...', 'magenta', attrs=['bold']))
    y_pred_dl = pipeline.model_evaluate(model, X_test_index)
    pipeline.model_report(y_test, y_pred_dl, y_names=category, confusion=True)


def main():
    init()

    # get the dataset
    print (colored("Where is the dataset?", 'cyan', attrs=['bold']))
    print (colored('Press return to load data from the default path..', 'cyan'))
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
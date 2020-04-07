#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file includes pre-defined classes and functions used in the machine learning project.
"""

import os
from termcolor import colored

# data manipulation and data clean
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix

# scikit-learn
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, \
                            precision_score, recall_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA


# keras
import keras.backend as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.models import load_model


# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx

# self-defined module
from document import document



## ***************** data extraction *************************
def parse_files(path):
    '''Parse document file pathes and labels from data path
    Parameters
    ----------
    path : str
        Source data file path
    Return
    ------
    file_path_list : list
        List of file paths of all documents
    ylabel: list
        List of classes each document belongs to
    '''

    folders = [f for f in os.listdir(path) if not f.startswith('.')]
    file_path_list = [] # list of file paths of all documents
    ylabel = []# list of classes each document belongs to

    for folder in folders:
        folder_path = os.path.join(path, folder)
        file_path = [os.path.join(folder_path,f) for f in os.listdir(folder_path)\
                     if not f.startswith('.')]
        file_path_list.extend(file_path)
        ylabel.extend([folder]*len(file_path))

    # TODO: remove incompatible files

    return file_path_list, ylabel


def load_document(path, label, header_seperator='\n\n'):
    ''' Load document from file path, return document class'''

    with open(path, 'r') as file:
        return document(path,label).parser(file,header_seperator)


## ***************** data preprocessing *************************
def clean_all_documents(documents_data, **kwargs):
    ''' Call `clean_text` method for every single document object included in the list
    parameter is a list of document objects, overwrite the object directly, return nothing
    '''
    for i, doc in enumerate(documents_data):
        documents_data[i].clean_text(**kwargs)

def label_encoder(train, test=None, encoder='ordinal'):
    # ordinal encoding
    if encoder == 'ordinal':
        ordinal_encoder = OrdinalEncoder()
        y_train= ordinal_encoder.fit_transform(np.array(train).reshape(-1, 1)).reshape(1,-1)[0]
        if test:
            y_test = ordinal_encoder.transform(np.array(test).reshape(-1, 1)).reshape(1,-1)[0]
        category = ordinal_encoder.categories_[0].tolist()

    # one-hot encoding
    elif encoder == 'onehot':
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = onehot_encoder.fit_transform(np.array(train).reshape(-1, 1))
        if test:
            y_test = onehot_encoder.transform(np.array(test).reshape(-1, 1))
        category = onehot_encoder.categories_[0].tolist()
    else:
        raise ValueError('encoder should be `ordinal` or `onehot`')

    if test:
        return y_train, y_test, category
    eles:
        return y_train, category

## ********************* bag of word ***************************
class DocumentSelector(BaseEstimator, TransformerMixin):
    '''Class to select a list of attributes from a list document objects'''

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return [x.get(self.attribute_names) for x in X]


class DocumentToWordCounterTransformer(BaseEstimator, TransformerMixin):
    '''Class to convert a list of text to word count array'''

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for text in X:
            word_counts = Counter(text.split())
            X_transformed.append(word_counts)
        return np.array(X_transformed)


class WordCounterToVectorTransformer():
    '''Class to convert count array to sparse matrix containing word count vector'''

    def __init__(self, vocabulary_size=10000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common(self.vocabulary_size)
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


BagOfWord = Pipeline([
        ('select_text', DocumentSelector(['body'])),
        ('document_to_wordcount', DocumentToWordCounterTransformer()),
        ('wordcount_to_vector', WordCounterToVectorTransformer(vocabulary_size=5000)),
        # TODO: normalize by len_of_body?
])

#FeatureAugmentation = Pipeline([
#        ('select_num', DocumentSelector(['num_of_words','num_of_special_char','num_of_numbers'])),
#        ('std_scalar',StandardScaler()),
#])

Tfidf = Pipeline([
        ('select_text', DocumentSelector(['body'])),
        ('document_to_wordcount', DocumentToWordCounterTransformer()),
        ('wordcount_to_vector', WordCounterToVectorTransformer(vocabulary_size=5000)),
        ('tfidf', TfidfTransformer()),
        ])

## ***************** word embedding *************************
def DocumentToWordTndex(train, test, max_sequnce_len, max_num_words):
    '''Convert a list of document object to an array of word index'''

    # Select body texts from document and convert them to an arrat of text
    selector = DocumentSelector('body')
    train_array = selector.fit(train).transform(train)
    test_array = selector.transform(test)

    # Instantiate the Tokenizer
    tokenizer = Tokenizer(num_words=max_num_words)

    # Learn the vocab and identify the most frequently occuring words
    tokenizer.fit_on_texts(train_array)

    # Transform texts to sequences of word indices
    X_train = tokenizer.texts_to_sequences(train_array)
    X_test = tokenizer.texts_to_sequences(test_array)

    # Save the look-up dictionary for words to indices
    word_index = tokenizer.word_index

    # Pad out sequences by prepending zeros to all text sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_sequnce_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sequnce_len)

    return X_train, X_test, word_index

def load_corpus(path):
    '''Load corpus from `path`, return a dictionary of word vectors
    {keys = words to be embedded, values = vector coefficients}
    download glove.6B online
    '''
    embeddings_dict = {}
    with open(path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs
    print(f'Found {len(embeddings_dict)} word vectors')
    return embeddings_dict

def construct_embedding_matrix(corpus_dir, word_index, max_num_words, embedding_dim=100,
                               oov='zero', verbose=0):
    '''Construct embedding matrix by matching `word_index` with corpus;
    for those words that cannot be found in corpus(out-of-vocabulary words),
    impute with zero vectors by default
    '''
    if embedding_dim not in [50, 100, 200, 300]:
        raise ValueError('dimension should be in [50, 100, 200, 300]')

    # TODO: add other oov methods
    if oov not in ['zero']:
        raise ValueError ("out-of-vocabulary method should be in ['zero']")

    # load corpus
    path = os.path.join(corpus_dir, corpus_dir.split('/')[-1]+'.'+str(embedding_dim)+'d.txt')
    embeddings_index = load_corpus(path)

    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    n_not_find = 0
    unmatched = []

    for word, index in word_index.items():
        if index < max_num_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            # out-of-vocabulary (oov) words
            else:
                n_not_find +=1
                unmatched += [word]
                # print some samples of oov words
                if verbose>0:
                    d = int(1000*0.5**(verbose-1))
                    if n_not_find%d==0:
                        print("Cannot find the GloVe vector for", word)
    print('...')
    print(f'Missed {n_not_find} unmatched words')
    return embedding_matrix, unmatched


## ************************ machine learning ****************************

def test_classifier(X, y, clf, test_size, y_names=None, confusion=False, random_state=42):
    # train-test split
    print ('test size is: %2.0f%%' % (test_size * 100))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    model_report(y_test, y_predicted, y_names, confusion)

    train_f1 = calculate_macro_f1(y_test, y_predicted)
    test_f1 = calculate_macro_f1(y_train, y_train_pred)

    return train_f1, test_f1


def cross_validation(X, y, classifier, cv=5, scoring= 'f1_macro', verbose=0, **kwargs):
    '''Does cross validation with the classifier, return a list of scores,
    where the length of the list is equal to `cv` argument
    return validation scores in all folds
    '''
    scores = cross_val_score(classifier, X, y, cv=cv, scoring=scoring, verbose = verbose, **kwargs)
    print ('The average %s score is %.4f, standard deviation is %.4f.'%\
           (scoring, scores.mean(), scores.std()))
    return scores


def grid_search (X, y, classifier, param_grid, cv=5, verbose=3, scoring='f1_macro', **kwargs):
    '''Does grid search with the classifier and a set of parameters
    '''
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring=scoring,
                               return_train_score=True, verbose = verbose, **kwargs)
    grid_search.fit(X, y)
    scores = grid_search.cv_results_
    best_clf = grid_search.best_estimator_
    print("\n\nThe best parameter set is %s with score %.4f."%(grid_search.best_params_, grid_search.best_score_))
    scores = pd.DataFrame(scores)
    return scores.filter(regex=r'^param_|mean_t|std_t'), best_clf


def calculate_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def model_selection(X, y, clf_list, cv=5, scoring='f1_macro'):
    best_score = 0
    best_clf = None
    for clf in clf_list:
        print('\nNow training %s ...'%(str(type(clf)).split('.')[-1][:-2]))
        s = cross_validation(X, y, clf, cv, scoring)
        if s.mean()>best_score:
            best_score = s.mean()
            best_clf = clf
    print('\nThe best classifer is %s,\nBest f1 macro score = %.4f'%\
          (str(type(best_clf)).split('.')[-1][:-2], best_score))
    return best_clf, best_score


def model_prediction(clf, X_train, y_train, X_test, **kwargs):

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, **kwargs)
    return y_pred


def model_report(y_true, y_pred, y_names=None, confusion=False):
    print (colored('Classification report:', 'magenta', attrs=['bold']))
    print (classification_report(y_true, y_pred, target_names=y_names))
    if confusion:
        print (colored('Confusion Matrix:', 'magenta', attrs=['bold']))
        plot_confusion_matrix(confusion_matrix(y_true, y_pred), y_names)


## ************************ deep learning ****************************

# solution 1 :
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1 = []
        self.val_recalls = []
        self.val_precisions = []
        self.f1 = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro', zero_division=0)
        _val_recall = recall_score(val_targ, val_predict, average='macro', zero_division=0)
        _val_precision = precision_score(val_targ, val_predict, average='macro', zero_division=0)
        self.val_f1.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print ("— val_f1: %f — val_precision: %f — val_recall %f -" %(_val_f1, _val_precision, _val_recall))
        return

    def get_f1(self):
        return self.val_f1

    def get_recall(self):
        return self.val_recalls

    def get_precision(self):
        return self.val_precisions


# solution 2: for early stop
# TODO: modify to macro average
def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
        possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + keras.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.sum(keras.round(keras.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+keras.epsilon()))


def get_cnn_model(embedding_matrix, max_sequence_len, num_classes, random_state=42):
    np.random.seed(random_state)
    keras.clear_session()

    model = Sequential()

    # input layer
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                        input_length=max_sequence_len,
                        weights=[embedding_matrix]))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_metric])
    print(model.summary())
    return model

def model_train(model, X_train, y_train, X_test, y_test, metrics, eary_stop = True, **kwargs):
    if eary_stop:
        # train the model
        earlystop = EarlyStopping(monitor='val_f1_metric', min_delta=0.05,
                              patience=0, verbose=1, mode='min')
        callbacks_list = [earlystop]
    else:
        callbacks_list = None

    model_history = model.fit(X_train, y_train,
                              validation_data=(X_test, y_test),
                              callbacks=callbacks_list+[metrics],
                              **kwargs)

    return model_history

def model_evaluate(model, X_test, **kwargs):
    '''Predict class with test set'''
    y_pred =  model.predict_classes(X_test, **kwargs)
    return y_pred

# classification_report(np.argmax(y3_test,axis=1), y_pred)
def train_dl_classifier(X, y, model, test_size, epochs, batch_size, verbose=1,
                        y_names=None, confusion=False, random_state=42):
    # train-test split
    print ('test size is: %2.0f%%' % (test_size * 100))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    y3_train = to_categorical(y_train)
    y3_test = to_categorical(y_test)

    print ('Start training...')
    metrics = Metrics()
    model_history = model_train(model, X_train, y3_train, X_test, y3_test,
                                metrics, eary_stop = True,
                                epochs=epochs, batch_size=batch_size, verbose=verbose)
    print('Finished training.')
    f = metrics.get_f1()

    plot_model_history(model_history)

    y_predicted = model_evaluate(model, X_test, batch_size=batch_size, verbose=verbose)
    y_train_pred = model_evaluate(model, X_train, batch_size=batch_size, verbose=verbose)

    model_report(y_test, y_predicted, y_names, confusion)

    test_f1 = calculate_macro_f1(y_test, y_predicted)
    train_f1 = calculate_macro_f1(y_train, y_train_pred)

    return train_f1, test_f1,f

def model_save(model, name):
    model.save('models/'+name+'.h5')

def model_load(path):
    loaded_model = load_model(path)
    return loaded_model

## ************************ visualization ****************************

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for f1
    axs[0].plot(range(1,len(model_history.history['f1_metric'])+1),model_history.history['f1_metric'])
    axs[0].plot(range(1,len(model_history.history['val_f1_metric'])+1),model_history.history['val_f1_metric'])
    axs[0].set_title('Macro F1 score')
    axs[0].set_ylabel('F1')
    axs[0].set_xlabel('Epoch')
    #axs[0].set_xticks(np.arange(1, epochs+1))
    axs[0].legend(['train', 'val'], loc='best')

#    # summarize history for accuracy
#    axs[1].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
#    axs[1].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
#    axs[1].set_title('Model Accuracy')
#    axs[1].set_ylabel('Accuracy')
#    axs[1].set_xlabel('Epoch')
#    #axs[0].set_xticks(np.arange(1, epochs+1))
#    axs[1].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    #axs[1].set_xticks(np.arange(1, epochs+1))
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def plot_confusion_matrix(confusion_matrix, label):
    plt.figure(figsize=(10,10))
    g = sns.heatmap(confusion_matrix, annot=True,cmap='Greens',
                    cbar=False, xticklabels=True,yticklabels=True, fmt='g')
    g.set_yticklabels(label, rotation =0)
    g.set_xticklabels(label, rotation =90)
    plt.title('Confusion Matrix')

    plt.show()


def plot_feature_pca(X, y, y_names):

    if not isinstance(X,np.ndarray):
        X = X.toarray().copy()

    pca = PCA(n_components = 2)
    x_pca = pca.fit_transform(X)

    # visualization
    hot = plt.get_cmap('tab20')
    cNorm  = colors.Normalize(vmin=0, vmax=len(y_names))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

    # Plot each object
    plt.figure(figsize=(16,8))
    for i in range(len(y_names)):
        indx = y == i
        plt.scatter(x_pca[indx,0], x_pca[indx,1], s=10, color=scalarMap.to_rgba(i), label=y_names[i])
    plt.xlabel('pca1')
    plt.ylabel('pca2')
    #plt.xlim(-0.4,0.6)
    plt.legend()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:41:04 2020

@author: xchen

Define document class and related methods. 
"""

# Dependencies
import re
from io import TextIOWrapper
import copy
#import warnings
#import os

class document:
    
    def __init__(self, path, topic):
        '''Document class
        
        Attributes:
        -----------
        path : str
            Path of document file.
        topic : str
            Topic label of document, performed as the target for ducument classifer.
        group : str
            First-level category of document, values include ['alt', 'comp', 
            'misc', 'rec','sci', 'soc', 'talk' ].
        rawdata : list
            List of document lines before preprocessing.
        header : str
            Meta data at the top of document. If the document contains a header, 
            you should explicitly pass ``header_seperator`` parameter while 
            calling ``load_document`` function
        body : str
            Body text of the document.
        lines_of_body : int
            Number of non-empty lines in body text.
        clean : bool
            Flag that indicates body text has been cleaned or nort.
        '''
        
        self.path = path
        self._topic = topic
        self._group = self._topic.split('.')[0]
        
        self.rawdata = []
        self.header = ''
        self.body = ''
        
        self.lines_of_body = 0
        self.clean = False
        
    def parser(self, file, header_seperator = None):
        '''Load a document as a list of lines'''
        
        if not isinstance(file, TextIOWrapper):
            file = TextIOWrapper(file, encoding='ascii', errors='surrogateescape')
        document = file.readlines()
        document = ''.join(line.strip(' ') for line in document)
        self.rawdata = document
        self.__parseheader(header_seperator)
        self.__countlines()
        if self.lines_of_body == 0:
            print('Body text from %s is empty.'%(self.path))
            #warnings.warn(msg)
        return self
            
    def clean_text(self, min_word_len = 3, max_word_len = 20, 
                   word_split_regex = '\w+|\$[\d\.]+|\S+', word_ignore_regex = '\\s+',
                   lower_case = True, contraction_dict = None, 
                   stop_words = None, stemmer = None, remove_header = True):
        '''Clean text data and tokenize sentences if required
        
        Parameters:
        -----------
        min_word_len : int, default 3
            The minimal length of words after cleaning. Words shorter than the 
            length will be filtered out.
        max_word_len : int, default 20
            The maximal length of words after cleaning. Words longer than the 
            length will be filtered out.
        word_split_regex : str, default '\w+|\$[\d\.]+|\S+'
            Remains the words that match the regular expression. The default value
            will only remain words with digits or characters and dollar values.
        word_ignore_regex : str, default '\\s+'
            Removes the words that match the regular expression. The default value
            will filter all white characters.
        lower_case : boolen, dafault True
            Lowercase all letters.
        contractoin_dict : dict, default None
            A dictionary that will be used to replace the key with the value
        stop_words : list, default None
            A list of stop words to be filtered out
        stemmer : object, default None
            A stemmer object used to stem words
        remove_header : bool, default True
            Exclude headers after cleaning data
        '''
        # initial cleaning
        if not self.clean:
            
            # filter out the first sentence and email address            
            first_line_regex1 = r"(?:In article)[\s\S]*(?:writes|wrote)\:[\n]*"
            first_line_regex2 = r"^.*(?:writes|wrote)\:[\n]"
            email_regex = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
            text = re.sub(first_line_regex1, '', self.body) 
            text = re.sub(first_line_regex2, '', text) 
            text = re.sub(email_regex, '', text) 
            
            # calculate number of special characters and numbers
            self.num_of_punctuations = self.__countregex(r"[^\w\s]", text) # excluding blank characters
            self.num_of_special_char = self.__countregex(r"[^\w\s.',->*]", text) # excluding blank and common characters
            self.num_of_numbers = self.__countregex(r"(?:\-|\+)?\d+(?:\.\d+)?", text) # numbers
        
            text = text.replace('\s',' ') # replace blank characters with space
            text = re.sub('\s{2,}', ' ', text) # remove continuous spaces
            
            self.clean = True
        else:
            text = self.body
            
        # lowercase all characters
        if lower_case:
            text = text.lower()
        
        # clean default regex contractions or retrieve contraction_dict
        if contraction_dict is not None:
            if not isinstance(contraction_dict,dict):
                text = re.sub(r"what's", "what is ", text)
                text = re.sub(r"\'s", " ", text)
                text = re.sub(r"\'ve", " have ", text)
                text = re.sub(r"n't", " not ", text)
                text = re.sub(r"i'm", "i am ", text)
                text = re.sub(r"\'re", " are ", text)
                text = re.sub(r"\'d", " would ", text)
                text = re.sub(r"\'ll", " will ", text)
                text = re.sub(r"\'ll", " will ", text)
            else:
                # TODO: retrieve contraction_dict
                pass
        
        # split sentence into words which match word_split_regex
        words = [w for w in re.findall(word_split_regex, text) \
                 if len(w)>=min_word_len and len(w)<=max_word_len]
        
        # remove words which match word_ignore_regex
        words = [w for w in words if re.match(word_ignore_regex,w) is None]
        
        # filter stop words
        if stop_words is not None:
            words = [w for w in words if not w in stop_words]
        
        #stem
        if stemmer is not None:
            try:
                words = [stemmer.stem(w) for w in words]
            except AttributeError:
                print ('Stemming is unavalible')
                
        # update body text after cleaning
        self.body = ' '.join(w for w in words)
        self.__countwords()

    
    def copy(self, deep_copy = True):
        '''Implement deep copy and shallow copy'''
        
        if deep_copy:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
             
        
    def get(self, varnames):
        '''Return the value of attributes given a list of attribute names
        Typically used for feature selection
        '''
        if isinstance(varnames, list):
            var_list = [getattr(self, var)for var in varnames]
            return var_list if len(var_list) >1 else var_list[0]
        elif isinstance(varnames, str):
            return getattr(self,varnames)
    
    def __parseheader(self, header_seperator = None):        
        '''Parse structural data and retrieve the body content'''
        
        # Seperate meta data at the top of each document
        if header_seperator is not None:
            index = self.rawdata.find(header_seperator)
            if index >=0:
                self.body = (self.rawdata[index+1:]).strip()
                self.header = (self.rawdata[:index]).strip()
            else:
                print('cannot find header seperator in raw text')
                self.body = self.rawdata
                self.header = ''
        else:
            self.body = self.rawdata
            self.header = ''
    
    def __countlines(self):
        if len(self.body)>0:
            self.lines_of_body = len([i for i in self.body.split('\n') if i !=''])#number of non-empty lines
        else:
            self.lines_of_body = 0
    
    def __countwords(self):
        if len(self.body)>0:
            self.num_of_words = len([i for i in self.body.split(' ') if i !=''])#number of non-empty words
        else:
            self.num_of_words = 0
    
    def __countregex(self, regex, text = None):
        if text is None:
            text = self.body
        return len(re.findall(regex, text))
    

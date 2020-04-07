
## Text Classifier with python



### Introduction

This project constructs a text classifier to label textual documents based on topics. 

Run `ML.py` to test machine learning models including Naive Bayes, Logitic Regression and SVM.

Run `DL.py` to test deep learning model consiting of CNN and LSTM.

The data is assumed to be stored in the `..\data\` directory by default.  While running `ML.py` and `DL.py`, it asks you for the root of dataset. You can press `return` to access default directory or supply your own dataset if it has similar directory structure.


### Requirements

* python 3.x

* python modules:

  * scikit-learn
  * keras
  * numpy
  * scipy
  * collections
  * colorama
  * termcolor
  * matplotlib
  * seaborn
  * nltk (for stop words)


### The code


The [EDA.ipynb](EDA.ipynb) notebook showcases the data preprocessing and EDA processes. 

The [Experiments-ML.ipynb](Experiments-DL.ipynb) notebook showcases the parameter tuning of machine learning models. 

The [Experiments-DL.ipynb](Experiments-DL.ipynb) notebook showcases the training of deep learning models. 

The [ML.py](ML.py) file runs the process to select and evaluate machine learning models.

The [DL.py](ML.py) file runs process to train and evaluate the deep learning model.

The [document.py](document.py) file contains the definition of document class.

The [pipeline.py](pipeline.py) file contains all functions utilized in the project


#### Running the code

	python ML.py
    	python DL.py

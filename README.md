# Sentiment Analysis using LSTM

Sentiment Analysis is a process of determining whether a given statement is positive, negative or neutral. A basic task in sentiment analysis is classifying the polarity of a given text of the document, sentence, or feature/aspect level—whether the expressed opinion in a document, a sentence or an entity feature/aspect is positive, negative, or neutral. 

[Proposal Report](https://github.com/anirudhramanan/sentiment-analysis-lstm/blob/master/Sentiment%20Analysis%20Proposal.pdf)

## Downloading Data-set

The dataset has been downloaded from http://ai.stanford.edu website. This dataset contains movie reviews along with their associated binary sentiment labels. The core dataset contains 50,000 reviews (25K positive and 25K negative, balanced distribution) split evenly into 25K training and 25K test sets. 

You can run the `data_download.py` script in the scripts directory to download the data set.

## Data Preprocessing

The dataset contains movie reviews along with their associated binary sentiment labels. The core dataset contains 50,000 reviews (25K positive and 25K negative, balanced distribution) split evenly into 25K training and 25K test sets. Using the files in the dataset, I have created a train.csv file which contains two features (one is the file_path, and second is the target ie 0 for negative and 1 for positive), which will be used to load the dataset.

You can run the `data_prep.py` script in the scripts directory to preprocess the data set.

# LSTM Graph 

# Requirements

In order to run [the iPython notebook](Oriole-LSTM.ipynb), you'll need the following libraries.

* **[TensorFlow](https://www.tensorflow.org/install/)
* **[[NumPy](https://docs.scipy.org/doc/numpy/user/install.html)
* **[[Jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
* **[[matplotlib](https://matplotlib.org/)

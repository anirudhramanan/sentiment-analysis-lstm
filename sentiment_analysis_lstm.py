
# coding: utf-8

# # Sentiment Analysis using LSTM

# In this notebook we will be implementing a LSTM network to classify sentiments from the statements. The model will be trained on a 25k row dataset of both positive and negative reviews.
# 
# This is what the overall architecture looks like:
# 
# ![Screen%20Shot%202018-07-18%20at%201.38.42%20PM.png](attachment:Screen%20Shot%202018-07-18%20at%201.38.42%20PM.png)
# 
# As described above, the words will be passed to the embedding layer, which will convert the words into vectors so that it can be passed as an input to the LSTM network. We will go in detail as we progress.
# 
# Let's start by importing libraries that we need

# In[184]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import libraries
import csv
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import callbacks


# # Downloading Data

# Run the following cell to download the dataset
# 
# 

# In[ ]:


# get_ipython().magic(u"run -i 'data_download.py'")


# # Data Preparation
# 
# The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). This means we have 12.5k files of positive and 12.5 files of negative reviews in the train set as well in the test set.
# 
# For ease of use, let's create a csv file for the train set which will contain the file name and the sentiment associated with it. 

# In[155]:


# get_ipython().magic(u"run -i 'data_prep.py'")


# The above script will loop through all the text files present in the pos and neg directory in the training set, and will create the csv file with the filename against the sentiment. 
# 
# The csv file will look something like this
# 
# ![Screen%20Shot%202018-07-18%20at%202.19.23%20PM.png](attachment:Screen%20Shot%202018-07-18%20at%202.19.23%20PM.png)

# Let's print one of the reviews to understand how the dataset looks like

# In[156]:


f = open('./aclImdb/train/pos/4715_9.txt','r')
message = f.read()
print(message)


# ### Loading Data

# In[157]:


training_reviews = []
test_reviews = []
target = []

# load training data
with open("./aclImdb/train/train.csv") as fd:
    rd = csv.reader(fd, delimiter=",", quotechar='"')
    for file_name, label in rd:
        if (file_name.endswith('.txt')):
            path = './aclImdb/train/pos/' if label is '1' else './aclImdb/train/neg/'
            file = open(path+file_name, "r") 
            training_reviews.append(file.read())
            target.append(label)

# load test data
with open("./aclImdb/test/test.csv") as fd:
    rd = csv.reader(fd, delimiter=" ", quotechar='"')
    for file_name in rd:
        if (file_name[0].endswith('.txt')):
            file = open('./aclImdb/test/'+file_name[0], "r") 
            test_reviews.append(file.read())
            
# print number of training reviews
print("Training reviews: {}".format(len(training_reviews)))
print("Training Targets: {}".format(len(target)))

# print number of test reviews
print("Test reviews: {}".format(len(test_reviews)))


# ### Embedding Layer (Encoding words to vectors)
# 
# It's input is a text corpus and its outputs a set of vectors i.e it turns text into numerical form that the neural network can understand. To create word embeddings, we will load the entire reviews (negative and positive) into a single variable, which can be then fed into the embedding layer to generate vectors.

# In[158]:


#Process the training data set, and create word arrays from the reviews
import re

processed_training_review = []

for review in training_reviews:
    # this is done to cleanup and remove special characters from the dataset.
    # This will remove all special characters such as brackets, quotes, etc.
    processed_training_review.append(re.sub('[^ a-zA-Z0-9]', '', review).lower())

# print the first row
print(processed_training_review[:1])

# join the rows as a string with '/n' as delimiter
all_train_review =' /n '.join(processed_training_review)

# split each reviews of the training dataset and join them as a string
train_reviews = all_train_review.split(' /n ')
all_train_review = ' '.join(train_reviews)

# split each word of the training dataset in the string to a list
train_words = all_train_review.split()
print(len(train_words))


# In[159]:


#Process the test data set, and create word arrays from the reviews
import re

processed_test_review = []
    
for review in test_reviews:
    # this is done to cleanup and remove special characters from the dataset.
    # This will remove all special characters such as brackets, quotes, etc.
    processed_test_review.append(re.sub('[^ a-zA-Z0-9]', '', review).lower())

# print the first row
print(processed_test_review[:1])

# join the rows as a string with '/n' as delimiter
all_test_review =' /n '.join(processed_test_review)

# split each reviews of the training dataset and join them as a string
test_reviews = all_test_review.split(' /n ')
all_test_review = ' '.join(test_reviews)

# split each word of the training dataset in the string to a list
test_words = all_test_review.split()
print(len(test_words))


# In[160]:


# combine the training and test words
total_words = train_words + test_words

print(len(total_words))


# Now that we have the reviews, we can start creating the word embeddings. This will convert the words present in the reviews into integers which can later be fed into the neural network.

# In[161]:


from collections import Counter
counts = Counter(total_words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

train_reviews_integers = []
for review in train_reviews:
    train_reviews_integers.append([vocab_to_int[word] for word in review.split()])
    
test_reviews_integers = []
for review in test_reviews:
    test_reviews_integers.append([vocab_to_int[word] for word in review.split()])


# Printing the integer mapping for the review words:

# In[162]:


train_reviews_integers[:10]


# In[163]:


review_lens = Counter([len(x) for x in train_reviews_integers])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# So, there are no zero-length reviews in our dataset. But, the maximum review length is way too much for the RNN to handle, we have to trim this down to let's say 220. For reviews longer than 220, it will be truncated to first 220 characters, and for reviews less than 220 we will add padding of 0's

limit = 220

# # Training and Validation

# We will split the training set into training and validation set. The validation set is used to evaluate a given model, but this is for frequent evaluation.
# 
# Commonly, 80 % of the whole training data set is used for training, and rest 20 % for the validation.

# In[187]:


# use 0.2 of the data set as validation set
split_factor= 0.8
split_index = int(len(train_reviews_integers)*0.8)

# setup training and validation set
x_train = sequence.pad_sequences(train_reviews_integers[:split_index], maxlen=limit)
x_val = sequence.pad_sequences(train_reviews_integers[split_index:], maxlen=limit)

y_train = np_utils.to_categorical(target[:split_index], 2)
y_val = np_utils.to_categorical(target[split_index:], 2)

print(split_index)

# setup test set
x_test = sequence.pad_sequences(test_reviews_integers, maxlen=limit)


# In[166]:


# print the shape of training set
print(x_train.shape)


# In[167]:


n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1


# # Building Model

# Let's start by defining hyperparameters for the model

# In[168]:


# number of hidden layers in the LSTM network
lstm_size = 256

# number of LSTM layers in the neural network
lstm_layers = 1

# Number of data to be fed into the network during the training period. Incase of OOM, we will have to decrease
# batch size to take in lesser number of reviews.
batch_size = 32

# embedding size
embed_size = 300


# Once the hyperparameters have been defined, we will build the tensorflow graph using the session api's. During the training, we might need to tune the hypermeters to get the best result/accuracy.

# In[188]:


tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,
                          write_graph=True, write_images=True)

# create a sequential model
model = Sequential()

# embedding layer
model.add(Embedding(len(total_words), embed_size, input_length=limit, dropout=0.2))

# conv 1d
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

# max pool
model.add(MaxPooling1D(pool_size=2))

# 1 layer of 100 units in the hidden layers of the LSTM cells
model.add(LSTM(100))

# dense layer
model.add(Dense(2, activation='softmax'))

#compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, verbose=1, batch_size=2, callbacks=[tensorboard])


# In[189]:


# Final evaluation of the model # Final  
scores = model.evaluate(x_val, y_val, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))


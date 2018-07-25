
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

# In[13]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import libraries
import re
import csv
import numpy as np
import tensorflow as tf
from string import punctuation


# # Downloading Data

# Run the following cell to download the dataset
# 
# 

# In[3]:


# %run -i 'scripts/data_download.py'


# # Data Preparation
# 
# The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). This means we have 12.5k files of positive and 12.5 files of negative reviews in the train set as well in the test set.
# 
# For ease of use, let's create a csv file for the train set which will contain the file name and the sentiment associated with it. 

# In[7]:


# %run -i 'scripts/data_prep.py'


# The above script will loop through all the text files present in the pos and neg directory in the training set, and will create the csv file with the filename against the sentiment. 
# 
# The csv file will look something like this
# 
# ![Screen%20Shot%202018-07-18%20at%202.19.23%20PM.png](attachment:Screen%20Shot%202018-07-18%20at%202.19.23%20PM.png)

# Let's print one of the reviews to understand how the dataset looks like

# In[5]:


# f = open('./aclImdb/train/pos/4715_9.txt','r')
# message = f.read()
# print(message)


# ### Loading Data

# In[8]:


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

# In[14]:


processed_training_reviews = []

for review in training_reviews:
    # remove special characteres.
    review = re.sub('[^ a-zA-Z0-9]', '', review).lower()
    processed_training_reviews.append(review)

all_text = ' '.join(processed_training_reviews)

print(all_text[:2000])
# split the reviews into word array
words = all_text.split()


# In[15]:


# print the word integer array
print(words[:100])


# In[16]:


# print some of the reviews
processed_training_reviews[:1]


# Now that we have the reviews, we can start creating the word embeddings. This will convert the words present in the reviews into integers which can later be fed into the neural network.

# In[17]:


from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []
for each in processed_training_reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])


# Printing the integer mapping for the review words:

# In[14]:


reviews_ints[:10]


# Let's check if the reviews are of zero length, so that we can remove it from the training set

# In[18]:


review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# So, there are no zero-length reviews in our dataset. But, the maximum review length is way too much for the RNN to handle, we have to trim this down to let's say 220. For reviews longer than 220, it will be truncated to first 220 characters, and for reviews less than 220 we will add padding of 0's

# In[19]:


# trim characters to first 220 characters
limit = 200
features = np.zeros((len(reviews_ints), limit), dtype=int)
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:limit]


# # Training and Validation

# We will split the training set into training and validation set. The validation set is used to evaluate a given model, but this is for frequent evaluation.
# 
# Commonly, 80 % of the whole training data set is used for training, and rest 20 % for the validation.

# In[20]:


split_fraction = 0.8

split_idx = int(len(features)*split_fraction)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = np.array(target)[:split_idx], np.array(target)[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]


# In[21]:


# print the shape of training set
print("Train set: \t\t{}".format(train_x.shape))


# In[22]:


# print the shape of validation set
print("\nValidation set: \t{}".format(val_x.shape))


# In[23]:


# Adding 1 because we use 0's for padding, dictionary started at 1
n_words = len(vocab_to_int) + 1


# Define a helper method to get batches. This takes in the x and y attribute, and the batch size (32 by default)

# In[24]:


def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# # Building Model

# Let's start by defining hyperparameters for the model

# In[25]:


# number of hidden layers in the LSTM network
lstm_size = 256

# number of LSTM layers in the neural network
lstm_layers = 1

# Number of data to be fed into the network during the training period. Incase of OOM, we will have to decrease
# batch size to take in lesser number of reviews.
batch_size = 250

# embedding size
embed_size = 300

# learning rate of 0.001
learning_rate = 0.001


# Once the hyperparameters have been defined, we will build the tensorflow graph using the session api's. During the training, we might need to tune the hypermeters to get the best result/accuracy.

# In[26]:


# create tensorflow graph
graph = tf.Graph()

# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# ### Embedding
# 
# Now we'll add an embedding layer. An embedding is a mapping from discrete objects, such as words, to vectors of real numbers. To create word embeddings in TensorFlow, we first split the text into words and then assign an integer to every word in the vocabulary.

# In[27]:


with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# ### LSTM Cell
# 
# Define the LSTM cell. This will not actually start the training, this is just defining the graph.

# In[28]:

def get_a_cell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# ### Dynamic RNN
# 
# Now we need to actually run the data through the RNN nodes. You can use tf.nn.dynamic_rnn to do this.

# In[29]:


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


# ### Output

# In[30]:


with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ### Validation Accuracy
# 
# Here we can add a few nodes to calculate the accuracy which we'll use in the validation pass.

# In[31]:


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# # Training
# 
# This is the training step.

# In[32]:


epochs = 100

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.8,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/sentiment.ckpt")


# # Test
# 
# Let's run the model on some test sentences

# In[31]:


test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# # Benchmark Model
# 
# I will be using random forest classifier as a benchmark model which will be trained on the vectors created from the words in the sentences, which in turn will be used to predict and classify the sentiment of a review.

# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train_x, train_y)

predicted_y = clf.predict(val_x)

accuracy_score(val_y, predicted_y)


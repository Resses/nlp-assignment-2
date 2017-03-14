import string
import numpy as np
import tensorflow as tf
import random

# Open file
with open('data/training/training-data.1m') as f:
    train_data = [x.strip('\n') for x in f.readlines()]
print('Data loaded.')

# remove punctuation and lower case all words
def prep(data):
    return [x.translate(None, string.punctuation).lower() for x in data]

train_data = prep(train_data)
print('Data preprocessed.')

# Create dictionary
def create_dict(data):
    dictionary = {}
    for line in data:
        for word in line.split():
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    return dictionary

dictionary_counts = create_dict(train_data)
print('Dictionary created with len',len(dictionary_counts))
total_counts = sum(dictionary_counts[word] for word in dictionary_counts)

# Option 1: Dictionary prune only
def prune_dict(dictionary, threshold):
    pruned_dict = {'UNK':0, '<s>':1, '</s>':2}
    count = 3
    for word in dictionary:
        if dictionary[word] > threshold:
            pruned_dict[word] = count
            count +=1
    return pruned_dict

def get_avg_count(dictionary):
    return sum(dictionary[word] for word in dictionary) / float(len(dictionary))

# Option 2: Dictionary prune and subsampling
# returns true p percent of the time
def should_remove(p):
    return True if random.random() < p else False

# Remove words based on probablility determined by frequency
# The dictionary returned will hold an index/ sequence of the words
def prune_and_subsample_dict(dictionary, total_counts, prune_threshold, t = 1e-5):
    new_dict = {'UNK':0, '<s>':1, '</s>':2}
    count = 3
    for word in dictionary:
        if dictionary[word] > prune_threshold:
            f = dictionary[word]/float(total_counts) # the word's fequency
            #p = ((f-t)/f) - np.sqrt(t/f) # probability that word will get removed.
            p = 1 - np.sqrt(t/f)
            if not should_remove(p):
                new_dict[word] = count # set it = dictionary[word] to store counts instead of indexes
                count +=1
    return new_dict

# Pick based on the parameters
sub_sampling_threshold = 0
prune_threshold = 10
if sub_sampling_threshold == 0:
    word_pruned_dictionary = prune_dict(dictionary_counts, prune_threshold)
    print("Dictionary pruned with threshold:", prune_threshold)
else:
    word_pruned_dictionary = prune_and_subsample_dict(dictionary_counts, total_counts, prune_threshold, sub_sampling_threshold)
    print("Dictionary pruned and subsampled with threshold:", prune_threshold, ' and ',sub_sampling_threshold)

del dictionary_counts

# Replace the word in the training sentences with the dictionary IDs.
data_words_coded = []
count_zeros = 0
for sentence in train_data:
    # Split sentence into words
    words = sentence.split()
    data_words = []
    for word in words:
        if word in word_pruned_dictionary:
            data_words.append(word_pruned_dictionary[word])
        else:
            data_words.append(0)
            count_zeros += 1
    data_words_coded.append(data_words)
print("Dataset words replaced with IDs")

# Create the batch,label pairs as (context, word).
# We pad the sentences using the special words <s>: 1 and </s>: 2
skip_window_size = 5
train_data_words = []
train_data_context = []
for sentence in data_words_coded:
    for pos,word in enumerate(sentence):
        context = sentence[max(pos-skip_window_size,0):pos] + sentence[pos+1:min(pos+skip_window_size+1, len(sentence))]
        context += [1] * (pos+skip_window_size+1 - len(sentence)) # Manually padding with <s> for the beginning of sentence
        context += [2] * (-pos+skip_window_size) # Manually padding with </s> for the end of sentence
        train_data_context.append(context)
        train_data_words.append(word)
print("Batch and labels created")

del data_words_coded
del train_data

# We are not introducing any positioning in the context. Therefore, both words
# and context dictionaries are the same.
context_pruned_dictionary = word_pruned_dictionary.copy()

# Convert the data structures to use in the NN
train_data_context = np.array(train_data_context) #It's a vector containing lists to use as batch
train_data_words = np.matrix(train_data_words).T #It's a matrix transpose to use as label

# Shuffle the input
idx = np.random.permutation(train_data_context.shape[0])
train_data_context = train_data_context[idx]
train_data_words = train_data_words[idx,:]
print("Data shuffled")

##### Define TensorFlow graph #####
# Dimensions of the embedding
embedding_size = 500

# Size of the batch to be processed in each step
batch_size = 128

# Number of iterations
num_steps = 300001

# Number of "sample" classes to pick randomly.
sampled_size = 30

# Length of the context dictionary
num_labels = len(word_pruned_dictionary)

# Length of the words dictionary
num_features = len(context_pruned_dictionary)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.int32, shape=[batch_size, skip_window_size * 2])
    tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # tf_valid_dataset = tf.constant(valid_dataset[50:60])
    # tf_test_dataset = tf.constant(test_dataset)

    # TODO: Generate values with a -1 to 1 range
    tf_embedding = tf.Variable(tf.truncated_normal([num_features, embedding_size]))
    tf.Variable(tf.random_uniform([num_features, embedding_size], -1.0, 1.0))
    tf_weights_softmax = tf.Variable(tf.truncated_normal([num_labels, embedding_size]))
    tf_bias_softmax = tf.Variable(tf.zeros([num_labels]))

    # We are going to compute a one hot encoding vector of a very large dataset. We save time of
    # unnecesary computation of the product of the vector with almost all zeros and a matrix and
    # just get the matrix value. For CBOW, we sum the embedding of the context of each word
    embed = tf.zeros([batch_size, embedding_size])
    for j in range(skip_window_size * 2):
        embed += tf.nn.embedding_lookup(tf_embedding, tf_train_dataset[:, j])

    # With the Sample Softmax, we are applying the Negative Sampling technique to the optimization
    # equation. Therefore, the optimization takes into account pairs that never occured in the
    # dataset and the convergence is going to be faster. We provide a number of samples from each
    # batch that are going to be randomly reassigned to another class.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf_weights_softmax, biases=tf_bias_softmax,
                                      labels=tf_train_labels, inputs=embed, num_sampled=sampled_size,
                                      num_classes=num_labels))

    # TODO: Use the Noisy
    # loss = tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled)


    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Normalize the embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(tf_embedding), 1, keep_dims=True))
    normalized_embeddings = tf_embedding / norm

##### Define TensorFlow session #####
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    average_loss = 0
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_data_context.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_data_context[offset:(offset + batch_size)]
        batch_labels = train_data_words[offset:(offset + batch_size)]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l = session.run(
          [optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
    output_embeddings = normalized_embeddings.eval()

# Create the reverse dictionary to generate the output file.
reverse_dictionary = dict(zip(word_pruned_dictionary.values(), word_pruned_dictionary.keys()))

f = open('output.txt','w')
for i in range(len(word_pruned_dictionary)):
    f.write(reverse_dictionary[i] + " " + " ".join(str(x) for x in output_embeddings[i]) + "\n")
f.close()

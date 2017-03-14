import string
import numpy as np
import random
import conllu
from conllu.parser import parse_tree

#READ IN DATA
with open('data/training/training-data.1m') as f:
    train_data = [x.strip('\n') for x in f.readlines()]

#PREPROCESS - LOWER CASE AND REMOVE PUNCTUATION
def prep(data):
    return [x.translate(None, string.punctuation).lower() for x in data]

train_data = prep(train_data)

# CREATE A DICTIONARY FROM ALL WORDS IN DATA WITH KEY = WORD, VALUE = COUNT OF FREQUENCY
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
total_counts = sum(dictionary_counts[word] for word in dictionary_counts)
len(dictionary_counts) # vocab size = 299293

# CUT OUT ALL WORDS WITH FREQUENCY COUNTS LESS THAN THE THRESHOLD. REPLACES THOSE WORDS WITH 'UNK' AND STORES AN INDEX VALUE FOR EACH WORD
def prune_dict(dictionary, threshold):
    pruned_dict = {'UNK':0}
    count = 1
    for word in dictionary:
        if dictionary[word] > threshold:
            pruned_dict[word] = count
            count +=1
    return pruned_dict

# returns true p percent of the time
def should_remove(p):
    return True if random.random() < p else False

# Remove words based on probablility determined by frequency
# The dictionary returned will hold an index/ sequence of the words
def prune_and_subsample_dict(dictionary, total_counts, prune_threshold, t = 1e-5):
    new_dict = {'UNK':0}
    count = 1
    for word in dictionary:
        if dictionary[word] > prune_threshold:
            f = dictionary[word]/float(total_counts) # the word's fequency
            #p = ((f-t)/f) - np.sqrt(t/f) # probability that word will get removed.
            p = 1 - np.sqrt(t/f)
            if not should_remove(p):
                new_dict[word] = count # set it = dictionary[word] to store counts instead of indexes
                count +=1
    return new_dict

# SET THRESHOLD FOR PRUNING AND SUBSAMPLING HERE. FOR NO SUBSAMPLING, SET THRESHOLD TO 0
sub_sampling_threshold = 0
prune_threshold = 10
if sub_sampling_threshold == 0:
    word_pruned_dictionary = prune_dict(dictionary_counts, prune_threshold)
else:
    word_pruned_dictionary = prune_and_subsample_dict(dictionary_counts, total_counts, prune_threshold, sub_sampling_threshold)

del dictionary_counts

# We read in the conll file - separating each sentence/entry 
# All lines/words from the same sentence will be in one string
# Returns data: A list where each element is a string containing all lines from the conll file
# that belong to that sentence
def read_conll(file_name):
    with open(file_name) as f:
        data = []
        lines = []
        for line in f.readlines():
            if line == '\n':
                data.append(''.join(lines))
                lines = []
            else:
                lines.append(line.translate(None, string.punctuation).lower())
    f.close()
    return data

def get_mod(tree_node):
    return tree_node.data['deprel'] 

# return the word of a tree node if the word is in the word dictionary
# other wise, return 'UNK'
def get_word(tree_node):
    word = tree_node.data['form']
    if word in word_pruned_dictionary:
        return word
    else:
        return 'UNK'

# Return the context word with the dependency relationship modifier appended
# If word parameter is passed in, then this is an inverse relationship where the parent word is the context word
# We keep the modifier from the child and add -1 to demonstrate inverse.
# Other wise, we use the word and modifier from the node passed in - the child.
# Prep is passed in if the prep modifier was collapsed
def get_word_with_context(child_node, word = None, prep = None):
    if prep != None:
        modifier = "prep_" + prep
    else:
        modifier = child_node.data['deprel'] 
    if word != None:
        modifier = modifier + "/-1"
    else:
        word = get_word(child_node)
    return word + "/" + modifier

# Returns a list of all the tree node's children
def get_children(tree_node):
    return tree_node.children

# This is a recursive algorithm that traverses through the dependency tree and creates the word-context pairs
def process_tree(tree, parent = None, prep = None):
    
    if parent != None: 
        word = get_word(tree)
        word_as_context = get_word_with_context(tree, prep = prep)
        parent_word = get_word(parent)
        parent_word_as_context = get_word_with_context(tree, word = get_word(parent), prep = prep)
        
        #print(parent_word, word_as_context)
        add_to_lists_and_dictionary(parent_word, word_as_context)
        #print(word, parent_word_as_context)
        add_to_lists_and_dictionary(word, parent_word_as_context)
        
    for child in get_children(tree):
    	# if the child's modifier is prep, we skip over it and move on to it's children using 'prep' + w as the modifier for the context
        if get_mod(child) == 'prep':
            w = get_word(child)
            for c in get_children(child):
                process_tree(c, tree, prep = w)
        else:
            process_tree(child, tree)

# Given a word and context pair, the word is appended to the words list and the associated context is appended to the context list
# Instead of adding the actual words, we add their "index" which is saved in the dictionary so that we can have 
# one hot encodings in the neural network later
# To do this, we add the context word to the context dictionary if it is not their yet
def add_to_lists_and_dictionary(word, context):
    words.append(word_pruned_dictionary[word])
    if not context in context_pruned_dictionary:
        context_pruned_dictionary[context] = len(context_pruned_dictionary)
    contexts.append(context_pruned_dictionary[context])

data = read_conll('data/training/training-data.1m.conll')

words = [] # to store the word from the generated pairs
contexts = [] # to store the context from the generated pairs
context_pruned_dictionary = {} # dictionary to hold contexts
for i in range(len(data)):
    tree = parse_tree(data[i])[0]
    process_tree(tree)

train_data_context = np.matrix(contexts).T #It's a matrix transpose to use as label
train_data_words = np.array(words) #It's a vector to use as batch

del words
del contexts

# Shuffle the input
idx = np.random.permutation(train_data_words.shape[0])
train_data_context = train_data_context[idx,:]
train_data_words = train_data_words[idx]

# Dimensions of the embedding
embedding_size = 500

# Size of the batch to be processed in each step
batch_size = 128

# Number of iterations
num_steps = 150001

# Number of "sample" classes to pick randomly. 
sampled_size = 20

# Length of the context dictionary
num_labels = len(context_pruned_dictionary)  

# Length of the words dictionary
num_features = len(word_pruned_dictionary) 

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    tf_embedding = tf.Variable(tf.truncated_normal([num_features, embedding_size]))
    tf.Variable(tf.random_uniform([num_features, embedding_size], -1.0, 1.0))
    tf_weights_softmax = tf.Variable(tf.truncated_normal([num_labels, embedding_size]))
    tf_bias_softmax = tf.Variable(tf.zeros([num_labels]))

    # We are going to compute a one hot encoding vector of a very large dataset. We save time of
    # unnecesary computation of the product of the vector with almost all zeros and a matrix and
    # just get the matrix value.
    embed = tf.nn.embedding_lookup(tf_embedding, tf_train_dataset)

    # With the Sample Softmax, we are applying the Negative Sampling technique to the optimization
    # equation. Therefore, the optimization takes into account pairs that never occured in the 
    # dataset and the convergence is going to be faster. We provide a number of samples from each
    # batch that are going to be randomly reassigned to another class.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf_weights_softmax, biases=tf_bias_softmax,
                                      labels=tf_train_labels, inputs=embed, num_sampled=sampled_size,
                                      num_classes=num_labels))
    
    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Normalize the embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(tf_embedding), 1, keep_dims=True))
    normalized_embeddings = tf_embedding / norm
  
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    average_loss = 0
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_data_words.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_data_words[offset:(offset + batch_size)]
        batch_labels = train_data_context[offset:(offset + batch_size)]
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

reverse_dictionary = dict(zip(word_pruned_dictionary.values(), word_pruned_dictionary.keys())) 

# write embeddings to file
f = open('skip-pru2-sub1e-5-dep2_100_150k.txt','w')
for i in range(len(word_pruned_dictionary)):
    f.write(reverse_dictionary[i] + " " + " ".join(str(x) for x in output_embeddings[i]) + "\n")
f.close()



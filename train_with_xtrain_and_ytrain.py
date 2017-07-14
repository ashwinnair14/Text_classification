try:
    import sklearn
    import re
    import tensorflow as tf
    import sklearn.preprocessing
    from tensorflow.contrib import learn
    import numpy as np
    import os
    import time
    import datetime
    from text_cnn import TextCNN
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

except :
    print("\n#############################################################\n")
    print("\n Please install all required packages before running \n")
    print("\n Install tensorflow,numpy and sklearn to run this evaluation script\n")
    print("\n#############################################################\n")
    sys.exit()
    
    
#Read  file and remove punctuation and special chars and use only lowercase
def read_file(filename):
    list_data=[]
    regex = re.compile('[^a-zA-Z\d\s]')#https://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string
    with open(filename,mode='r') as f:
        for each_line in f:
            list_data.append(regex.sub('',str(each_line)).lower().replace('\n',''))
    return list_data

def write_2_txt(filename,data_list):
    with open(filename,mode='w') as f:
        f.write('\n'.join(data_list))


#get all data and return it in batches
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


#Read data
data=read_file('xtrain.txt')
labels=list(map(int, read_file('ytrain.txt')))


#create one hot vectors for labels
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(labels)+1))
labels = label_binarizer.transform(labels)


#find max lenght of the line
max_document_length = max([len(x.split(" ")) for x in data])


#Vectorise the data
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vector_data = np.array(list(vocab_processor.fit_transform(data)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(vector_data)))
data = np.array(vector_data)[shuffle_indices]
labels = np.array(labels)[shuffle_indices]

# Split train/test set I choose 80/20 split
index_to_split = int(0.2 * float(len(labels)))
data_val,data_train  = data[:index_to_split], data[index_to_split:]
label_val ,label_train= labels[:index_to_split], labels[index_to_split:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(label_train), len(label_val)))


#Params were decided after a few tries
dropout_keep_prob=0.6
batch_size=512
valdate_on= 50
save_on=200
num_epochs=50
max_document_length=73 #pre calculated
num_classes=12# number of classes
vocab_size=len(vocab_processor.vocabulary_),
embedding_size=100
filter_sizes=[1,4,7]
num_filters=64
l2_reg_lambda=0.01

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_document_length,
            num_classes=num_classes,# number of classes
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_size,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False) # Display number of steps
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        
            

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        Val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        Val_summary_dir = os.path.join(out_dir, "summaries", "Val")
        Val_summary_writer = tf.summary.FileWriter(Val_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(train_summary_dir):
            os.makedirs(train_summary_dir)
        if not os.path.exists(Val_summary_dir):
            os.makedirs(Val_summary_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            
            train_summary_writer.add_summary(summaries, step)

        def Val_ntwk(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, Val_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            sstr="{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
            print(sstr)
            if 1:
                Val_summary_writer.add_summary(summaries, step)


        batches = batch_iter(list(zip(data_train, label_train)), batch_size, num_epochs)
        
        with open(out_dir+"/param.txt",mode='a') as file:
            file.write("Paramaters: \ndropout_keep_prob={}\nbatch_size={}\nvaldate_on={}\nsave_on={}\nnum_epochs={}\nmax_document_length={}\nnum_classes={}\nvocab_size={}\nembedding_size={}\nfilter_sizes={}\nnum_filters={}\nl2_reg_lambda={}\n".format(
                dropout_keep_prob,batch_size,valdate_on,save_on,num_epochs,max_document_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda
            ))
            
        # Training loop. For each batch...
        for batch in batches:
            train_x_batch, train_y_batch=zip(*batch)
            train_step(train_x_batch, train_y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % valdate_on== 0:
                print("\nEvaluation:")
                Val_ntwk(data_val, label_val, writer=Val_summary_writer)
                print("")
            if current_step % save_on == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


print ("done")
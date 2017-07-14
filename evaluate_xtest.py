try:
    import sys
    import tensorflow as tf
    import numpy as np
    import os
    import time
    from text_cnn import TextCNN
    from tensorflow.contrib import learn
    import re
    import csv
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except :
    print("\n#############################################################\n")
    print("\n Please install all required packages before running \n")
    print("\n Install tensorflow,numpy to run this evaluation script\n")
    print("\n#############################################################\n")
    sys.exit()


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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
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

try:
    vocab_path='vocab'
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
except:
    print("\n#############################################################\n")
    print("Please make sure the vocab file is in the current directory")
    print("\n#############################################################\n")
    sys.exit()

try:
    testdata=read_file('xtest.txt')
except:
    print("Please make sure the 'xtest.txt' file is in the current directory")

x_test = np.array(list(vocab_processor.transform(testdata)))

graph = tf.Graph()
with graph.as_default():

    sess = tf.Session()
    with sess.as_default():
        #restore varibles

        check = os.path.join(os.path.dirname(os.path.join(os.path.abspath(__file__))), "best_model/checkpoints")
        try:
            checkpoint_file = tf.train.latest_checkpoint(check)
        except:
            print("\n#############################################################\n")
            print("Please make sure that you have made changes in the /best_model/checkpoints/checkpoint file\n")
            print("Change the 'path/until/this/folder' to where the submission directory is on your pc")
            print("\n#############################################################\n")
            sys.exit()

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), 200, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])



out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ytest_predicted.txt")
print("Saving evaluation to {}".format(out_path))
with open(out_path, 'w') as f:
    f.write("\n".join(map(str,list(np.array(all_predictions,dtype=np.int)))))

print("Done!")
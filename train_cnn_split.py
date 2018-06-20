#! /usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers

# from text_bilinear_kernel_cnn import TextCNN
from text_cnn import TextCNN
from tflearn.data_utils import VocabularyProcessor

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import tqdm
import ipdb
# from .visualization import put_kernels_on_grid
# from tensorflow.data import learn
# from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "./rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

tf.flags.DEFINE_string("dataset","data/stsa.fine", "Data source for data.")

tf.flags.DEFINE_string("word2vec", "./GoogleNews-vectors-negative300.bin", "Data source for pre-trained word2vec.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("early_stopping_step", 15, "Number of early stopping step (default: 15)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
import sysls

word2vec_given = True

# FLAGS(sys.argv)
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ================================================
# Load data
# print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

print("Loading pre-split data:"+FLAGS.dataset)

if FLAGS.dataset=='data/TREC':
    word_to_idx, idx_to_word, x_train, y_train, x_test, y_test, _, _, all_labels = data_helpers.load_data_pre_split(FLAGS.dataset)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=FLAGS.dev_sample_percentage, random_state=1)
else:
    word_to_idx, idx_to_word, x_train, y_train, x_test, y_test, x_dev, y_dev, all_labels = data_helpers.load_data_pre_split(FLAGS.dataset)


# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = VocabularyProcessor(max_document_length)
# # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# del x, y, x_shuffled, y_shuffled

vocab_size = len(word_to_idx) + 1
print("Vocabulary Size: {:d}".format(vocab_size))
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.0001)
        # optimizer = tf.train.AdadeltaOptimizer(0.95)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # optimizer = tf.train.AdadeltaOptimizer(0.001)
        # # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # grads_and_vars = optimizer.compute_gradients(cnn.loss)
        grads_and_vars = [(tf.clip_by_norm(grad, 0, 3.0), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':','_')), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':','_')), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        if word2vec_given:
        # if FLAGS.word2vec:
            # import gensim

            # model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True)

            # import ipdb;ipdb.set_trace()
            # reverse_vocabulary=vocab_processor.vocabulary_._reverse_mapping
            # vocabulary=vocab_processor.vocabulary_._mapping

            reverse_vocabulary = idx_to_word
            vocabulary = word_to_idx

            voc_keys = list(vocabulary.keys())

            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(min_count=1, size=300)
            model.build_vocab([voc_keys])
            model.intersect_word2vec_format(FLAGS.word2vec, binary=True)

            w2vW=np.empty((vocab_size,FLAGS.embedding_dim))
            w2vW[0] = 0
            for i in range(1,len(reverse_vocabulary)):
                w2vW[i]=model[reverse_vocabulary[i]] if reverse_vocabulary[i] in model else 2*(np.random.rand(FLAGS.embedding_dim)-0.5)
            sess.run(cnn.W.assign(w2vW))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 0.5
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return step, loss, accuracy

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...

        best_dev_loss = 9999999
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluating dev set:")
                _ , dev_loss, _ = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")

                ## early-stopping
                if (dev_loss < best_dev_loss):
                    stopping_step = 0
                    best_dev_loss = dev_loss
                else:
                    stopping_step += 1
                if stopping_step >= FLAGS.early_stopping_step:
                    should_stop = True
                    print("Early stopping is trigger at step: {} loss:{}".format(global_step,dev_loss))
                    # run_context.request_stop()
                    break

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


print("\nEvaluating test...\n")

# Evaluation on test set
# ==================================================
print(checkpoint_dir)
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 0.5})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:

    lb = preprocessing.LabelBinarizer()
    lb.fit(all_labels)
    y_test = lb.inverse_transform(y_test)
    correct_predictions = float(sum(all_predictions == y_test))
    # correct_predictions = 0
    # for ip, label in enumerate(y_test):
    #     if label[1]==all_predictions[ip]:
    #         correct_predictions += 1

    print("Total number of test examples: {}".format(len(y_test)))
    test_acc = correct_predictions/float(len(y_test))
    print("Test Accuracy: {:g}".format(test_acc))

# # Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))

with open('out_put_'+FLAGS.dataset+'.txt', 'w') as f:
    f.write("Parameters:"+"\n")
    for attr, value in sorted(FLAGS.__flags.items()):
        f.write(str(attr.upper())+":"+str(value)+"\n")
    f.write("\nTest Accuracy:"+str(test_acc))

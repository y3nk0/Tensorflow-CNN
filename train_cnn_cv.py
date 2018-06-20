#! /usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers

from text_bilinear_kernel_cnn import TextCNN
# from text_cnn import TextCNN
from tflearn.data_utils import VocabularyProcessor

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tqdm
# from .visualization import put_kernels_on_grid
# from tensorflow.data import learn
# from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("dataset","data/subj.all", "Data source for data.")

tf.flags.DEFINE_string("word2vec", "./GoogleNews-vectors-negative300.bin", "Data source for pre-trained word2vec.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

word2vec_given = False

import sys
# FLAGS(sys.argv)
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")

word_to_idx, label, x, y, text_data = data_helpers.load_data_all(FLAGS.dataset)

# # Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = VocabularyProcessor(max_document_length)
# # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# K fold cross validation
test_accs = []

seed = 10
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
count_fold = 1
for train_index, test_index in kfold.split(x_shuffled, y_shuffled):

    x_train, x_test = x_shuffled[train_index], x_shuffled[test_index]
    y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]

    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=FLAGS.dev_sample_percentage, random_state=1)

    # del x, y, x_shuffled, y_shuffled

    vocab_size = len(word_to_idx) + 1

    print("Fold: {:d}".format(count_fold))
    print("Vocabulary Size: {:d}".format(vocab_size))
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
            optimizer = tf.train.AdamOptimizer(1e-4)
            # optimizer = tf.train.AdadeltaOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

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
                  cnn.dropout_keep_prob: 1.0
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
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluating dev set:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            #     if earlystop_flag == 1:
            #         break
            # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            # print("Saved model checkpoint to {}\n".format(path))


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
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        # correct_predictions = float(sum(all_predictions == y_test))
        correct_predictions = 0
        for ip, label in enumerate(y_test):
            if label[1]==all_predictions[ip]:
                correct_predictions += 1

        print("Total number of test examples: {}".format(len(y_test)))
        test_acc = correct_predictions/float(len(y_test))
        print("Test Accuracy: {:g}".format(test_acc))

    test_accs.append(test_acc)

    # # Save the evaluation to a csv
    # predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    # out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    # print("Saving evaluation to {0}".format(out_path))
    # with open(out_path, 'w') as f:
    #     csv.writer(f).writerows(predictions_human_readable)

    count_fold += 1

print("Mean test accuracy: {:g}".format(np.mean(test_accs)))

with open("results.txt", 'w') as f:
    f.write(str(test_accs)+"\n\n")
    f.write(str(np.mean(test_accs)))
f.close()

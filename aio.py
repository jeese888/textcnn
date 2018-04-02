#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import jieba.posseg as jpsg  # 分词 + 词性标注
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("pred_source_file", "./data/pre_keywords_source.txt", "Data source for prediction.")
tf.flags.DEFINE_string("source_file", "./data/keywords_source.txt", "Data source for the title data.")
tf.flags.DEFINE_string("target_file", "./data/keywords_target.txt", "Data source for the department data.")
tf.flags.DEFINE_string("test_source_file", "./data/test_keywords_source.txt",
                       "Data source for the small test title data.")
tf.flags.DEFINE_string("test_target_file", "./data/test_keywords_target.txt",
                       "Data source for the small test department data.")
tf.flags.DEFINE_string("labels_file", "./data/labels.txt", "Data source for the labels data.")
tf.flags.DEFINE_string("vocab_processor_file", "./data/Vocab_Processor", "Vocabulary Processor.")
tf.flags.DEFINE_string("vocabulary_txt_file", "./data/Vocabulary.txt", "Vocabulary text file.")
tf.flags.DEFINE_string("test_vocab_processor_file", "./data/Test_Vocab_Processor", "Vocabulary Processor for test.")
tf.flags.DEFINE_string("test_vocabulary_txt_file", "./data/Test_Vocabulary.txt", "Vocabulary text file for test.")
tf.flags.DEFINE_string("model_dir", "./model/", "the directory of models")
tf.flags.DEFINE_string("sample_arrays_file", "./data/sample_arrays.npz", "integer sample arrays ")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_length", 5, "length of input sequence (default: 8)")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 400, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# pattern parameters
tf.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.flags.DEFINE_boolean("prediction", False, "Set to True to predict the class for input text")

FLAGS = tf.flags.FLAGS


def words2ids(words_list, vocab_dict, max_docment_length):
    ids = []
    for item in words_list:
        if item in vocab_dict and len(ids) < max_docment_length:
            ids.append(vocab_dict[item])
    if len(ids) < max_docment_length:
        for i in range(max_docment_length - len(ids)):
            ids.append(vocab_dict['<UNK>'])
    return ids


def load_data_for_train(source_file, target_file, labels_file):
    """load data for files and process data """
    print("load data ...")
    x_text, y_text, labels_dict = data_helpers.load_data_and_labels(source_file, target_file, labels_file)
    samples_size = len(y_text)
    print('Samples size: {:d}'.format(samples_size))
    # Build vocabulary
    max_document_length = FLAGS.sequence_length
    min_freq = 4
    # 如果上述两个参数有变或者样本有变，则需要手动删除目录中的Vocab_Processor和Vocabulary文件，以便重新生成
    if FLAGS.self_test:
        vocab_processor_file = FLAGS.test_vocab_processor_file
        vocabulary_txt_file = FLAGS.test_vocabulary_txt_file
    else:
        vocab_processor_file = FLAGS.vocab_processor_file
        vocabulary_txt_file = FLAGS.vocabulary_txt_file
    # 保存VocabularyProcessor
    if not gfile.Exists(vocab_processor_file):
        print('create new vocab_processor...')
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=min_freq)
        vocab_processor.fit(x_text)
        vocab_processor.save(vocab_processor_file)
    else:
        print('load the old vocab_processor..')
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_file)
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    # Extract word:id mapping from the object
    vocab_dict = vocab_processor.vocabulary_._mapping
    # 保存词典文本
    if not gfile.Exists(vocabulary_txt_file):
        print('save vocabulary.txt...')
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        with open(FLAGS.vocabulary_txt_file, 'w', encoding='utf-8') as f:
            for item in sorted_vocab:
                tmp_str = str(item[0]) + "\n"
                f.write(tmp_str)
    if gfile.Exists(FLAGS.sample_arrays_file):
        print('loading sample arrays...')
        arrays = np.load(FLAGS.sample_arrays_file)
        x_train = arrays['x_train']
        x_dev = arrays['x_dev']
        y_train = arrays['y_train']
        y_dev = arrays['y_dev']
        return [x_train, x_dev, y_train, y_dev, vocab_processor]

    # 生成样本的数值数组
    print('words to ids ...')
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    ids_list = []
    for item in x_text:
        item = str.split(item)
        ids_list.append(words2ids(item, vocab_dict, max_document_length))
    x = np.array(ids_list)
    # 生成数值化的类别标签
    y = []
    class_num = len(labels_dict)
    for k in y_text:
        tmp_list = [0] * class_num
        tmp_list[labels_dict[k]] = 1
        y.append(tmp_list)
    y = np.array(y)
    # Randomly shuffle data
    print('Randomly shuffle data...')
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_shuffled, y_shuffled
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    # save arrays
    print("save arrays...")
    np.savez(FLAGS.sample_arrays_file, x_train=x_train, x_dev=x_dev, y_train=y_train, y_dev=y_dev)
    return [x_train, x_dev, y_train, y_dev, vocab_processor]


def keyword_extract(input_str, stopwords_list):
    allow_pos_dict = {'n', 'nz', 'nt', 'ns', 'nr', 'nrt', 'ng', 'v', 'vn', 'i', 'l', 'd', 'a', 'z', 'j', 'b',
                      't'}  # 筛选词性表
    swords = jpsg.cut(input_str)
    keywords = []
    for w in swords:
        if w.flag in allow_pos_dict and w.word not in stopwords_list:  # 根据词性和停止词筛选
            keywords.append(w.word)
    return keywords


def load_modules_for_pred(stopwords_file, labels_file, vocab_processor_file, model_file, sess):
    print("load modules for predicting...")
    # load stop words
    print("loading stop words from file...")
    stop_words = []
    with open(stopwords_file, "r", encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.append(line.replace('\r\n', ''))
    # load labels
    print("loading labels dictionary from file...")
    with open(labels_file, "r", encoding='utf-8') as f:
        labels_list = []
        for line in f.readlines():
            labels_list.append(line.replace('\r\n', ''))
    labels_dict = dict(zip(labels_list, range(len(labels_list))))
    # load vocabulary processor
    print('loading the vocab_processor from file...')
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_file)
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    # load textcnn model
    print('modeling...')
    cnn = TextCNN(
        sequence_length=FLAGS.sequence_length,
        num_classes=len(labels_list),
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)
    print("cnn model finished!")
    # create saver for model
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    # loading model parameters
    ckpt = tf.train.get_checkpoint_state(model_file)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("loading model parameters from %s" % ckpt.model_checkpoint_path)

        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("ERROR: Failed to load model parameters.")
        return 2
    return [stop_words, labels_dict, vocab_processor, cnn]
    #  source_texts = list(open(source_file, "r", encoding='utf-8').readlines())
    #  source_texts = [s.strip() for s in source_texts]
    # print("tokenizing and extracting keyword...")
    # ret_text = []
    # for item in source_texts:
    #     keywords = keyword_extract(item, stop_words)
    #     ret_text.append(item + "/" .join(keywords))


def train_with_dev(x_train, x_dev, y_train, y_dev, vocab_processor):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print('modeling...')
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            print("cnn model finished!")
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time())) + "summaries"
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "model", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # create saver for model
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # initialize model
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Initialize model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

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
                if step % 100 == 0:
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

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            debug_step = -1  # 100
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "textcnn.ckpt")
                    path = saver.save(sess, checkpoint_path, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step == debug_step:
                    print('training is over caused by debug_step')
                    break


def prediction(x_text, x_input, vocab_processor, labels_dict):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print('create model...')
            cnn = TextCNN(
                sequence_length=x_input.shape[1],
                num_classes=len(labels_dict),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            print("cnn model finished!")
            # create saver for model
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # loading model parameters
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("loading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("ERROR: Failed to load model parameters.")
                return 2
            # predict
            idx = 0
            for x in x_input:
                print("the text keyword: %s" % x_text[idx])
                idx += 1
                feed_dict = {cnn.input_x: x, cnn.dropout_keep_prob: 1.0}  # 喂数据
                score, pred_value = sess.run([cnn.scores, cnn.predictions], feed_dict)
                for k, v in labels_dict:
                    if v == pred_value:
                        pred_text = labels_dict[pred_value]
                        break
                print("the prediction result: {}.".format(pred_text))


def main(__):
    if FLAGS.self_test:
        print("self_test for textcnn ...")
        x_train, x_dev, y_train, y_dev, vocab_processor = load_data_for_train(FLAGS.test_source_file,
                                                                              FLAGS.test_target_file,
                                                                              FLAGS.labels_file)
        train_with_dev(x_train, x_dev, y_train, y_dev, vocab_processor)
    elif FLAGS.prediction:
        print("predicting with textcnn ...")
       #  x_text, x_input, vocab_processor, labels_dict = load_data_for_pred(FLAGS.pred_source_file, FLAGS.labels_file)
        #  prediction(x_text, x_input, vocab_processor, labels_dict)
    else:
        print("training for textcnn ...")
    x_train, x_dev, y_train, y_dev, vocab_processor = load_data_for_train(FLAGS.source_file,
                                                                          FLAGS.target_file,
                                                                          FLAGS.labels_file)
    train_with_dev(x_train, x_dev, y_train, y_dev, vocab_processor)


if __name__ == "__main__":
    tf.app.run()
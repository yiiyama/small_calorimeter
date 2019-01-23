import os
import importlib
import random
import configparser as cp
import tensorflow as tf
from tensorflow.python.client import timeline
from utils.params import get_num_parameters

def read_config(config_file_path, config_name):
    config_file = cp.ConfigParser()
    config_file.read(config_file_path)
    return config_file[config_name]

class Trainer(object):
    def __init__(self, config_file, config_name):
        config = read_config(config_file, config_name)

        self.model_path = config['model_path']
        self.summary_path = config['summary_path']
        self.train_for_iterations = int(config['train_for_iterations'])
        self.save_after_iterations = int(config['save_after_iterations'])
        self.validate_after = int(config['validate_after'])

        self.training_files = config['training_files_list']
        self.validation_files = config['validation_files_list']
        self.test_files = config['test_files_list']

        model_type = config['model_type']

        module = importlib.import_module('models.' + model_type[:model_type.rfind('.')])
        model_class = getattr(module, model_type[model_type.rfind('.') + 1:])
        self.model = model_class(config)

        try:
            os.makedirs(self.summary_path)
        except OSError:
            pass

    def initialize(self):
        self.model.initialize()
        self.saver_sparse = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model.variable_scope))

    def clean_summary_dir(self):
        print("Cleaning summary dir")
        for the_file in os.listdir(self.summary_path):
            file_path = os.path.join(self.summary_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def get_input_feeds(self, files_list, repeat=True, shuffle_size=None):
        with open(files_list) as f:
            file_paths = [x.strip() for x in f.readlines()]

        random.shuffle(file_paths)

        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP') \
                         .map(self._parse_one_input)

        if hasattr(self.model, 'input_filter'):
            dataset = dataset.filter(self.model.input_filter)

        dataset = dataset.shuffle(buffer_size=self.model.batch_size * 3 if shuffle_size is None else shuffle_size) \
                         .repeat(None if repeat else 1) \
                         .batch(self.model.batch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    def _parse_one_input(self, example_proto):
        parsed_features = tf.parse_single_example(example_proto, dict(self.model.keys_to_features))
        return parsed_features

    def train(self, from_scratch = True):
        self.initialize()

        print("Beginning to train network with parameters", get_num_parameters(self.model.variable_scope))

        if from_scratch:
            self.clean_summary_dir()

        inputs_feed = self.get_input_feeds(self.training_files)
        inputs_validation_feed = self.get_input_feeds(self.validation_files)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        with tf.Session() as sess:
            sess.run(init)

            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

            if not from_scratch:
                self.saver_sparse.restore(sess, self.model_path)
                print("\n\nINFO: Loading model\n\n")
                with open(self.model_path + '.txt', 'r') as f:
                    iteration_number = int(f.read())

                self.model.set_learning_rate(iteration_number)
            else:
                iteration_number = 0

            print("Starting iterations")
            while iteration_number < self.train_for_iterations:
                try:
                    loss, _, summary_data, summary = self.model.train(sess, inputs_feed, iteration_number)
                except (tf.errors.OutOfRangeError, ValueError):
                    break

                summary_writer.add_summary(summary_data, iteration_number)
                print("Training - Iteration %4d: %s" % (iteration_number, summary))

                if iteration_number % self.save_after_iterations == 0:
                    print("\n\nINFO: Saving model\n\n")
                    self.saver_sparse.save(sess, self.model_path)
                    with open(self.model_path + '.txt', 'w') as f:
                        f.write(str(iteration_number))

                if iteration_number % self.validate_after == 0:
                    _, _, summary_data, summary = self.model.validate(sess, inputs_validation_feed)

                    summary_writer.add_summary(summary_data, iteration_number)
                    print("Validation - Iteration %4d: %s" % (iteration_number, summary))

                iteration_number += 1

    def evaluate(self, nbatches = 100):
        self.initialize()
        self.model.init_evaluate()

        inputs_feed = self.get_input_feeds(self.test_files, repeat=False)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            # Load network parameters
            self.saver_sparse.restore(sess, self.model_path)

            ibatch = 0
            while ibatch != nbatches:
                try:
                    self.model.evaluate(sess, inputs_feed)
                except (tf.errors.OutOfRangeError, ValueError):
                    break

                ibatch += 1
                if ibatch % 100 == 1:
                    print(ibatch, 'batches')

            self.model.print_evaluation_result()

    def debug(self, trained = False):
        print('Initializing model')

        self.model.batch_size = 5

        self.initialize()

        print('Parameter count:', get_num_parameters(self.model.variable_scope))

        print('Constructing input feeds')

        inputs_feed = self.get_input_feeds(self.training_files)
        inputs_validation_feed = self.get_input_feeds(self.validation_files)

        print('Running one session')

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            if trained:
                # Load network parameters
                self.saver_sparse.restore(sess, self.model_path)
            else:
                summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

            feed_dict = self.model.make_feed_dict(sess, inputs_feed)

            for tag, tensor in self.model._debug:
                res, = sess.run([tensor], feed_dict = feed_dict)
                print(tag, '=', res)

            if not trained:
                print("\n\nINFO: Saving model\n\n")
                self.saver_sparse.save(sess, self.model_path)

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            loss, _ = sess.run([self.model.loss, self.model.optimizer],
                               options=options, run_metadata=run_metadata, feed_dict=feed_dict)

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('/afs/cern.ch/user/y/yiiyama/www/smallcalo_debug.json', 'w') as out:
                out.write(chrome_trace)


    def get_one_input(self):
        inputs_feed = self.get_input_feeds(self.training_files)

        with tf.Session() as sess:
            inputs_train = sess.run([inputs_feed[key] for key, _ in self.model.keys_to_features])

            return dict((key, inputs_train[i]) for i, (key, _) in enumerate(self.model.keys_to_features))

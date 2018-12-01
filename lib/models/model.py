import tensorflow as tf

class Model(object):
    error = "Trying to instantiate an abstract class"

    def __init__(self, config, var_scope, human_name):
        self.initialized = False

        self.variable_scope = var_scope
        self.human_name = human_name

        self.batch_size = int(config['batch_size'])
        self.learning_rate = float(config['learning_rate'])
        try:
            self.learning_rate_decay = int(config['learning_rate_decay'])
        except:
            self.learning_rate_decay = 0

        self._features = None
        self.keys_to_features = None

        self.placeholders = [] # [(N_batch, ...)]
        self.logits = None # (N_batch, N_classes)
        self.loss = None # scalar
        self.optimizer = None

        self.summary = [] # [(name, scalar)]
        self._summary = None

        self._debug = [] # [(tag, tensor)]

        self.plot_dir = config['plot_dir']
        self.data_dir = config['data_dir']

    def _make_input_map(self):
        self.keys_to_features = []

        for name, dtype, shape in self._features:
            self.keys_to_features.append((name, tf.FixedLenFeature(shape, dtype)))

    def _make_placeholders(self):
        self.placeholders = []
        for name, dtype, shape in self._features:
            self.placeholders.append(tf.placeholder(dtype=dtype, shape=[self.batch_size] + shape))

    def initialize(self):
        if self.initialized:
            print("Already initialized")
            return

        self._make_input_map()
        self._make_placeholders()
        with tf.variable_scope(self.variable_scope):
            self._make_network()
        self._make_loss()
        self._make_summary()

        summary_scalars = []
        for key, value in self.summary:
            summary_scalars.append(tf.summary.scalar(key, value))
        self._summary = tf.summary.merge(summary_scalars)

        self._learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.loss)

        self.initialized = True

    def make_feed_dict(self, sess, next_input):
        inputs = sess.run([next_input[key] for key, _ in self.keys_to_features])
        feed_dict = dict(zip(self.placeholders, inputs))

        if self.learning_rate_decay > 0 and iteration_number > 0 and iteration_number % self.learning_rate_decay == 0:
            self.learning_rate *= 0.5

        feed_dict[self._learning_rate] = self.learning_rate

        return feed_dict

    def train(self, sess, next_input, iteration_number):
        feed_dict = self.make_feed_dict(sess, next_input)

        fetches = [self.loss, self.optimizer, self._summary]
        fetches += [s[1] for s in self.summary]

        res = sess.run(fetches, feed_dict=feed_dict)

        summary = tuple((self.summary[i][0], res[3 + i]) for i in range(len(self.summary)))

        return res[:3] + [summary]

    def validate(self, sess, next_input):
        feed_dict = self.make_feed_dict(sess, next_input)

        fetches = [self.loss, self.optimizer, self._summary]
        fetches += [s[1] for s in self.summary]

        res = sess.run(fetches, feed_dict=feed_dict)

        summary = tuple((self.summary[i][0], res[3 + i]) for i in range(len(self.summary)))

        return res[:3] + [summary]

    def init_evaluate(self):
        pass

    def evaluate(self, sess, next_input):
        feed_dict = self.make_feed_dict(sess, next_input)

        results = sess.run(self._evaluate_targets, feed_dict=feed_dict)
        summary = sess.run([s[1] for s in self.summary], feed_dict=feed_dict)
        summary_dict = dict((self.summary[i][0], summary[i]) for i in range(len(self.summary)))
        self._do_evaluate(results, summary_dict)

    def debug(self, name, tensor):
        self._debug.append((name, tensor))

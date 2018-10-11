import tensorflow as tf

from models.model import Model

NTHRES = 201

class ClassificationModel(Model):
    def __init__(self, config, var_scope, human_name):
        Model.__init__(self, config, var_scope, human_name)

        self.num_classes = int(config['num_classes'])

        self.accuracy = None
        self.confusion_matrix = None

    def _make_input_map(self):
        self.keys_to_features = []

        for name, dtype, shape in self._features:
            self.keys_to_features.append((name, tf.FixedLenFeature(shape, dtype)))

        self.keys_to_features.append(('labels_one_hot', tf.FixedLenFeature((self.num_classes,), tf.int64)))

    def _make_placeholders(self):
        self.placeholders = []
        for name, dtype, shape in self._features:
            self.placeholders.append(tf.placeholder(dtype=dtype, shape=[self.batch_size] + shape))

        self.placeholders.append(tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.num_classes]))

    def _make_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.placeholders[-1]))

        argmax_labels = tf.argmax(self.placeholders[-1], axis=1)
        prediction = tf.argmax(self.logits, axis=1)

        self.debug.append(('placeholders[-1]', self.placeholders[-1]))
        self.debug.append(('argmax_labels', argmax_labels))
        self.debug.append(('logits', self.logits))
        self.debug.append(('prediction', prediction))
        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(argmax_labels, prediction), tf.float32))
        # self.confusion_matrix = tf.confusion_matrix(labels=argmax_labels, predictions=prediction, num_classes=self.num_classes) # not used

        self.softmax = tf.nn.softmax(self.logits)

    def _make_summary(self):
        self.summary = [
            ('Loss', self.loss),
            ('Accuracy', self.accuracy)
        ]

        thresholds = tf.constant([[0.005 * x for x in range(NTHRES)]]) # (1, NTHRES)
        self._evaluate_targets = []

        if self.num_classes == 2:
            truth = tf.gather(tf.cast(self.placeholders[-1], tf.int32), [0], axis=1) # (N_batch, 1)
            prob = tf.gather(self.softmax, [0], axis=1) # (N_batch, 1)

            roc_auc, auc_update = tf.metrics.auc(truth, prob)
            self.summary.append(('ROCAUC', auc_update))

            predictions = tf.greater(prob, thresholds) # (N_batch, NTHRES)

            mask = tf.squeeze(truth) # (N_batch)

            pred_correct = tf.boolean_mask(predictions, mask, axis=0)
            pred_incorrect = tf.boolean_mask(predictions, 1 - mask, axis=0)
    
            self._evaluate_targets.append(pred_correct)
            self._evaluate_targets.append(pred_incorrect)

        else:
            for icls in range(self.num_classes):
                truth = tf.gather(tf.cast(self.placeholders[-1], tf.int32), [icls], axis=1) # (N_batch, 1)
                prob = tf.gather(self.softmax, [icls], axis=1) # (N_batch, 1)
    
                roc_auc, auc_update = tf.metrics.auc(truth, prob)
                self.summary.append(('ROCAUC%d' % icls, auc_update))
    
                predictions = tf.greater(prob, thresholds) # (N_batch, NTHRES)
    
                mask = tf.squeeze(truth) # (N_batch)
    
                pred_correct = tf.boolean_mask(predictions, mask, axis=0)
                pred_incorrect = tf.boolean_mask(predictions, 1 - mask, axis=0)
    
                self._evaluate_targets.append(pred_correct)
                self._evaluate_targets.append(pred_incorrect)

    def init_evaluate(self):
        self.n_correct = []
        self.n_incorrect = []
        self.n_class = []

        for icls in range(self.num_classes):
            self.n_correct.append(np.zeros((NTHRES,), dtype=np.float64))
            self.n_incorrect.append(np.zeros((NTHRES,), dtype=np.float64))
            self.n_class.append(0)

    def _do_evaluate(self, results):
        for icls in range(self.num_classes):
            pred_correct, pred_incorrect = results[icls * 2:icls * 2 + 1]

            n_true = np.shape(pred_correct)[0]
            n_fake = np.shape(pred_incorrect)[0]

            self.n_class[icls] += n_true

            for ient in range(n_true):
                self.n_correct[icls] += pred_correct[ient]

            for ient in range(n_fake):
                self.n_incorrect[icls] += pred_incorrect[ient]

    def print_evaluation_result(self):
        for icls in range(self.num_classes):
            self.n_correct[icls] /= self.n_class[icls]
            self.n_incorrect[icls] /= sum(n for i, n in enumerate(self.n_class) if i != icls)

            pairs = np.concatenate((np.expand_dims(self.n_correct[icls], axis=1), np.expand_dims(self.n_incorrect[icls], axis=1)), axis=1)
            print(pairs)

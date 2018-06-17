import re
from pprint import pprint

import pandas as pd
import numpy as np

from textblob import TextBlob
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import normalize

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback

from utils import print_step
from preprocess import normalize_text
from cache import get_data, is_in_cache, save_in_cache

#####Thomas
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

from keras.layers import K, Activation
from keras.engine import Layer, InputSpec
import tensorflow as tf

from keras.layers import add, dot
from itertools import combinations



class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


EMBEDDING_FILES = {'crawl': 'cache/crawl/crawl-300d-2M.vec',
                   'glove': 'cache/glove/glove.840B.300d.txt',
                   'twitter': 'cache/twitter/glove.twitter.27B.200d.txt',
                   'lexvec': 'cache/lexvec/lexvec.commoncrawl.300d.W.pos.vectors',
                   'fasttext': 'cache/wiki/wiki.en.vec'}
#                   'local': 'cache/local_fasttext_model.vec'}
EMBED_SIZE_LOOKUP = {'crawl': 300,
                     'glove': 300,
                     'twitter': 200,
                     'lexvec': 300,
                     'fasttext': 300,
                     'local': 100}


def tokenize_and_embed(train_df, test_df, embedding_file, max_features, maxlen, embed_size, label):
    X_train = train_df['comment_text'].fillna('peterhurford').values
    X_test = test_df['comment_text'].fillna('peterhurford').values

    print_step('Tokenizing data...')
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    x_train = tokenizer.texts_to_sequences(X_train)
    x_test = tokenizer.texts_to_sequences(X_test)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    print_step('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print_step('Defining pre-trained embedding (' + label + ')')
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))
    if label != 'local':
        print_step('Defining local embedding')
        local_embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('cache/local_fasttext_model.vec'))

    print_step('Defining tokenization - embedding scheme')
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    if label != 'local':
        local_embedding_matrix = np.zeros((nb_words, 100))
        print_step('Calculating pre-trained <-> local shared words')
        shared_words = np.intersect1d(embeddings_index.keys(), local_embeddings_index.keys())
        reference_matrix = np.array([local_embeddings_index.get(w) for w in shared_words])
        reference_matrix = normalize(reference_matrix).T

    non_alphas = re.compile(u'[^A-Za-z]+')
    stemmer = PorterStemmer()

    print_step('Beginning embedding')
    for word, i in word_index.items():
        if i >= max_features: continue
        # First try to find the embedding vector as-is
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            # Second, try to replace in' -> ing
            print("in' -> ing")
            word = word.replace("in'", "ing")
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                # Third, remove all non-letters
                print('remove punct')
                new_word = non_alphas.sub('', word)
                if new_word == '':
                    # If the word is now blank, replace with null embedding
                    print('blank')
                    embedding_vector = np.zeros(embed_size)
                else:
                    try:
                        embedding_vector = embeddings_index.get(new_word)
                        if embedding_vector is None:
                            # Otherwise, try Porter stemming
                            print('stem')
                            stemmed_word = stemmer.stem(new_word)
                            embedding_vector = embeddings_index.get(stemmed_word)
                            if embedding_vector is None:
                                # Fifth, impute from local FastText
                                print('local impute: ' + word + '(' + new_word + ')')
                                if local_embeddings_index.get(new_word) is not None:
                                    lookup_matrix = normalize([local_embeddings_index.get(new_word)])
                                    similarity = np.matmul(lookup_matrix, reference_matrix)
                                    similar_word = shared_words[np.argmax(similarity)]
                                    embedding_vector = embeddings_index.get(similar_word)
                                    print(word + ' -> ' + similar_word)
                                else:
                                    # Sixth, try to correct a contraction
                                    if "'s" in word:
                                        new_word = word.replace("'s", "")
                                        print('local impute: ' + word + '(' + new_word + ')')
                                        if local_embeddings_index.get(new_word) is not None:
                                            lookup_matrix = normalize([local_embeddings_index.get(new_word)])
                                            similarity = np.matmul(lookup_matrix, reference_matrix)
                                            similar_word = shared_words[np.argmax(similarity)]
                                            embedding_vector = embeddings_index.get(similar_word)
                                            print(word + ' -> ' + similar_word)
                                    if embedding_vector is None:
                                        print('normalize text')
                                        new_words = normalize_text(word).split()
                                        if len(new_words) == 2 and embeddings_index.get(new_words[0]) is not None and embeddings_index.get(new_words[1]) is not None:
                                            embedding_vector = embeddings_index.get(new_words[0]) + embeddings_index.get(new_words[1]) / 2
                                            print(word + ' -> ' + ' '.join(new_words))
                                        else:
                                            print('spell correct')
                                            # Seventh, try to spell correct
                                            try:
                                                new_word = str(TextBlob(word).correct())
                                            except:
                                                new_word = word
                                            embedding_vector = embeddings_index.get(new_word)
                                            if embedding_vector is not None:
                                                print(word + ' -> ' + str(new_word))
                                            else:
                                                # Eighth, give up
                                                print('Giving up on ' + str(word))
                                                import pdb
                                                pdb.set_trace()
                                                embedding_vector = np.zeros(embed_size)
                    except Exception as e:
                        print('error')
                        import pdb
                        pdb.set_trace()
        
        embedding_matrix[i] = embedding_vector

    return x_train, x_test, embedding_matrix


def run_nn_model(label, model, max_features, maxlen, epochs, batch_size, predict_batch_size):
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for embedding_name, embedding_file in EMBEDDING_FILES.items():
        if is_in_cache(label + '_' + embedding_name):
            print_step('Already trained ' + label + '_' + embedding_name + '! Skipping...')
        else:
            train_df, test_df = get_data()

            print_step('Loading embed ' + embedding_name + '...')
            embed_size = EMBED_SIZE_LOOKUP[embedding_name]
            x_train, x_test, embedding_matrix = tokenize_and_embed(train_df, test_df, embedding_file, max_features, maxlen, embed_size, embedding_name)
            y_train = train_df[classes].values

            print_step('Build model...')
            model = model(max_features, maxlen, embed_size, embedding_matrix)
            model.save_weights('cache/gru-model-weights.h5')

            print_step('Making KFold for CV')
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

            i = 1
            cv_scores = []
            pred_train = np.zeros((train_df.shape[0], 6))
            pred_full_test = np.zeros((test_df.shape[0], 6))
            for dev_index, val_index in kf.split(x_train, y_train[:, 0]):
                print_step('Started fold ' + str(i))
                model.load_weights('cache/' + label + '_' + embedding_name + '-model-weights.h5')
                dev_X, val_X = x_train[dev_index], x_train[val_index]
                dev_y, val_y = y_train[dev_index, :], y_train[val_index, :]
                RocAuc = RocAucEvaluation(validation_data=(val_X, val_y), interval=1)
                model.fit(dev_X, dev_y, batch_size=batch_size, epochs=epochs,
                          validation_data=(val_X, val_y), callbacks=[RocAuc])
                val_pred = model.predict(val_X, batch_size=predict_batch_size, verbose=1)
                pred_train[val_index, :] = val_pred
                test_pred = model.predict(x_test, batch_size=predict_batch_size, verbose=1)
                pred_full_test = pred_full_test + test_pred
                cv_score = [roc_auc_score(val_y[:, j], val_pred[:, j]) for j in range(6)]
                print_step('Fold ' + str(i) + ' done')
                pprint(zip(classes, cv_score))
                cv_scores.append(cv_score)
                i += 1
            print_step('All folds done!')
            print('CV scores')
            pprint(zip(classes, np.mean(cv_scores, axis=0)))
            mean_cv_score = np.mean(np.mean(cv_scores, axis=0))
            print('mean cv score : ' + str(mean_cv_score))
            pred_full_test = pred_full_test / 5.
            for k, classx in enumerate(classes):
                train_df['gru_' + classx] = pred_train[:, k]
                test_df['gru_' + classx] = pred_full_test[:, k]

            print('~~~~~~~~~~~~~~~~~~')
            print_step('Cache Level 1')
            save_in_cache('lvl1_' + label + '_' + embedding_name, train_df, test_df)
            print_step('Done!')

            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print_step('Prepping submission file')
            submission = pd.DataFrame()
            submission['id'] = test_df['id']
            submission['toxic'] = test_df[label + '_' + embedding_name + '_toxic']
            submission['severe_toxic'] = test_df[label + '_' + embedding_name + '_severe_toxic']
            submission['obscene'] = test_df[label + '_' + embedding_name + '_obscene']
            submission['threat'] = test_df[label + '_' + embedding_name + '_threat']
            submission['insult'] = test_df[label + '_' + embedding_name + '_insult']
            submission['identity_hate'] = test_df[label + '_' + embedding_name + '_identity_hate']
            submission.to_csv('submit/submit_lvl1_' + label + '_' + embedding_name + '.csv', index=False)
            print_step('Done')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]




class GetBest(Callback):
    """Get the best model at the end of training.
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    # Example
        callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
        mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)

class AMSgrad(Optimizer):
  """AMSGrad optimizer.
  Default parameters follow those provided in the Adam paper.
  # Arguments
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor.
      decay: float >= 0. Learning rate decay over each update.
  # References
      - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
      - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
  """
 
  def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., **kwargs):
    super(AMSgrad, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
      self.decay = K.variable(decay, name='decay')
    self.epsilon = epsilon
    self.initial_decay = decay

  
  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

    t = K.cast(self.iterations, K.floatx()) + 1
    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /(1. - K.pow(self.beta_1, t)))

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params] 
    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      vhat_t = K.maximum(vhat, v_t)
      p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))
      self.updates.append(K.update(vhat, vhat_t))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(K.update(p, new_p))
    return self.updates

  def get_config(self):
      config = {'lr': float(K.get_value(self.lr)),
                'beta_1': float(K.get_value(self.beta_1)),
                'beta_2': float(K.get_value(self.beta_2)),
                'decay': float(K.get_value(self.decay)),
                'epsilon': self.epsilon}
      base_config = super(AMSgrad, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))


def tokenize(text_train, text_test, num_words=200000, maxlen=100):
  """
  Tokenize training and test set text

  Args:
      text_train: Training set text
      text_test: Testing set text
      num_words: The maximum number of words to keep, based on word
        frequency. Only the most common `num_words` words will be kept.
      maxlen: Maximum length of sequence. Shorter sequences will be
        pre-padded with zeros
  Returns:
      A tuple of tokenized text
  """
  from numpy import concatenate 
  from keras.preprocessing.text import Tokenizer
  from keras.preprocessing.sequence import pad_sequences

  tokenizer = Tokenizer(num_words=num_words)
  tokenizer.fit_on_texts(concatenate([text_train, text_test]))

  tokenized_train = tokenizer.texts_to_sequences(text_train)
  tokenized_test = tokenizer.texts_to_sequences(text_test)

  X_tr = pad_sequences(tokenized_train, maxlen=maxlen)
  X_te = pad_sequences(tokenized_test, maxlen=maxlen)  
  return  X_tr, X_te

"""
from: https://www.kaggle.com/sermakarevich/hierarchical-attention-network
"""
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):

    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, axis=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

        assert axis in [1,2],  'expected dimensions (samples, filters, convolved_values),\
                   cannot fold along samples dimension or axis not in list [1,2]'
        self.axis = axis

        # need to switch the axis with the last elemnet
        # to perform transpose for tok k elements since top_k works in last axis
        self.transpose_perm = [0,1,2] #default
        self.transpose_perm[self.axis] = 2
        self.transpose_perm[2] = self.axis

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[self.axis] = self.k
        return tuple(input_shape_list)

    def call(self, x):
        # swap sequence dimension to get top k elements along axis=1
        transposed_for_topk = tf.transpose(x, perm=self.transpose_perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(transposed_for_topk, k=self.k, sorted=True, name=None)[0]

        # return back to normal dimension but now sequence dimension has only k elements
        # performing another transpose will get the tensor back to its original shape
        # but will have k as its axis_1 size
        transposed_back = tf.transpose(top_k, perm=self.transpose_perm)

        return transposed_back
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

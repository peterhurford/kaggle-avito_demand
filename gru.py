# Following https://www.kaggle.com/yekenot/pooled-gru-fasttext/code
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras_utils import run_nn_model


max_features = 200000
maxlen = 300
epochs = 2
batch_size = 32
predict_batch_size = 1024

def model(max_features, maxlen, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

np.random.seed(42)
run_nn_model('gru', model, max_features, maxlen, epochs, batch_size, predict_batch_size)

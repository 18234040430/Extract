#-*- coding:UTF-8 -*-
'''
Created on 2019年1月17日

@author: wyf
'''
import Tools.Load as load
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras_contrib.layers import CRF
from keras.layers.wrappers import Bidirectional
EMBEDDING_DIM = 100
BiRNN_UNITS = 200
def create_model():
    x_train,x_test,y_train,y_test,embedding,wordtoid,typetoid = load.prepareData()
    
    model = Sequential()
    
    model.add(Embedding(len(wordtoid),EMBEDDING_DIM,mask_zero=True))
    model.add(Bidirectional(LSTM(BiRNN_UNITS//2,return_sequences = True)))
    crf = CRF(len(typetoid),sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(x_train, y_train,batch_size=16,epochs=10, validation_data=[x_test, y_test])
if __name__ == '__main__':
    create_model()
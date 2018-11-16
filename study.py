# 0.  사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
import json


class Exam:
    def train(self, model):
        # 1.  데이터셋 생성하기
        # 훈련셋과 시험셋 불러오기
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 데이터셋 전처리
        x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
        x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

        # 원핫인코딩 (one-hot encoding) 처리
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        # 훈련셋과 검증셋 분리
        x_val = x_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
        x_train = x_train[42000:]
        y_val = y_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
        y_train = y_train[42000:]

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))


    def save_model(self, model, filename):
        #json 저장
        json_string = model.to_json()
        with open(filename + '.json', 'w') as f:
            f.write(json_string)


    def load_model(self, filename):
        if filename:
            # json 불러오기
            with open(filename + '.json', 'r') as f:
                json_string = f.read()

            model = model_from_json(json_string)
        else:
            # 2.  모델 구성하기
            model = Sequential()
            model.add(Dense(units=64, input_dim=28 * 28, activation='relu'))
            model.add(Dense(units=10, activation='softmax'))

        # 3.  모델 학습과정 설정하기
        return model


    def test(self, model):
        # 1.  실무에 사용할 데이터 준비하기
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
        y_test = np_utils.to_categorical(y_test)
        xhat_idx = np.random.choice(x_test.shape[0], 5)
        xhat = x_test[xhat_idx]

        # 3.  모델 사용하기
        yhat = model.predict_classes(xhat)

        for i in range(5):
            print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))

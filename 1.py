import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "kddcup.data_10_percent_corrected - 복사본.csv"

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score #정밀도, 재현률

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#dataset = pd.read_csv('/content/drive/My Drive/Colab Notebooks/kddcup.data_10_percent_corrected')
dataset = pd.read_csv('/content/drive/My Drive/kddcup.data_10_percent_corrected.csv')  # 데이터를 읽어서 dataset에 넣어줘
# 처음 다섯 개 데이터를 조회
dataset.head()
#  마지막 label 컬럼 안에 담긴 고유의 카테고리 값을 확인
dataset['normal.'].unique()
# x = 마지막 label 열을 제외한 모든 열을 특징으로 사용
# y = 마지막 label 열을 카테고리로 사용

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values

print(x.shape, y.shape)
uniq1 = dataset.tcp.unique()
uniq2 = dataset.http.unique()
uniq3 = dataset.SF.unique()

print(uniq1, '\n', uniq2, '\n', uniq3)
print(uniq1.size, '\n', uniq2.size, '\n', uniq3.size)
# tcp(프로토콜), http(서비스), SF(플래그) 열을 One-Hot Encoding 변환
from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

# onehotencoder_1 = OneHotEncoder(categorical_features = [1])
onehotencoder_1 = ColumnTransformer([("tcp", OneHotEncoder(), [1])],
                                    remainder = 'passthrough')
# onehotencoder_2 = OneHotEncoder(categorical_features = [4])  # + 3
onehotencoder_2 = ColumnTransformer([("http", OneHotEncoder(), [4])],
                                    remainder = 'passthrough')
# onehotencoder_3 = OneHotEncoder(categorical_features = [70]) # + 66
onehotencoder_3 = ColumnTransformer([("SF", OneHotEncoder(), [70])],
                                    remainder = 'passthrough')

x = np.array(onehotencoder_1.fit_transform(x))
x = np.array(onehotencoder_2.fit_transform(x))
x = np.array(onehotencoder_3.fit_transform(x))

labelencoder_y = LabelEncoder() # 문자를 숫자로 치환
y = labelencoder_y.fit_transform(y)

print(x.shape, y.shape)

# 최종 데이터 개수
# X = 494,020 x 118
# Y = 494,020 x 1
import keras
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

# X와 Y를 훈련 데이터(train)와 테스트 데이터(test)로 분류 = 70 : 30
x_train, x_test, y_train, y_test = train_test_split(x, y,
                   test_size = 0.3, random_state = 0) #148206 테스트용 데이터
#345814=train
#148206=test

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

seed = 0
np.random.seed(seed)
#tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)

x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = Sequential()
model.add(Dense(30, input_dim=118, activation='relu'))
# fully connected layer
model.add(Dense(1, activation='sigmoid')) #1단 딥러닝

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=50) #학습

print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))
"""
y_test = np.ndarray.reshape(y_test,len(y_test))
print(y_test.dtype)
print(y_test[100])
"""
#y_pred = model.predict_classes(x_test) #labelling 0,1
#y_pred = model.predict(x_test)
y_pred = (model.predict(x_test, verbose=1) > 0.5).astype("int32") #sigmoid인 경우

#y_pred = np.argmax(model.predict(x), axis=-1) #softmax인 경우

"""
for t in y_pred:
    if t != 1.0 and t != 0.0:
        print(t)

y_pred = np.ndarray.reshape(y_pred,len(y_test))
print(y_pred.dtype)
print(y_pred[100])
"""
# 혼돈 행렬과 정확도, F1 점수로 모델 평가
cnf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy : ", acc, "/ F1-Score : ", f1)
print(cnf_matrix)

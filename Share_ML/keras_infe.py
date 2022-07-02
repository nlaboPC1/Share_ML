'''
保存されたデータを読み込んで推論のみを行う
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib .pyplot as plt

# モデルデータの読み込み
model = keras.models.load_model("my_keras_model1.h5")
'''
keras_train1で生成したモデルデータを読み込む
下の検証はkeras_train1と同様
'''

fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 検証データの作成(入力xはピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# クラス名（出力の分類）を定義
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# テストデータで検証を行う
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

# モデルを使った予測
x_new = x_test[:3]
y_pred = model.predict(x_new)
print(y_pred.round(2))

# クラス分類
y_pred = model.predict(x_new) 
classes_y = np.argmax(y_pred,axis=1)
print(classes_y)
print (np.array(class_names)[classes_y])
y_new = y_test[:3]
print(y_new)


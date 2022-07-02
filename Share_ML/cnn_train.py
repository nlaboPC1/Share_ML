'''
CNNによる画像訓練
'''

from pickletools import optimize
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib .pyplot as plt

# fashion_mnistデータの読み込み
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# データ形状の確認
print(x_train_full.shape)
print(x_train_full.dtype)

# 検証データの作成(入力はピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 学習層の生成
model = keras.models.Sequential([
    # 畳み込み層：7×7のフィルタを64個　ストライドなし
    keras.layers.Conv2D(64, 7, activation ="relu", padding="same",
                        input_shape=[28, 28, 1]),
    # プールサイズ2の最大値プーリング　空間次元がそれぞれ1/2になる
    keras.layers.MaxPooling2D(2),
    #---------------------------------------------------------------
    # 畳み込み層
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    # プーリング層
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    #----------------------------------------------------------------
    # 全結合層に入れるために１次元化
    keras.layers.Flatten(),
    # 全結合層
    keras.layers.Dense(128, activation="relu"),
    # ドロップアウト
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    # 出力層
    keras.layers.Dense(10,activation="softmax")
])
'''
ここではAlexNetに近い構造を紹介
畳み込み層２＋プーリング層１でワンセット　
画像サイズに合わせてセット数を決める
'''

# 層の名前，タイプ，出力の形，パラメータ数が表として表示される
model.summary()

# モデルのコンパイル
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = keras.optimizers.SGD(learning_rate=0.01),
              metrics = ["accuracy"])

# モデルの訓練と評価
history = model.fit(x_train, y_train, epochs = 30,
                    validation_data = (x_valid, y_valid))

#表示
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

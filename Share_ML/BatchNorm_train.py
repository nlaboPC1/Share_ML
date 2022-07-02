'''
BatchNormalizationの実装

学習テクニックに関する変更を行っている
ハイパーパラメータを適当な値としているため
性能の向上は確認できない
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib .pyplot as plt

'''
前準備
'''
# fashion_mnistデータの読み込み
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 検証データの作成(入力xはピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

'''
学習層の生成
'''
# シーケンシャルAPI
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    # KerasのシーケンシャルAPIでは下の文を追加するだけでバッチ正規化を利用できる
    keras.layers.BatchNormalization(),
    # 活性化関数:ELU(ReLUの変種)    初期値:Heの初期値  としている
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

'''
コンパイル
'''

# モデルのコンパイル
model.compile(loss = "sparse_categorical_crossentropy",
              # 最適化にAdamを使用している
              optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
              metrics = ["accuracy"])

'''
訓練
'''
# モデルの訓練と評価
history = model.fit(x_train, y_train, epochs = 30,
                    validation_data = (x_valid, y_valid))

#表示
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.summary()
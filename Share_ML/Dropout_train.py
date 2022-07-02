'''
Dropoutの実装

学習テクニックに関する変更を行っている
ハイパーパラメータを適当な値としているため
性能の向上は確認できないが，
結果を見ると過学習の発生を抑え込んでいることがわかる
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

pixel_means = x_train.mean(axis=0, keepdims=True)
pixel_stds = x_train.std(axis=0, keepdims=True)
x_test_scaled = (x_test - pixel_means) / pixel_stds


'''
学習層の生成
'''
# シーケンシャルAPI
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    # KerasのシーケンシャルAPIでは下の文を追加するだけでDropoutを利用できる
    keras.layers.Dropout(rate=0.2),
    # rateは通常10%~50%の間で設定される
    # 畳み込みニューラルネットワークでは40~50%にされることが多い
    # 過学習が発生する場合はrateを上げ，小規模な層ではrateを下げる
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])

'''
コンパイル
'''

# モデルのコンパイル
model.compile(loss = "sparse_categorical_crossentropy",
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

# モデルを使った予測
y_probas = np.stack([model(x_test_scaled, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
print(np.round(model.predict(x_test_scaled[:1]),2))
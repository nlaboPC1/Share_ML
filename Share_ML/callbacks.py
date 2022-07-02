'''
コールバックと早期打ち切りについての解説
'''

'''
コールバックと早期打ち切りを併用すれば，
クラッシュとリソース（訓練時間）の浪費対策が実現できる
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib .pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 検証データの作成(入力xはピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# モデルデータの読み込み
model = keras.models.load_model("my_keras_model1.h5")
#model = keras.models.load_model("my_keras_model2.h5")
'''
内容を上書きしないためにファイルを分けているが，実際は保存ファイルを分ける必要はない
2回目以降はmy_keras_model2.h5を使う
'''


'''
コールバックの使い方
fitメソッドはcallbacks引数を持っており，訓練の開始，終了時，
各エポックの開始，終了時，バッチを1つ処理する前後にKerasが呼び出すオブジェクトのリストを
指定できるようになっている

'''
# モデルの構築
#checkpoint_cb = keras.callbacks.NodelCheckpoint("my_keras_model2.h5")
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model2.h5",
                                                save_best_only = True)
'''
save_best_only = Trueを書き足すと検証セットに対する性能が最大になった時のデータを保存する
'''
# コンパイル
#history = model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint_cb])
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_valid, y_valid),
                    callbacks=[checkpoint_cb])
# 最良の結果へのロールバック
model = keras.models.load_model("my_keras_model2.h5")


'''
早期打ち切り
EarlyStoppingコールバックでの実装
指定されたエポック数だけ性能が上がらないときに
中止し,最良の結果をロールバックする
自動的に打ち切れるのでエポックを大きく設定してもよい
'''

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                 restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100,
                    validation_data=(x_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
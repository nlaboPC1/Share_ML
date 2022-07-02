'''
fashion_mnist（tensorflowからアクセスできるデータ集合）を用いた学習練習

レイヤの作成とデータの可視化について扱う
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

# データ形状の確認
print(x_train_full.shape) # 結果：(60000, 28, 28)
print(x_test.shape) # 結果：(10000, 28, 28)
print(y_train_full.shape) # 結果：(60000,)
print(y_test.shape) # 結果：(10000,)
print(x_train_full.dtype) # 結果：uint8

# 検証データの作成(入力xはピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# クラス名（出力の分類）を定義
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

'''
学習層の生成
'''
# シーケンシャルAPI（層を順に重ねる単純なネットワークモデル）
model = keras.models.Sequential([
    # 入力を1次元配列に変換 
    # 最初の層なのでinput_shape（入力画像の解像度）を指定しなければならない
    keras.layers.Flatten(input_shape=[28, 28]),
    # 300個のニューロンを持つDense隠れ層（全結合層）を生成（活性化関数はrelu）
    keras.layers.Dense(300, activation="relu"),
    # 100個のニューロンを持つDense隠れ層（全結合層）を生成（活性化関数はrelu）
    keras.layers.Dense(100, activation="relu"),
    # 10個（分類クラス数）のニューロンを持つDense出力層（全結合層）を生成（活性化関数はsoftmax）
    keras.layers.Dense(10,activation="softmax")
])
'''
# 別の書き方(学習層)
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))
'''


'''
学習層内部の可視化
'''

# 層の名前，タイプ，出力の形，パラメータ数が表として表示される
model.summary()
'''
層の名前は作成時に指定しなければ自動生成される
出力のNoneはバッチサイズがいくつでも構わないことを示す
表の下部にはパラメータ数の合計と訓練できるパラメータ，
訓練できないパラメータの内数が表示される
'''

# モデルの層のリストの取得
print(model.layers)
'''
出力結果
[<keras.layers.reshaping.flatten.Flatten object at 0x000002153DA15EA0>, <keras.layers.core.dense.Dense object at 0x0000021538AA4400>, 
<keras.layers.core.dense.Dense object at 0x0000021538AA46A0>, <keras.layers.core.dense.Dense object at 0x0000021538AA4850>]
'''

# 特定の層のみ取得
hidden1 = model.layers[1]
print(hidden1.name)

# パラメータの読み書き
# get_weights()メソッドでパラメータの読み取り
weights, biases = hidden1.get_weights()
# 重みは無作為に初期化
print(weights)
print(weights.shape)
# バイアス項を0で初期化
print(biases)
print(biases.shape)
'''
set_weight()メソッドでパラメータの書き込みが可能
'''


'''
コンパイル
'''
# モデルのコンパイル
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = keras.optimizers.SGD(learning_rate=0.01),
              metrics = ["accuracy"])
'''
損失関数とオプティマイザを指定
optimizer = "sgd"でも良いが学習率がディフォルトで0.01に固定される

リストは公式ページ参照
https://keras.io/api/losses/
https://keras.io/api/optimizers/
https://keras.io/api/metrics/
'''

'''
訓練
'''
# モデルの訓練と評価
history = model.fit(x_train, y_train, epochs = 30,
                    validation_data = (x_valid, y_valid))
'''
入力特徴量：x_train
ターゲットクラス：y_train
訓練のエポック数：epochs
検証データ：validation_data
'''
#表示
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


'''
パラメータモデルの保存
'''
# モデルの保存
model.save("my_keras_model1.h5")
'''
HDF5形式でモデルのアーキテクチャ，オプティマイザ，
モデルパラメータの値を保存する
サブクラス化モデルでは使えない
サブクラス化モデルはより高度な柔軟性を持たせるための応用分野であるため解説は省略
'''

'''
推論
'''
# テストデータで検証を行う
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)
'''
6月27日時点で損失関数の値が間違って出力されている
原因は今後特定してアップデートを行う
'''
# モデルを使った予測
x_new = x_test[:3]
y_pred = model.predict(x_new)
print(y_pred.round(2))
'''
この処理は実際のデータに適応させることに相当する
実際の新インスタンスはこの実験ではないので
テストデータから3つ取り出して疑似的な実験データとする

predictメソッドで予測を行う

こちらも出力が間違っている
実際には0~1の範囲でいくつかの値をとる
'''

# クラス分類
y_pred = model.predict(x_new) 
classes_y = np.argmax(y_pred,axis=1)
print(classes_y)
print (np.array(class_names)[classes_y])
y_new = y_test[:3]
print(y_new)
'''
実際の推論ではどのクラスに分類されたかだけを見ればよいので
上記のプログラムで分類されたクラスを出力する
この例ではクラスの要素番号[9 2 1]が等しいため正しく分類されたことがわかる
'''


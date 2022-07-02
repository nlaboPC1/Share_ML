
'''
ハイパーパラメータ最適化を行うプログラム
'''

'''
動作させるとわかるがこの程度の規模のランダム探索でも非常に時間がかかる
このほかに下記のライブラリを利用する探索法もある
解説はreadme等を参照
------------------------------------------------
Hyperopt:あらゆる探索空間に対し広く使えるライブラリ
https://github.com/hyperopt/hyperopt

Hyperas:Kerasモデルの最適化に便利なライブラリ
https://github.com/maxpumperla/hyperas

Spearmint:ベイズ最適化ライブラリ
https://github.com/JasperSnoek/spearmint
------------------------------------------------

'''
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
# scikit-learnについては解説省略
import numpy as np
import tensorflow as tf
from tensorflow import keras

# データ入力
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 検証データの作成(入力xはピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


#  モデルのビルド（やっていることはkeras_trainの学習層生成とほぼ同じ） 
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28, 28]):
# n_hidden:隠れ層の数, n_neurons:ニューロン数
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    # 隠れ層の数に設定した値だけ隠れ層を作成
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(10,activation="softmax"))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss = "sparse_categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    return model

# KerasRegressorオブジェクトの定義
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
'''
薄いラップをかぶせてオブジェクト化することでハイパーパラメータを指定できる
指定せずに使用する場合，ビルド関数で設定したディフォルト値を使用する
'''

# ハイパーパラメータの範囲設定
param_distribs = {
    "n_hidden": list(range(0, 3, 1)),
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}
# 設定範囲を使用してランダム探索を行う
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(x_train, y_train, epochs=100,
                  validation_data=(x_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
'''
RandomizedSearchCVはK分割交差検証(下に解説)を使用しているため
検証データ(valid_data)は早期打ち切りにしか使用していないことに注意

n_iter:サンプリングされるパラメータ設定の数　大きな値をとると品質は増加するが実行時間も増加する
cv: K分割交差検証のKの値
'''
# 最終結果表示
print("best_params:",rnd_search_cv.best_params_)
print("best_score:",rnd_search_cv.best_score_)


'''
K分割交差検証法
元のデータをK個にランダムに分割する
K個のデータは順番に検証セットとして使われ，使われていない残りデータが学習セットとなる
これによりK個の検証結果が得られ，その中から統計量が最良のものを最終的なモデルとする
この方法は小規模データセットに適している
'''
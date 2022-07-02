'''
tensorboardによる学習曲線の可視化
'''

'''
優れた可視化ツールがtensorflowに内蔵されている
プログラム実行後コマンドで
tensorboard --logdir=./my_logs --port=6006
と入力するとページアドレスが表示されるのでWebページで結果を閲覧できる

'''
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorboard

# データ入力
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 検証データの作成(入力xはピクセル数で割って正規化)
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# モデルデータの読み込み
model = keras.models.load_model("ml\my_keras_model1.h5")

# my_logフォルダへのアクセス（作成）
root_logdir = os.path.join(os.curdir, "my_logs")

# my_logフォルダに実行時間のイベントファイル名を提供
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir =get_run_logdir()

# ログディレクトリの作成
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
# 推論
history = model.fit(x_train, y_train, epochs=30,
                    validation_data = (x_valid, y_valid),
                    callbacks = [tensorboard_cb])
'''
ローカルディレクトリからの画像データの取り込み

サイズ変更後の画像を見ればわかるが
アスペクト比を崩してリサイズするので
入力前にアスペクト比はある程度調整する必要がある
'''

import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# 相対パスの設定
imgdir_path = pathlib.Path('ml\cat_dog.images')
# フォルダ内のjpg画像をリスト化
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

# 画像の可視化
fig = plt.figure(figsize=(10,5))
for i, file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape: ',img.shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
    
plt.tight_layout()
plt.show()

# ファイル名に"dog"があるとき１，それ以外を０にラベリングする
labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)

# データセットの作成，結合
ds_files_labels = tf.data.Dataset.from_tensor_slices((file_list, labels))
for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy())
    
# 読み込み，デコード，サイズ変更を行う
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0
    return image, label

# 目的のサイズ
img_width, img_height = 120, 80
# load_and_preporcessの結果をリスト化
ds_images_labels = ds_files_labels.map(load_and_preprocess)

# 表示
fig = plt.figure(figsize= (10, 5))
for i, example in enumerate(ds_images_labels):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0]) 
    ax.set_title('{}'.format(example[1].numpy()), size=15)
    
plt.tight_layout()
plt.show()
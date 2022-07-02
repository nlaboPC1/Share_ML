'''
画像変換とデータ拡張

学習データが足りないときや表現力を落とさずに過学習を抑えたいときに
データを拡張することで訓練データの水増しを行うことができる
※あくまで水増しなので頼りすぎない
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# 画像データのダウンロード（サイズが大きいので注意）
celeba_bldr =tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)
celeba_train = celeba['train']
celeba_valid = celeba['validation']
celeba_test = celeba['test']

# example配列に画像データを５つ入れる
examples = []
for example in celeba_train.take(5):
    examples.append(example['image'])

# 全体のfigサイズ設定
fig = plt.figure(figsize=(16, 8.5))

'''
列１：画像を矩形に切り抜く
'''
# 元データのプロット位置
ax = fig.add_subplot(2, 5, 1)
# グラフタイトル
ax.set_title('Crop to a \nbounding-box', size=15)
# 元データの表示
ax.imshow(examples[0])
# 加工データのプロット位置
ax = fig.add_subplot(2, 5, 6)
# tf.imageを利用した画像の切り出し
img_cropped=tf.image.crop_to_bounding_box(examples[0], 50, 20, 128, 128)
'''
50,20 : 矩形の左上隅の座標（どこを切り抜くかの指示）左上端（0, 0）
128, 128 : 切り抜きサイズ
'''
# 加工データの表示
ax.imshow(img_cropped)

'''
列２：水平方向に反転させる
'''
ax = fig.add_subplot(2, 5, 2)
ax.set_title('Frip(forizontal)', size=15)
ax.imshow(examples[1])
ax = fig.add_subplot(2, 5, 7)
img_flipped=tf.image.flip_left_right(examples[1])
ax.imshow(img_flipped)

'''
列３：コントラストを強調する
'''
ax = fig.add_subplot(2, 5, 3)
ax.set_title('Adjust contrast', size=15)
ax.imshow(examples[2])
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast=tf.image.adjust_contrast(examples[2],contrast_factor=2)
ax.imshow(img_adj_contrast)

'''
列４：明度を調整する
'''
ax = fig.add_subplot(2, 5, 4)
ax.set_title('Adjust brightness', size=15)
ax.imshow(examples[3])
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness=tf.image.adjust_brightness(examples[3],delta=0.3)
ax.imshow(img_adj_brightness)


'''
列５：画像を中心でくり抜き、元のサイズ(218×178)に戻す
'''
ax = fig.add_subplot(2, 5, 5)
ax.set_title('Central crop\nand resize', size=15)
ax.imshow(examples[4])
ax = fig.add_subplot(2, 5, 10)
img_center_crop = tf.image.central_crop(examples[4], 0.7)
img_resized = tf.image.resize(img_center_crop, size=(218, 178))
ax.imshow(img_resized.numpy().astype('uint8'))

plt.show()


#---------------------------------------------------------------
'''
ランダム要素の追加とパイプラインの作成
'''
# 乱数シードを固定して再現性を持たせる
tf.random.set_seed(1)
fig = plt.figure(figsize=(14, 12))
# ３つのサンプル画像で実行
for i,example in enumerate(celeba_train.take(3)):
    image = example['image']
    ax = fig.add_subplot(3, 4, i*4+1)
    ax.imshow(image)
    # 最初のみタイトルをつける
    if i==0:
        ax.set_title('Orig', size=15)
    # プロット位置指定
    ax = fig.add_subplot(3, 4, i*4+2)
    # 矩形に切り抜く（左上隅座標をランダム）
    img_crop = tf.image.random_crop(image, size=(178, 178, 3))
    ax.imshow(img_crop)
    if i==0:
        ax.set_title('Step 1: Random crop', size=15)
    
    # 反転（水平または垂直軸に沿って0.5の確率でランダム）
    ax = fig.add_subplot(3, 4, i*4+3)
    img_flip = tf.image.random_flip_left_right(img_crop)
    ax.imshow(tf.cast(img_flip, tf.uint8))
    if i == 0:
        ax.set_title('Step 2: Random flip', size=15)
    
    # 画像コントラストをランダムに変化
    ax = fig.add_subplot(3, 4, i*4+4)
    img_resize = tf.image.resize(img_flip, size=(128, 128))
    ax.imshow(tf.cast(img_resize, tf.uint8))
    if i == 0:
        ax.set_title('Step 3: Resize', size=15) 

plt.show()

#---------------------------------------------------------------------
'''
データ拡張パイプラインを使うラッパー関数を定義

'image'と'attributes'２つのキーを含んだディクショナリを受け取り
戻り値として返還後の画像と属性のディクショナリから抽出したラベルを含むタプルを返す
'''
def preprocess(example, size=(64,64), mode='train'):
    image = example['image']
    label = example['attributes']['Male']
    if mode == 'train':
        image_cropped = tf.image.random_crop(image, size=(178, 178, 3))
        image_resized = tf.image.resize(image_cropped, size=size)
        image_flip = tf.image.random_flip_left_right(image_resized)
        return (image_flip/255.0, tf.cast(label, tf.int32))
    else: # 非訓練データではランダムを使用しない
        image_cropped = tf.image.crop_to_bounding_box(
            image, offset_height=20, offset_width=0,
            target_height=178, target_width=178)
        image_resized = tf.image.resize(image_cropped, size=size)
        return (image_resized/255.0, tf.cast(label, tf.int32))
        
'''
サブセットを作成しデータ拡張を確かめる
'''
tf.random.set_seed(1)
# データをシャッフル
ds = celeba_train.shuffle(1000, reshuffle_each_iteration=False)
'''
reshuffle_each_iteration=Falsとすれば毎回同じ順番で取り出される
'''
# ２種類の画像をそれぞれ５枚ずつ用意する
ds = ds.take(2).repeat(5)
# preprocess関数の利用
ds = ds.map(lambda x:preprocess(x, size=(178, 178), mode='train'))
fig = plt.figure(figsize=(15, 6))
# 表示
for j,example in enumerate(ds):
    ax = fig.add_subplot(2, 5, j//2+(j%2)*5+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])

plt.show()

'''
preprocess関数を訓練データと検証データに適用
'''
BATCH_SIZE = 32
BUFFER_SIZE = 1000
IMAGE_SIZE = (64, 64)
step_per_epoch = np.ceil(16000/BATCH_SIZE)
'''
np.ceil：小数点以下切り上げ
'''
ds_train = celeba_train.map(lambda x: preprocess(x, size=IMAGE_SIZE, mode='train'))
ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat()
ds_train = ds_train.batch(BATCH_SIZE)
ds_valid = celeba_valid.map(lambda x: preprocess(x, size=IMAGE_SIZE, mode='eval'))
'''
mode='eval'を指定することでデータ拡張パイプラインのランダムな要素が訓練データにのみ適用される
'''
ds_valid = ds_valid.batch(BATCH_SIZE)

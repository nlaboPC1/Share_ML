'''
tf.dataを使った入力パイプラインの構築
'''
import tensorflow as tf

'''
データセットを既存のリストから作成
'''
# データセットとしてリストを作成
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
# 配列をスライスしてデータセットを作成
ds = tf.data.Dataset.from_tensor_slices(a)
print("データセット\n-----------------------------------")
print(ds)
print("\nデータセットの要素\n-----------------------------------")
# 要素ごとに処理
for item in ds:
      print(item)
# サイズ３のバッチを作成
ds_batch = ds.batch(3)
print("\nバッチ作成\n-----------------------------------")
for i, elem in enumerate(ds_batch, 1):
      print('batch {}:'.format(i), elem.numpy())

      
'''
データの結合

特徴量データ，ラベル，アノテーションデータなど入力するデータは
複数種類で存在する場合があり，これを１つのデータセットに結合すると
テンソルの要素をタプル（対応した組み合わせ）として取り出せるようになる
'''

tf.random.set_seed(1)
# 特徴量データ
t_x =tf.random.uniform([4, 3], dtype=tf.float32)
# ラベルデータ
t_y = tf.range(4)

# ２つのデータセットを別々に作成
ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
# zip関数でデータセットを結合
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
print("\nデータの結合１\n-----------------------------------")
for example in ds_joint:
      print(' x:', example[0].numpy(), 'y:', example[1].numpy())

# ２つのデータに対しデータセット作成と結合を１度に行う(結果は上と同じ)
print("\nデータの結合２\n-----------------------------------")
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))
for example in ds_joint:
      print(' x:', example[0].numpy(), 'y:', example[1].numpy())

# データセットの各要素に変更を適用
ds_trans = ds_joint.map(lambda x, y: (x*2-1.0, y))
'''
データセットの値の範囲を[-1,1]にするために特徴量スケーリングを適用
'''
print("\n各要素に変換を適用\n-----------------------------------")
for example in ds_trans:
      print(' x:', example[0].numpy(), ' y:', example[1].numpy())


'''
シャッフル，バッチ，リピート
'''
# 乱数シードを固定して再現性を持たせる
tf.random.set_seed(1)
# シャッフル
ds = ds_joint.shuffle(buffer_size=len(t_x))
'''
buffer_size：シャッフル前にサンプリングされる要素の数
buffer_sizeの値が小さすぎると完全にシャッフルされないことがある
'''
print("\nシャッフル\n-----------------------------------")
for example in ds:
      print(' x:', example[0].numpy(), ' y:', example[1].numpy())


# バッチ分割
print("\nバッチ分割\n-----------------------------------")
ds = ds_joint.batch(batch_size=3, drop_remainder=False)
# 要素を１つずつ繰り返し参照
batch_x, batch_y = next(iter(ds))
print('Batch-x:\n', batch_x.numpy())
print('Batch-y:', batch_y.numpy())

# 目的のエポック回数に基づいてシャッフル／リピートを行う
'''
リピート：コピーの生成
'''
print("\nシャッフル／リピート\n-----------------------------------")
ds = ds_joint.batch(3).repeat(count=2)
for i,(batch_x, batch_y)in enumerate(ds):
      print(i, batch_x.shape, batch_y.numpy())

# リピート／シャッフルの順で行う
print("\nリピート／シャッフル\n-----------------------------------")
ds = ds_joint.repeat(count=2).batch(3)
for i,(batch_x, batch_y)in enumerate(ds):
      print(i, batch_x.shape, batch_y.numpy())

# 組み合わせ
print("\nシャッフル／バッチ／リピート\n-----------------------------------")
tf.random.set_seed(1)
ds = ds_joint.shuffle(4).batch(2).repeat(3)
for i,(batch_x, batch_y) in enumerate(ds):
      print(i, batch_x.shape, batch_y.numpy())

print("\nバッチ／シャッフル／リピート\n-----------------------------------")      
tf.random.set_seed(1)
ds = ds_joint.batch(2).shuffle(4).repeat(3)
for i,(batch_x, batch_y) in enumerate(ds):
      print(i, batch_x.shape, batch_y.numpy())

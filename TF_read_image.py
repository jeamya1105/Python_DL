import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import skimage.io as io
import os

class IMG:
    tfrecords_filename = 'testimg.tfrecords'
    def __init__(self, path):
        add_path = "testimg\\"
        files = os.listdir(path)
        self.img_list = []
        self.label_list = []
        i = 0
        for file in files:
            self.img_list.append(add_path + file)
            self.label_list.append(i)
            i+=1

class compress:
    def compression(self, tfrecords_filename):
        # 建立 TFRecordWriter
        writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=tf.python_io.TFRecordOptions(self.compression))
        return writer
    def record_iterator(self, tfrecords_filename):
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename, options=tf.python_io.TFRecordOptions(self.compression))
        return record_iterator
    def TF_reader(self, tfrecords_filename):
        reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(self.compression))
        return reader
    def __init__(self, tfrecords_filename):
        # 設定以 gzip 壓縮
        compression = tf.python_io.TFRecordCompressionType.GZIP

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_img(tfrecords_filename, imglist, labellist):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for image_filename, label in zip(imglist, labellist):
      image = io.imread(image_filename)
      #io.imshow(image)
      height, width, depth = image.shape
      image_string = image.tostring()
      # 建立包含多個 Features 的 Example
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(height),
          'width': _int64_feature(width),
          'depth':_int64_feature(depth),
          'image_string': _bytes_feature(image_string),
          'label': _float32_feature([label])}))

      writer.write(example.SerializeToString())
    writer.close()

def read_and_check(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path = tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])
        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])
        depth = int(example.features.feature['depth']
                    .int64_list
                    .value[0])
        image_string = (example.features.feature['image_string']
                        .bytes_list
                        .value[0])
        label = (example.features.feature['label']
                 .float_list
                 .value[0])

        image_1d = np.fromstring(image_string, dtype=np.uint8)
        image = image_1d.reshape((height, width, depth))
        io.imshow(image)
        #plt.show()

def read_and_resize(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH):
  reader = tf.TFRecordReader()
  # 讀取 TFRecords 的資料
  _, serialized_example = reader.read(filename_queue)
  # 讀取一筆 Example
  features = tf.parse_single_example(
    serialized_example,
    features = {
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64),
      'image_string': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.float32)
      })

  # 將序列化的圖片轉為 uint8 的 tensor
  image = tf.decode_raw(features['image_string'], tf.uint8)
  # 將 label 的資料轉為 float32 的 tensor
  label = tf.cast(features['label'], tf.float32)

  # 將圖片的大小轉為 int32 的 tensor
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  image = tf.reshape(image, [height, width, 3])
  # 這裡可以進行其他的圖形轉換處理 ...

  # 圖片的標準尺寸
  image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
  # 將圖片調整為標準尺寸
  resized_image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)

  # 打散資料順序
  images, labels = tf.train.shuffle_batch(
    [resized_image, label],
    batch_size=3,
    capacity=10,
    num_threads=1,
    min_after_dequeue=1)
  # capacity 要比 min_after_dequeue 更大，多出來的部分可用於預先載入資料， min_after_dequeue + (num_threads + a small safety margin) * batch_size
  # min_after_dequeue 指定打散資料用的緩衝區大小，這個值越大代表資料打散資料的效果越好，不過值越大則啟動準備時間較長，記憶體用量也較大
  return images, labels

def read_img(tfrecords_filename):
    # 建立檔名佇列
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs = 10, shuffle = True)
    # 讀取並解析 TFRecords 的資料
    images, labels = read_and_resize(filename_queue, 480, 480)

    # 初始化變數
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:
        # 初始化
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # 示範用的簡單迴圈
        for i in range(3):
            img, lab = sess.run([images, labels])
            print(img.shape)
            # 顯示每個 batch 的第一張圖
            io.imshow(img[0, :, :, :])
            plt.show()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    img = IMG('E:/Workspace_Python/testimg')
    write_img(IMG.tfrecords_filename,img.img_list, img.label_list)
    #comp = compress(IMG.tfrecords_filename)
    read_and_check(IMG.tfrecords_filename)
    read_img(IMG.tfrecords_filename)
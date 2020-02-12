# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time

import numpy as np
import tensorflow as tf

from Timer import *

from augment.weakaugment import *
from augment.randaugment import *

class TFRecord_Reader:
    def __init__(self, tfrecord_format, batch_size, image_size = [224, 224], use_repeat = False, use_augment = False, is_training = False, use_prefetch = False):
        self.use_augment = use_augment
        self.image_size = image_size
        
        dataset = tf.data.Dataset.list_files(tfrecord_format, shuffle = is_training)
        if use_repeat: 
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, buffer_size = 16 * 1024 * 1024), 
            cycle_length = 16, 
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.shuffle(1024)

        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                self.parser,
                batch_size = batch_size,
                num_parallel_calls = 2,
                drop_remainder = is_training,
            )
        )

        if use_prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.iterator = dataset.make_initializable_iterator()
        self.image_op, self.label_op = self.iterator.get_next()
        self.initializer_op = self.iterator.initializer

    def parser(self, record):
        parsed = tf.parse_single_example(
            record, 
            features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label'    : tf.FixedLenFeature([], tf.int64),

                'height'   : tf.io.FixedLenFeature([], tf.int64),
                'width'    : tf.io.FixedLenFeature([], tf.int64),
                'channel'  : tf.io.FixedLenFeature([], tf.int64),
        })

        height = tf.cast(parsed['height'], tf.int64)
        width = tf.cast(parsed['width'], tf.int64)
        channel = tf.cast(parsed['channel'], tf.int64)

        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, [height, width, channel])
        image = tf.image.resize(image, self.image_size)

        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed['label'], tf.float32)
        
        if self.use_augment:
            [image] = tf.py_func(randaugment, [image], [tf.float32])
            # [image] = tf.py_func(weakaugment, [image], [tf.float32])
        
        return image, label

# batch_size = 64
# randaugment = 2400~2500ms
# weakaugment = 30~40ms
# no augment  = 20~30ms

# reader = TFRecord_Reader('./dataset/train_*.tfrecord', batch_size = 64, is_training = False, use_augment = False, use_prefetch = False) # => 0.15sec
# reader = TFRecord_Reader('./dataset/train_*.tfrecord', batch_size = 64, is_training = False, use_augment = False, use_prefetch = True) # => 0.02sec
# reader = TFRecord_Reader('./dataset/train_*.tfrecord', batch_size = 64, is_training = False, use_augment = True, use_prefetch = False) # => 2.4sec
reader = TFRecord_Reader('./dataset/train_*.tfrecord', batch_size = 64, is_training = False, use_augment = True, use_prefetch = True) # => 2.5sec

sess = tf.Session()
sess.run(reader.initializer_op)

timer = Timer()

while True:
    try:
        timer.tik()
        batch_image_data, batch_label_data = sess.run([reader.image_op, reader.label_op])
        print('{}ms'.format(timer.tok()))

    except tf.errors.OutOfRangeError:
        break

    print(batch_image_data.shape)
    print(batch_label_data.shape)


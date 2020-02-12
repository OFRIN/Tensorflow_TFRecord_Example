# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import glob

import cv2
import copy
import time
import random

import numpy as np

from threading import Thread

from Timer import *

from augment.weakaugment import *
from augment.randaugment import *

class Teacher(Thread):
    
    def __init__(self, train_data_list, batch_size, augment_func):
        Thread.__init__(self)

        self.timer = Timer()

        self.batch_size = batch_size
        self.augment_func = augment_func

        self.train_data_list = copy.deepcopy(train_data_list)
        
    def run(self):
        while True:
            self.timer.tik()

            batch_image_data = []
            batch_label_data = []

            np.random.shuffle(self.train_data_list)
            batch_data_list = self.train_data_list[:self.batch_size]

            for data in batch_data_list:
                image, label = data
                image = cv2.resize(image, (224, 224))

                if self.augment_func is not None:
                    image = self.augment_func(image)

                batch_image_data.append(image)
                batch_label_data.append(label)

            batch_image_data = np.asarray(batch_image_data, dtype = np.float32)
            batch_label_data = np.asarray(batch_label_data, dtype = np.float32)

            print('{}ms'.format(self.timer.tok()))

# prepare dataset
label_dic = {name : label for label,name in enumerate(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])}

train_data_list = []
src_dir = './flower_dataset/train/'

for label_name in os.listdir(src_dir):
    label = label_dic[label_name]
    image_paths = glob.glob(src_dir + label_name + '/*')

    train_data_list += [[cv2.imread(image_path), label] for image_path in image_paths]

# batch_size = 64
# randaugment = 2400~2500ms, 4300~4600ms
# weakaugment = 30~40ms, 40~50ms
# no augment  = 20~30ms, 20~30ms

num_threads = 2
batch_size = 64
augment_func = None

train_threads = []

for i in range(num_threads):
    train_thread = Teacher(train_data_list, batch_size, augment_func)
    train_thread.start()
    
    train_threads.append(train_thread)



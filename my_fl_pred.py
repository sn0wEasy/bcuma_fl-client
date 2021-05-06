# -*- coding: utf-8 -*-
from my_tff import my_fl
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# ***** RqP-Client-backend **************************************
"""
:param model: 学習済みNNモデル
:param data_dir: 予測したいデータを記録したディレクトリのパス
:param mode: ラベルありデータの予測精度を出力(mode=1) 
    or ラベルなしデータの予測ラベルを出力(mode=0)
"""

def _eval(input, model, mode):
    
    if mode == 0:
        # ラベルなしデータのラベル予測
        NUM_SAMPLES = len(input[0])
        data = {
            'x':
                np.array([input[0][i].flatten() / 255.0 for i in range(0, NUM_SAMPLES)],
                            dtype=np.float32)
        }

        @tf.function
        def forward_pass(model, batch):
            predicted_y = tf.nn.softmax(
                tf.matmul(batch['x'], model['weights']) + model['bias'])
            return predicted_y

        print("shape: ", data['x'].shape)
        probability = forward_pass(model, data)
        prediction = [np.argmax(e) for e in probability]
        print("first 100 prediction result:")
        print(prediction[:100])
        return prediction

    elif mode == 1:
        # ラベルありデータの予測精度
        NUM_SAMPLES = len(input[0])
        data = {
            'x':
                np.array([input[0][i].flatten() / 255.0 for i in range(0, NUM_SAMPLES)],
                            dtype=np.float32),
            'y':
                np.array([input[1][i] for i in range(0, NUM_SAMPLES)], dtype=np.int32)
        }

        @tf.function
        def forward_pass(model, batch):
            predicted_y = tf.nn.softmax(
                tf.matmul(batch['x'], model['weights']) + model['bias'])
            return predicted_y

        probability = forward_pass(model, data)
        prediction = [np.argmax(e) for e in probability]
        print("first 100 prediction result:")
        print(prediction[:100])

        correct = 0
        for i, e in enumerate(data['y']):
            if e == prediction[i]:
                correct = correct + 1

        num_test = len(data['y'])
        acc = correct / num_test
        print("accuracy: {}".format(acc))

        return acc

    else:
        return "Invalid mode: expecting input 0 or 1"


def federated_eval(model, data_dir, mode=0):
    fl_data = [[], []]
    DIR = data_dir
    print("dir: ", DIR)
    li_dirs = os.listdir(DIR)
    for dir in li_dirs:
        _dir = DIR + dir
        files = os.listdir(_dir)
        li_file = [f for f in files if os.path.isfile(os.path.join(_dir, f))]
        for file in li_file:
            if mode == 0:
                img = np.array(Image.open(_dir + '/' + file).convert('L'))
                fl_data[0].append(img)
            elif mode == 1:
                img = np.array(Image.open(_dir + '/' + file).convert('L'))
                fl_data[0].append(img)
                fl_data[1].append(int(dir))
            else:
                return "Error"

    return _eval(fl_data, model, mode)
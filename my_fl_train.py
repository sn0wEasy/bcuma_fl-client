# -*- coding: utf-8 -*-
from my_tff import my_fl
import numpy as np
import tensorflow as tf
from PIL import Image
import os, sys
import pickle
import random

# ***** FL-Server **************************************

BATCH_SIZE = 100

def get_data_for_digit(source, digit):
    output_sequence = []
    # all_samples: digit と等しいラベルのデータの index 番号のリスト
    # {0, 1, 2}, {3, 4, 5}, {6, 7, 8, 9} の三種類に分割
    if digit != 6:
        all_samples = [i for i, d in enumerate(source[1]) if d == digit or d == digit+1 or d == digit+2]
    else:
        all_samples = [i for i, d in enumerate(source[1]) if d == digit or d == digit+1 or d == digit+2 or d == digit+3]
    print("all_samples: ", len(all_samples))
    print("digit: ", digit)
    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
            'y':
                np.array([source[1][i]
                            for i in batch_samples], dtype=np.int32)
        })
    return output_sequence


def federated_train(parent_dir, li_data_dir):
    """
    :param li_data_dir: 訓練したいデータを記録したディレクトリのリスト．ディレクトリ名はラベルになるように．
    """
    fl_data = [[], []]  # 全データ [[img], [label]]
    for dir in li_data_dir:
        DIR = parent_dir + dir + '/'
        files = os.listdir(DIR)
        li_file = [f for f in files if os.path.isfile(os.path.join(DIR, f))]
        for file in li_file:
            # img.shape: (28, 28)
            img = np.array(Image.open(DIR + file).convert('L'))
            fl_data[0].append(img)
            fl_data[1].append(int(dir)) # ラベルを数値に変換する（必要に応じて修正可）


    # データの順序をシャッフル
    p = list(zip(fl_data[0], fl_data[1]))
    random.shuffle(p)
    fl_data[0], fl_data[1] = zip(*p)

    th = 0.9
    fl_train = [fl_data[i][:int(len(fl_data[i])*th)] for i in range(0, 2)]
    fl_test = [fl_data[i][int(len(fl_data[i])*th):] for i in range(0, 2)]
    NUM_DIRS = len(li_data_dir) // 3  # データセットの個数を判定（1~3個）
    INDEX_LIST = [[0], [0, 3], [0, 3, 6]]
    federated_train_data = [get_data_for_digit(fl_train, d) for d in INDEX_LIST[NUM_DIRS-1]]
    federated_test_data = [get_data_for_digit(fl_test, d) for d in INDEX_LIST[NUM_DIRS-1]]

    model = my_fl.my_training_model(federated_train_data, federated_test_data)
    return model

def exec_fl_train(*args):
    if len(args) != 5:
        return False

    parent_dir = args[1]
    _li_data_dir = args[2]
    li_data_dir = _li_data_dir.split(",")
    model_dir = args[3]
    model_name = args[4]
    model = federated_train(parent_dir, li_data_dir)

    os.makedirs(model_dir, exist_ok=True)
    with open(model_dir + model_name, "wb") as f:
        pickle.dump(model, f)
    
    return True


if __name__ == '__main__':
    args = sys.argv
    print(os.getcwd())
    flag = exec_fl_train(*args)
    
    if flag:
        print("success")
    else:
        print("failed")
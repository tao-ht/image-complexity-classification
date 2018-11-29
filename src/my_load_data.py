import os
import numpy as np
import cv2
import os.path

# 图片的存放路径
TEST_IMAGE_PATH = "../datasets/test/test_half/"
# 图片名存放路径
test_file_dir = '../datasets/test/test_half.csv'

IMAGE_WIDHT = 200
IMAGE_HEIGHT = 200
classes = 3

def resize_norl(img_dir, w_h = 200):
    img = cv2.imread(img_dir, 0)
    weight, height = img.shape[1], img.shape[0]
    ratio = weight/height
    if ratio == 1:
        img = cv2.resize(img, (w_h, w_h))
    elif ratio > 1:
        r = weight/w_h
        h = int(height // r)
        img = cv2.resize(img, (w_h, h))
        h_l = int((w_h-h)//2)
        h_r = w_h-h-h_l
        img = np.pad(img, ((h_l, h_r), (0, 0)), 'constant')
    elif ratio < 1:
        r = height/w_h
        w = int(weight//r)
        img = cv2.resize(img, (w, w_h))
        w_l = int((w_h-w)//2)
        w_r = w_h-w-w_l
        img = np.pad(img, ((0, 0), (w_l, w_r)), 'constant')
    return img

"""get name and labels"""
def pic_name_label_load(filename_dir, type = "train"):
    X_name = []
    Y_label = []
    if type == "train":
        with open(filename_dir, "r") as file_in:
            pics_name = file_in.readlines()
            for i in pics_name:
                X_name.append(i.split(" ", 1)[0])
                Y_label.append(int((i.split(" ", 1)[1]).split()[0]))
    elif type == "test":
        with open(filename_dir, "r") as file_in:
            pics_name = file_in.readlines()
            for i in pics_name:
                X_name.append(i.split(" ", 1)[0][:-1])
                Y_label.append(0)

    Y_label = np.array(Y_label)
    return X_name, Y_label

def pic_data_label_load(data_type, X_name, Y_label):
    if data_type == 'train':
        totalNumber = len(X_name)
        img_data = np.zeros([totalNumber, IMAGE_WIDHT, IMAGE_HEIGHT])
        for img_index in range(totalNumber):
            img_dir = os.path.join(TRAIN_IMAGE_PATH, X_name[img_index])
            img = cv2.imread(img_dir, 0)
            img_data[img_index, :, :] = np.array(img)

    elif data_type == "test":
        totalNumber = len(X_name)
        img_data = np.zeros([totalNumber, IMAGE_WIDHT, IMAGE_HEIGHT])
        for img_index in range(totalNumber):
            img_dir = os.path.join(TEST_IMAGE_PATH, X_name[img_index])
            img = resize_norl(img_dir)
            img_data[img_index, :, :] = np.array(img)

    return img_data, Y_label, X_name

def onehot_encoding(Y_data, classes= 3):
    Y_label = np.zeros([Y_data.shape[0], classes], dtype=int)
    for i in range(Y_data.shape[0]):
        Y_label[i, Y_data[i]] = 1
    return Y_label

def load_data():

    test_name, test_label = pic_name_label_load(test_file_dir, "test")

    return test_name, test_label














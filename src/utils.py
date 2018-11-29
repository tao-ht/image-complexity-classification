import numpy as np
import cv2

"""resize to w_h*w_h, add padding"""
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

def load_myself(img_dir):
    totalNumber = len(img_dir)
    img_data = np.zeros([totalNumber, 200, 200])

    for i in range(totalNumber):
        img = resize_norl(img_dir[i])
        img_data[i, :, :] = np.array(img)

    img_data = np.expand_dims(img_data, axis=-1)
    img_data = normalize(img_data)

    return img_data

def normalize(X_data):

    mean = np.mean(X_data, axis=(0, 1, 2, 3))
    std = np.std(X_data, axis=(0, 1, 2, 3))
    X_data = (X_data - mean) / std

    return X_data

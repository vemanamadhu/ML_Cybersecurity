import pickle as pkl
import pdb
import os
import numpy as np
import cv2




datadir_train = '../data/images_train'
datadir_test = '../data/images_test'
images_train = os.listdir(datadir_train)
images_test = os.listdir(datadir_test)
x = []
x_dict = {}

for name in images_train:
    fname = datadir_train+'/'+name
    im = cv2.imread(fname)
    try:
        im = cv2.resize(im, (232, 232), 0, 0, cv2.INTER_LINEAR)
    except:
        continue
    x.append(im)
    x_dict[name.split('.')[0]] = im

for name in images_test:
    fname = datadir_test+'/'+name
    im = cv2.imread(fname)
    try:
        im = cv2.resize(im, (232, 232), 0, 0, cv2.INTER_LINEAR)
    except:
        continue
    x.append(im)
    x_dict[name.split('.')[0]] = im

pkl.dump(x_dict, open('../data/image_embed.pkl', 'wb'))

x = np.asarray(x)
print("Images got successfully with len", x.shape)

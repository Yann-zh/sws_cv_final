from cv2 import imread
import glob
from pandas import read_csv
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
import numpy as np

dataset_path = 'data\\images'
cropped_path = 'data\\cropped_images'
annotation_path = 'data\\annotations.csv'

train_num = 4798
test_num = 1200
total_num = 5998


def load_img_label(cropped=False):
    """
    :param: None
    :return:
    images: numpy array in BGR channels
    labels: '000'-'057'
    """
    imgs = []
    labels = []
    # limit = total_num
    for i in glob.glob((dataset_path if not cropped else cropped_path) + '\\*.png', recursive=True):
        label = i.split("images")[1][1:4]

        img = imread(i, 1)
        # cv2.IMREAD_COLOR == 1

        imgs.append(img)
        labels.append(label)

        # limit -= 1
        # if limit == 0:
        #     break
    return imgs, labels


def img_split(imgs, labels):
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(imgs, labels,
                                                                        test_size=test_num,
                                                                        train_size=train_num,
                                                                        shuffle=True)
    return imgs_train, imgs_test, labels_train, labels_test

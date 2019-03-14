import argparse
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='/Users/Eric Fowler/Downloads/carvana/train/_128x128/',
                        help='Directory for storing input data')
    parser.add_argument('--test_data_path', type=str,
                        default='/Users/Eric Fowler/Downloads/carvana/test/_128x128/',
                        help='Directory for storing input data')
    parser.add_argument('--target', type=str,
                        default='mnist',
                        choices=['carvana', 'mnist'],
                        help='MNIST or Carvana?')
    parser.add_argument('--sample', type=str,
                        default='0cdf5b5d0ce1_01',
                        help='Sample image file for sizing feature tensor')
    parser.add_argument('--numclasses', type=int,
                        default=16,
                        help='Carvana=16, MNIST=10')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--show', type=bool,
                         default=False,
                         help='Show some images?')
    parser.add_argument('--scale', type=float,
                        default=1.0,
                        help='Scaling factor for images')
    parser.add_argument('--epochs', type=int,
                        default=2,
                        help='Epochs')
    parser.add_argument('--batch_size', type=int,
                        default=10,
                        help='Cut samples into chunks of this size')
    parser.add_argument('--tb_dir', type=str,
                        default='./logs/',
                        help='Directory For Tensorboard log')
    parser.add_argument('--img_file_extension', type=str,
                        default='png',
                        help='Extension of image file names')

    return parser.parse_known_args()


def get_tensor_list(path, num_classes=16, num=None, onehot=False, extension='png'):
    files = os.listdir(path)

    if not files:
        return None

    pngs = [f for f in files if f.endswith('jpg')]  # this gets 'filename_37.png'

    number_in_filename = [name_fragment.split('_')[1] for name_fragment in pngs]  # this gets '37.png'
    number_in_filename = [name_fragment.split('.')[0] for name_fragment in number_in_filename]  # this gets '37'
    label_array = np.asarray(number_in_filename, dtype=np.int32) - 1
    if onehot == True:
        labels = np.zeros((len(label_array), num_classes), dtype=np.float32)
        labels[np.arange(len(label_array)), label_array] = 1.
    else:
        labels = label_array

    if num is None:
        num = len(pngs)

    # return must be list of tuples (filename, label array [one-hot bool])
    ret = list(zip(pngs[:num], labels[:num]))
    random.shuffle(ret)
    return ret

def scale_image(img, scale=1.0):
    if scale != 1.0:
        s = img.size
        img=img.resize((int(s[0] // scale), int(s[1] // scale)))
    return img

def flatten_image(img):
    img = img.flatten('F')
    return img

def get_image_shape(filename, scale, show=False):
    img = read_image(filename=filename, show=show, scale=scale)
    return img.size

def pixnum_from_img_shape(img_shape):
    pixel_num = 1
    for t in img_shape:
        pixel_num *= t

    return pixel_num


def read_image(filename, show, scale=1.0):
   mm = Image.open(filename).convert('LA')
   mm = scale_image(img=mm, scale=scale)

   if show == True:
       plt.imshow(mm)
       plt.show()

   return mm

import itertools as it
import sys

if sys.version[0]=='2':
    it.zip_longest=it.izip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    a = [iter(iterable)]*n
    ret = it.zip_longest(*a, fillvalue=fillvalue)
    args= [n for n in ret if n is not None]

    return args

def load_data(train_path = "", numclasses=16, num_images = None, onehot=False, extension='jpg'):
    training_list = get_tensor_list(num_classes=numclasses, path=train_path, num=num_images, onehot=onehot, extension=extension)
    random.shuffle(training_list)
    n=len(training_list)
    tr = training_list[:7 * n // 8]
    te = training_list[7 * n // 8:]
    return ([pngs for (pngs,labels) in tr],[labels for (pngs,labels) in tr]),([pngs for (pngs,labels) in te],[labels for (pngs,labels) in te])
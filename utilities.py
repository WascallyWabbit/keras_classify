import argparse
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors as clr
import tensorflow.keras.models

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool,
                        default=False,
                        help='Train model?')
    parser.add_argument('--evaluate', type=bool,
                        default=True,
                        help='Evaluate model?')
    parser.add_argument('--predict', type=bool,
                        default=True,
                        help='Predict using model?')
    parser.add_argument('--train_data_path', type=str,
                        default='/Users/Eric Fowler/Downloads/carvana/train/_128x128/',
                        help='Directory for storing input data')
    parser.add_argument('--test_data_path', type=str,
                        default='/Users/Eric Fowler/Downloads/carvana/train/_128x128/',
                        help='Directory for storing input data')
    parser.add_argument('--save_model', type=bool,
                        default=False,
                        help='Save model? Need to specify model directory and name.')
    parser.add_argument('--load_model', type=bool,
                        default=False,
                        help='Load stored model? Need to specify model directory and name.')
    parser.add_argument('--model_dir', type=str,
                        default=None,
                        help='Directory for storing model as JSON file. Leave empty to dump data')
    parser.add_argument('--model_name', type=str,
                        default=None,
                        help='Name of stored model as JSON file.')
    parser.add_argument('--save_data', type=bool,
                        default=False,
                        help='Save data? Need to specify data directory and name.')
    parser.add_argument('--load_data', type=bool,
                        default=False,
                        help='Load stored data? Need to specify data directory and name.')
    parser.add_argument('--data_dir', type=str,
                        default=None,
                        help='Directory for storing data as H5 file. Leave empty to dump data')
    parser.add_argument('--data_name', type=str,
                        default=None,
                        help='Name of stored data as H5 file.')
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
    parser.add_argument('--model', type=str,
                        default='dense',
                        choices=['dense', 'conv2d'],
                        help='Dense or Conv2D?')
    parser.add_argument('--optimizer', type=str,
                        default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer?')
    parser.add_argument('--metric', type=str,
                        default='accuracy',
                        choices=[
                            "accuracy",
                            "binary_crossentropy",
                            "categorical_hinge", "categorical_crossentropy", "cosine_proximity",
                            "hinge",
                            "kullback_leibler_divergence",
                            "logcosh",
                            "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_error",
                            "mean_squared_logarithmic_error",
                            "poisson",
                            "sparse_categorical_crossentropy", "squared_hinge"
                        ],
                        help='Metric?')
    parser.add_argument('--loss', type=str,
                        default='accuracy',
                        choices=[
                            "accuracy",
                            "binary_crossentropy",
                            "categorical_hinge","categorical_crossentropy","cosine_proximity",
                            "hinge",
                            "kullback_leibler_divergence",
                            "logcosh",
                            "mean_absolute_error","mean_absolute_percentage_error","mean_squared_error","mean_squared_logarithmic_error",
                            "poisson",
                            "sparse_categorical_crossentropy","squared_hinge"
                        ],
                        help='Loss?')
    parser.add_argument('--show_results', type=bool,
                        default=False,
                        help='Show results?')
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
                        default=16,
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

    pngs = [f for f in files if f.endswith(extension)]  # this gets 'filename_37.png'

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

def thrash_img(img):
    return img.astype(float)/255.


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

def load_data(train_path = "", test_path = "", numclasses=16, num_images = None, onehot=False, extension='jpg'):
    training_list = get_tensor_list(num_classes=numclasses, path=train_path, num=num_images, onehot=onehot, extension=extension)
    random.shuffle(training_list)

    test_list = get_tensor_list(num_classes=numclasses, path=test_path, num=num_images, onehot=onehot,
                                    extension=extension)
    random.shuffle(test_list)
    return ([pngs for (pngs,labels) in training_list],[labels for (pngs,labels) in training_list]),([pngs for (pngs,labels) in test_list],[labels for (pngs,labels) in test_list])


def make_dense_model(flags=None):
    model = tf.keras.models.Sequential(
        [
         tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='sigmoid', name='d1'),
            tf.keras.layers.Dense(128, activation='sigmoid', name='d2'),
            tf.keras.layers.Dense(64, activation='sigmoid', name='d3'),
            tf.keras.layers.Dense(16, activation='softmax', name='softmax_d4')])

    return model


def make_convnet_model(flags, shape):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32,(8,8), strides=2, activation='relu',input_shape=shape,batch_size=flags.batch_size,name='conv2d_1'),
            tf.keras.layers.Conv2D(24, (4,4), strides=1, activation='relu',name='conv2d_2'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, (3, 3), strides=2, activation='sigmoid', input_shape=shape,batch_size=flags.batch_size, name='conv2d_3'),
            tf.keras.layers.Conv2D(8, (3, 3), strides=1, activation='sigmoid', name='conv2d_4'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='sigmoid', name='d3'),
            tf.keras.layers.Dense(16, activation='softmax', name='softmax_d4')
        ])

    return model

def load_stored_model(name):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    return loaded_model

def load_stored_data(model, date_file_name):
    model.load_weights(date_file_name+'.h5')
    return model

def save_model(model, file_path_and_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_path_and_name + '.json', "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5

def process_predictions(predictions, y):
    arrays=[]
    for p in predictions:
        n = np.argmax(p, axis=0)
        arr = p[n:]
        arr = np.append(arr, p[:n])
        arrays.append(arr)
    return arrays

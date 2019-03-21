import tensorflow as tf
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import random
import numpy as np
import utilities as ut
import matplotlib.pyplot as plt
import datetime
import  PIL.Image as Img, PIL.ImageMath as IM
#file_path_carvana_test = '/Users/Eric Fowler/Downloads/carvana/test/'
file_path_carvana_train = '/Users/Eric Fowler/Downloads/carvana/train/'


def main():
    thrash = True
    print('tf version:{0}'.format(tf.VERSION))
    print('tf.keras version:{0}'.format(tf.keras.__version__))
    flags, unparsed = ut.parseArgs()
    print(flags)
    # TEST_DATA_PATH      = flags.test_data_path
    SAMPLE_FILE = flags.train_data_path + flags.sample + '.' + flags.img_file_extension
    img = ut.read_image(filename=SAMPLE_FILE, show=False)
    img = np.array(img)
    if thrash == True:
        img = ut.thrash_img(img)

    IMG_SHAPE=img.shape
    (x_train, y_train), (x_test, y_test)=ut.load_data(numclasses=flags.numclasses, train_path=flags.train_data_path, onehot=True, extension=flags.img_file_extension)

    print('IMG_SHAPE:{0},  y_train shape:{1}'.format(IMG_SHAPE,y_train[0].shape))

    if flags.model == 'dense':
        model = make_dense_model(flags=flags)
    elif flags.model  == 'conv2d':
        model = make_convnet_model(flags=flags, shape=IMG_SHAPE)

    print('Saving in {0}'.format(flags.tb_dir + datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    tensorboard = TensorBoard(log_dir=flags.tb_dir + '{0}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    adam=tf.keras.optimizers.Adam(lr=flags.learning_rate)
    model.compile(optimizer=adam,
                  loss=flags.loss,
                  metrics=[flags.metric]
                  )

    scores = []
    for epoch in range(1,flags.epochs+1):
        print('Epoch:{0} of {1}'.format(epoch, flags.epochs))
        n = len(x_train)
        for batch in range(0,len(x_train), flags.batch_size):
            print('Batch {0} of {1}, epoch {2} of {3}.'.format(batch,n, epoch, flags.epochs))
            bunch_x, bunch_y = x_train[batch:batch+flags.batch_size], y_train[batch:batch+flags.batch_size]
            if len(bunch_x) < flags.batch_size: # skip partial batches
                print('Skipping {0} samples..'.format(len(bunch_x)))
                continue

            xs = []
            ys = []
            for datum in range(len(bunch_x)):
                file = bunch_x[datum]
                img = ut.read_image(filename=flags.train_data_path+file, show=False)
                img=np.array(img)
                if thrash == True:
                    img = ut.thrash_img(img)
                xs.append(img)
                ys.append(bunch_y[datum])

            X= np.stack(xs, axis=0)
            Y= np.stack(ys, axis=0)

            score_before = model.evaluate(x=X,y=Y, batch_size=flags.batch_size)

            _ = model.fit(x=X, y=Y, shuffle=flags.shuffle, callbacks=[tensorboard])

            score_after = model.evaluate(x=X,y=Y, batch_size=flags.batch_size)

            if score_before == score_after:
                print("Scores before and after training are identical")

            scores.append(score_after)
            if epoch == 0 and batch == 0:
                model.summary()

            print('Score:{0}'.format(score_after))

        loss,acc = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
    print("Average loss:{0}  Average accuracy:{1}%".format(np.mean(loss), 100*np.mean(acc)))

    pass


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
            # tf.keras.layers.Conv2D(8,(8,8), strides=2, activation='relu',input_shape=shape,batch_size=flags.batch_size,name='conv2d_1'),
            # tf.keras.layers.Conv2D(64, (5, 5), name="conv2_5x5"),
            # tf.keras.layers.MaxPool2D(name='pool1'),
            # tf.keras.layers.Conv2D(64, [5, 5], name="conv3_5x5"),
            # tf.keras.layers.Conv2D(128, [3, 3], name="conv4_3x3"),
            # tf.keras.layers.MaxPool2D([2, 2], name='pool2'),
            # tf.keras.layers.Conv2D(128, [3, 3], name="conv5_3x3"),
            # tf.keras.layers.MaxPool2D([2, 2], name='pool3'),
            # tf.keras.layers.Conv2D(32, [1, 1], name="conv6_1x1")
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='sigmoid', name='d3'),
            tf.keras.layers.Dense(16, activation='softmax', name='softmax_d4')
        ])

    return model


if __name__ == '__main__':
    main()
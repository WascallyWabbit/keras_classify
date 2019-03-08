import tensorflow as tf
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import random
import numpy as np
import utilities as ut
import matplotlib.pyplot as plt
import datetime
#file_path_carvana_test = '/Users/Eric Fowler/Downloads/carvana/test/'
file_path_carvana_train = '/Users/Eric Fowler/Downloads/carvana/train/'


def main():
    FLAGS, unparsed = ut.parseArgs()
    print(FLAGS)
    # TEST_DATA_PATH      = FLAGS.test_data_path
    SAMPLE_FILE = FLAGS.train_data_path + FLAGS.sample + '.' + FLAGS.img_file_extension

    img = ut.read_image(filename=SAMPLE_FILE, show=False)
    img = np.array(img)
    IMG_SHAPE=img.shape
    (x_train, y_train), (x_test, y_test)=ut.load_data(numclasses=FLAGS.numclasses, train_path=FLAGS.train_data_path, onehot=True)

    print(f'IMG_SHAPE:{IMG_SHAPE},  y_train shape:{y_train[0].shape}')

    model = tf.keras.models.Sequential(
    [
    tf.keras.layers.Conv2D(4,(8,8), strides=4, activation='relu',input_shape=IMG_SHAPE,batch_size=FLAGS.batch_size),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='softmax', name='softmax')])
    print(f'Saving in {FLAGS.tb_dir+datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')
    tensorboard = TensorBoard(log_dir=FLAGS.tb_dir+f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')

    optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy']
                  )

    model.summary()

    for i in range(FLAGS.epochs):
        print(f'Epoch:{i} of {FLAGS.epochs}')
        n = len(x_train)
        for batch in range(0,len(x_train), FLAGS.batch_size):
            print(f'Batch {batch} of {n}.')
            bunch_x, bunch_y = x_train[batch:batch+FLAGS.batch_size], y_train[batch:batch+FLAGS.batch_size]
            if len(bunch_x) < FLAGS.batch_size: # skip partial batches
                print(f'Skipping {len(bunch_x)} samples..')
                continue

            xs = []
            ys = []
            for datum in range(len(bunch_x)):
                file = bunch_x[datum]
                img = ut.read_image(filename=FLAGS.train_data_path+file, show=False)
                img=np.array(img)
                xs.append(img)
                ys.append(bunch_y[datum])

            X= np.stack(xs, axis=0)
            Y= np.stack(ys, axis=0)
            model.fit(x=X, y=Y,steps_per_epoch=10, callbacks=[tensorboard])

            score = model.evaluate(x=X,y=Y, batch_size=FLAGS.batch_size)
            print(f'Score:{score}')

    pass



if __name__ == '__main__':
    main()
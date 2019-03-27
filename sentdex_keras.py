import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import utilities as ut
import datetime

file_path_carvana_train = '/Users/Eric Fowler/Downloads/carvana/train/'
file_path_carvana_test = '/Users/Eric Fowler/Downloads/carvana/train/'

def main():
    start_time = datetime.datetime.now().strftime("%Y%d%H%M%S")
    #print(f'datetime.datetime={str(datetime.datetime)}, datetime.date={str(datetime.date)}, strftime():{datetime.datetime.now().strftime("%Y%d%H%M%S")}')
    thrash = True
    print('tf version:{0}'.format(tf.VERSION))
    print('tf.keras version:{0}'.format(tf.keras.__version__))
    flags, unparsed = ut.parseArgs()
    print(flags)
    SAMPLE_FILE = flags.train_data_path + flags.sample + '.' + flags.img_file_extension
    img = ut.read_image(filename=SAMPLE_FILE, show=False)
    img = np.array(img)
    if thrash == True:
        img = ut.thrash_img(img)

    IMG_SHAPE=img.shape
    (x_train, y_train), (x_test, y_test)=ut.load_data(numclasses=flags.numclasses, train_path=flags.train_data_path, onehot=True, extension=flags.img_file_extension)

    print('IMG_SHAPE:{0},  y_train shape:{1}'.format(IMG_SHAPE,y_train[0].shape))

    if flags.load_model:
        model = ut.load_stored_model(name=flags.model_dir + flags.model_name)
    elif flags.model == 'dense':
        model = ut.make_dense_model(flags=flags)
    elif flags.model  == 'conv2d':
        model = ut.make_convnet_model(flags=flags, shape=IMG_SHAPE)
    else:
        print('No model, no hope. Quitting...')
        return

    print('Saving in {0}'.format(flags.tb_dir + start_time))
    tensorboard = TensorBoard(log_dir=flags.tb_dir + '{0}'.format(start_time))

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

    if flags.save_model:
        model_name = flags.model_name if flags.model_name != None else start_time
        ut.save_model(model, flags.model_dir+model_name)
        print(f'Saved model to disk, json in {flags.model_dir + model_name + ".json"}')

    if flags.save_data:
        data_name = flags.data_name if flags.data_name != None else start_time
        model.save_weights(flags.data_dir + data_name + ".h5")

    if flags.evaluate:
        test_scores = []
        if flags.evaluate:
            n = len(x_test)
            for batch in range(0, len(x_train), flags.batch_size):
                print('Batch {0} of {1}, epoch {2} of {3}.'.format(batch, n, epoch, flags.epochs))
                bunch_x, bunch_y = x_test[batch:batch + flags.batch_size], y_test[batch:batch + flags.batch_size]
                if len(bunch_x) < flags.batch_size:  # skip partial batches
                    print('Skipping {0} samples..'.format(len(bunch_x)))
                    continue

                xs = []
                ys = []
                for datum in range(len(bunch_x)):
                    file = bunch_x[datum]
                    img = ut.read_image(filename=flags.test_data_path + file, show=False)
                    img = np.array(img)
                    if thrash == True:
                        img = ut.thrash_img(img)
                    xs.append(img)
                    ys.append(bunch_y[datum])

                X = np.stack(xs, axis=0)
                Y = np.stack(ys, axis=0)

                score = model.evaluate(x=X, y=Y, batch_size=flags.batch_size)
                test_scores.append(score)
                print(f'Test score:{0}'.format(score))

            print(f'Average score:{0},{1}'.format(np.mean(test_scores[0]),np.mean(test_scores[1])))

    pass


if __name__ == '__main__':
    main()
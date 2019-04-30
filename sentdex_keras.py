import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import utilities as ut
import datetime
import matplotlib.pyplot as plt

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
    (x_train, y_train), (x_test, y_test)=ut.load_data(numclasses=flags.numclasses, train_path=flags.train_data_path, test_path=flags.test_data_path, onehot=True, extension=flags.img_file_extension)

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

    if flags.load_data:
        model = ut.load_stored_data(model=model, date_file_name=flags.data_dir + flags.data_name)

    print('Saving in {0}'.format(flags.tb_dir + start_time))
    tensorboard = TensorBoard(log_dir=flags.tb_dir + '{0}'.format(start_time))

    adam=tf.keras.optimizers.Adam(lr=flags.learning_rate)

    model.compile(optimizer=adam,
                  loss=flags.loss,
                  metrics=[flags.metric]
                  )

    if flags.train == True:
        scores = []
        for epoch in range(flags.epochs):
            print('Epoch:{0} of {1}'.format(epoch+1, flags.epochs))
            n = len(x_train)
            for batch in range(0,len(x_train), flags.batch_size):
                print('Batch {0} of {1}, epoch {2} of {3}.'.format(batch+1,n+1, epoch+1, flags.epochs))
                bunch_x, bunch_y = x_train[batch:batch+flags.batch_size], y_train[batch:batch+flags.batch_size]
                if len(bunch_x) < flags.batch_size: # skip partial batches
                    print('Skipping {0} samples..'.format(len(bunch_x)))
                    continue

                xs = []
                ys = []
                print("Iterating {0} samples".format(len(bunch_x)))
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
        print('Saved model to disk, json in {0}'.format(flags.model_dir + model_name + ".json"))

    if flags.save_data:
        data_name = flags.data_name if flags.data_name != None else start_time
        model.save_weights(flags.data_dir + data_name + ".h5")
        print('Saved data to disk in {0}'.format(flags.model_dir + data_name + ".h5"))

    test_scores = []
    predictions = []
    if flags.evaluate or flags.predict:
        n = len(x_test)
        nTotal = 0
        sums_array = None
        for batch in range(0, len(x_test), flags.batch_size):
            print('Batch {0} of {1}.'.format(batch+1, n+1))
            bunch_x, bunch_y = x_test[batch:batch + flags.batch_size], y_test[batch:batch + flags.batch_size]
            if len(bunch_x) < flags.batch_size:  # skip partial batches
                print('Skipping {0} samples..'.format(len(bunch_x)))
                continue

            xs = []
            ys = []
            for d in range(len(bunch_x)):
                file = bunch_x[d]
                img = ut.read_image(filename=flags.test_data_path + file, show=False)
                img = np.array(img)
                if thrash == True:
                    img = ut.thrash_img(img)
                xs.append(img)
                ys.append(bunch_y[d])

            X = np.stack(xs, axis=0)
            Y = np.stack(ys, axis=0)

            if flags.evaluate:
                score = model.evaluate(x=X, y=Y, batch_size=flags.batch_size)
                test_scores.append(score)
                print('Test score:{0}'.format(score))


            if flags.predict:
                prediction = model.predict(X, verbose=2)
                processed_predictions = ut.process_predictions(prediction, Y)

                for pp in processed_predictions:
                    if sums_array is None:
                        sums_array = np.zeros_like(pp)
                    sums_array = np.add(sums_array, pp)
                    nTotal = nTotal+1

                pass
        sums_array /= nTotal
        if predictions != None:
            pass


        print('Average score:{0},{1}'.format(np.mean(test_scores[0]),np.mean(test_scores[1])))

        if flags.show_results:
            y_axis = np.arange(0, 1.0, 1.0/float(len(sums_array)))
            plt.plot(y_axis,sums_array)
            plt.show()

    pass


if __name__ == '__main__':
    main()
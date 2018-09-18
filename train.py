import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from vessel_images import VesselImages
import matplotlib.pyplot as plt


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                    activation='relu',
                    input_shape=(18, 18, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


def main():

    batch_size = 128
    epochs = 5


    vessel = VesselImages("DRIVE", 18)
    images = vessel.load_images(training=True)
    masks = vessel.load_masks(training=True)
    images = images[:10]
    masks = masks[:10]

    X, y = vessel.prepare_data_for_the_model(images,masks)

    X = X/255
    bound = int(0.7 * y.shape[0])

    X_train, X_test, y_train, y_test = X[:bound], X[bound:], y[:bound], y[bound:]

    model = create_model()

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])


    history = AccuracyHistory()

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[history])

    model.save("cnn_model.h5")
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, epochs + 1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    

if __name__ == "__main__":
    main()
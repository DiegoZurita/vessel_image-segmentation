import keras
from vessel_images import VesselImages
import numpy as np
import matplotlib.pyplot as plt

def main():
    vessel = VesselImages("DRIVE", 18)
    model = keras.models.load_model("cnn_model.h5")

    images = vessel.load_images(training=True)
    masks = vessel.load_masks(training=True)

    image = images[19]
    mask = masks[19]


    X = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            X.append(vessel.get_square_of_pixel(image, [i, j]))

    X = np.array(X)
    X = X/255

    y = model.predict(X, verbose=1)

    print("X", X.shape)
    print("image", image.shape) #(584, 565, 3)


    is_vessel = y[:, :1]
    is_vessel = is_vessel.flatten()
    is_vessel[is_vessel >= 0.8] = 255
    is_vessel[is_vessel < 0.8] = 0
    is_vessel = is_vessel.reshape((584, 565))

    print("is_vessel", is_vessel.shape)
    print("mask", mask.shape)
    

    plt.imsave("output_predicted", is_vessel)
    plt.imsave("output_desired", mask)



if __name__ == "__main__":
    main()
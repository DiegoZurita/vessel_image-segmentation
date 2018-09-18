import matplotlib.pyplot as plt
import numpy as np
import os

class VesselImages:

    def __init__(self, root_images_folder, square_size):
        self.root_images_folder = root_images_folder
        self.TEST_FOLDER = "test"
        self.TRAIN_FOLDER = "training"
        self.square_size = square_size

    def load_images(self, training=False):
        images = self.get_images_uris(training)
        return np.array([plt.imread(path) for path in images])
    
    def get_images_uris(self, training=False):
        images_path = ""
        if training:
            images_path = os.path.join(self.root_images_folder, self.TRAIN_FOLDER + "/images")
        else:
            images_path = os.path.join(self.root_images_folder, self.TEST_FOLDER + "/images")
        
        return [os.path.join(images_path, i) for i in os.listdir(images_path)]

    def load_masks(self, training=False):
        masks_path = self.get_masks_uris(training)

        return np.array([plt.imread(i) for i in masks_path])

    def get_masks_uris(self, training=False):
        masks_path = ""
        if training:
            masks_path = os.path.join(self.root_images_folder, self.TRAIN_FOLDER + "/1st_manual")
        else:
            masks_path = os.path.join(self.root_images_folder, self.TEST_FOLDER + "/1st_manual")
        
        return [os.path.join(masks_path, i) for i in os.listdir(masks_path)]

    def discriminate_pixel_of_mask(self, mask):
        vessel = []
        not_vessel = []

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 255:
                    vessel.append((i, j))
                else:
                    not_vessel.append((i, j))

        return np.array(vessel), np.array(not_vessel)

    def discriminate_masks(self, masks):
        result = []        
        for mask in masks:
            vessel, not_vessel = self.discriminate_pixel_of_mask(mask)
            result.append(np.array([vessel, not_vessel]))
        return np.array(result)

    def get_random_not_vessel_pixels(self, not_vessel_pixels, n):
        return not_vessel_pixels[np.random.choice(not_vessel_pixels.shape[0], n)]

    def get_square_of_pixel(self, image, pixel):
        middle_square = int(self.square_size/2)

        cima = baixo = esquerda = direita = 0    

        #Vertical
        if pixel[0] - middle_square < 0:
            cima = -1 * (pixel[0] - middle_square)
        if pixel[0] + middle_square > image.shape[0]:
            baixo = -1 * (image.shape[0] - pixel[0] - middle_square)
        #Horizontal
        if pixel[1] - middle_square < 0:
            esquerda = (-1 * (pixel[1] - middle_square))
        if pixel[1] + middle_square > image.shape[1]:
            direita = -1 * (image.shape[1] - pixel[1] - middle_square)

        image_cropped = image[pixel[0] + cima - middle_square : pixel[0] - baixo + middle_square , \
                               pixel[1] + esquerda - middle_square : pixel[1]  - direita + middle_square]
    
        
        if image_cropped.shape[0] == self.square_size and image_cropped.shape[1] == self.square_size:
            return image_cropped
        else:
            return self.pad_image(
                image_cropped, 
                {
                    "cima": cima,
                    "baixo": baixo,
                    "esquerda": esquerda,
                    "direita": direita
                },
                (self.square_size, self.square_size, 3),
                0
            )

    def pad_image(self, image, coord, final_shape, value):
        new_image = value * np.ones(final_shape, dtype=int)
        
        new_image[coord["cima"]: final_shape[0] - coord["baixo"], coord["esquerda"]: final_shape[1] - coord["direita"]] = image  
        return new_image

    def prepare_data_for_the_model(self, images, masks):
        pixels_data_from_mask = self.discriminate_masks(masks)

        y = None
        X = []
        
        for i in range(images.shape[0]):
            image_data = pixels_data_from_mask[i]
            vessel_pixels = image_data[0]
            not_vessel_pixels = self.get_random_not_vessel_pixels(image_data[1], 2 * len(vessel_pixels))

            
            vessels_output = np.ones((vessel_pixels.shape[0], 2))
            vessels_output[:, 1:] = vessels_output[:, 1:] * 0

            not_vessels_output = np.ones((not_vessel_pixels.shape[0], 2))
            not_vessels_output[:, :1] = not_vessels_output[:, :1] * 0

            pixels_concat = np.concatenate((vessel_pixels, not_vessel_pixels), axis=0)
            y_concat = np.concatenate((vessels_output, not_vessels_output), axis=0)

            for pixel in pixels_concat:
                X.append(self.get_square_of_pixel(images[i], pixel))

            if y is None:
                y = y_concat
            else:
                y = np.concatenate((y, y_concat), axis=0)

        X = np.array(X)

        #shufle
        indexes = np.random.permutation(y.shape[0])

        X = X[indexes]
        y = y[indexes]

        return X, y

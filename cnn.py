"""
Convolutional neural network architecture.
Author: Fejiro Aigboro + Hanna Abrahem
Date: April 26th
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import parse_args
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from scipy.ndimage import gaussian_filter

class CNNmodel(tf.keras.Model):
    """
    A convolutional neural network; the architecture is:
    Conv -> ReLU -> Conv -> ReLU -> Dense
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5,5), use_bias=True)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(3,3), use_bias=True)
        self.relu2 = tf.keras.layers.ReLU()
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(5749, activation=tf.keras.activations.softmax, use_bias=True)

    def call(self, x):
        """ 
            x (tensor) : image vectors
            
            return (tensor) : the final weights of the neural network
        """
        out1 = self.conv1(x)
        out2 = self.relu1(out1)
        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.flatten(out4)
        out6 = self.dense(out5)

        return out6
    
def main(train=True, model_name="cnn1")->None:
    """trains or tests model"""
    if model_name == "cnn1":
        lfw_people = fetch_lfw_people(data_home="/homes/fanigboro/cmsc360/", color=True, min_faces_per_person=10)
        max = 158
    else:
        max = 5479
        lfw_people = fetch_lfw_people(data_home="/homes/fanigboro/cmsc360/data", color=True)

    images = lfw_people.images
    images  = np.pad(images, ((0, 0), (0, 0), (0, 48 - 47), (0, 0))) # pad images
    target = lfw_people['target']
    
    names = lfw_people.target_names

    if train:
        model = CNNmodel()
        model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        if model == "cnn1":
            model_ckpt = tf.keras.callbacks.ModelCheckpoint(
                "cnn_ckpt", monitor="val_accuracy", save_best_only=True)
            history = model.fit(images, target, epochs=20, validation_split=0.1, callbacks=[model_ckpt], batch_size=32)
        else:
            history = model.fit(images, target, epochs=10, batch_size=32)
        model.save(f"{model_name}.tf", save_format='tf')
        return
    else:
        evaluate(names, images, target, model_name, max)
        
        
def evaluate(names, images, target, model, max)->None:
    """Gets accuracies on distortions"""
    # get model
    model = tf.keras.models.load_model(f"{model}.tf")

    actual = []
    pred = []
    correct = []
    
    # gets images cnn classifies correctly
    for i in range(len(names)):
        test_image = [images[i]]
        test_pred_cnn = model(test_image)
        pred_cnn = np.argmax(test_pred_cnn, axis=1)
        if pred_cnn <= max: # max changes based on model
            predicted_name = names[pred_cnn]
            
            actual_name = names[target[i]]
            actual.append(actual_name) 
            pred.append(predicted_name[0])

            if(predicted_name[0] == actual_name):
                correct.append([test_image[0], actual_name])

    index = 1
    predicted = [0, 0, 0, 0]
    labels = ["invert", "contrast", "blur", "noise"]
    # of the correct images, get predictions for their distorted pictures
    for image, real_name in correct:
        pred = predict_fake(model, image, real_name, names, max)
        for j in range(len(predicted)):
            predicted[j] += pred[j]
        index += 1
    
    predicted_accuracy = [pred/len(correct) for pred in predicted]
    
    for i in range(len(predicted_accuracy)):
        print(f"Distortion type: {labels[i]}, Accuracy: {round(predicted_accuracy[i]*100,2)}%")
    

def predict_fake(model, image, actual, names, max)->list:
    """Makes all the distortions and makes predictions for each"""
    fake_images = [invert_image(image), alter_contrast(image, factor=5), \
        blur_image(image), add_random_noise(image)]
    
    preds_for_fake = [model([img]) for img in fake_images]
    preds_cnn = [np.argmax(pred, axis=1) for pred in preds_for_fake]
    labels = ["invert", "contrast", "blur", "noise"]
    predicted = [0, 0, 0, 0]
    for i in range(len(preds_cnn)):
        if preds_cnn[i] <= max:
            if names[preds_cnn[i][0]] == actual:
                predicted[i] += 1
                plt.imshow(image/255)
                plt.savefig(f"correct_image_distortion/real{actual}.png", format="png")
                plt.imshow(fake_images[i]/255)
                plt.savefig(f"correct_image_distortion/fake{actual}{labels[i]}{i}.png", format="png")
            
    return predicted
    
def invert_image(image):
    # Normalize the image to [0, 1]
    normalized_image = image / 255.0
    # Invert the normalized image
    inverted_image = 1.0 - normalized_image
    # Scale the inverted image back to [0, 255]
    inverted_image = inverted_image * 255
    return inverted_image

def alter_contrast(image, factor):
    mean = np.mean(image)
    image = (1-factor)*mean + factor*image
    np.clip(image, 0., 1.)
    return image

def blur_image(image):
    blurred_image = gaussian_filter(image, sigma=1)
    return blurred_image

def add_random_noise(image):
    # Normalize the image to [0, 1]
    normalized_image = image / 255.0
    # Generate noise
    noise = np.random.normal(0, 0.05, normalized_image.shape)
    # Add noise to the normalized image
    noisy_image = normalized_image + noise
    # Clip the values of the noisy image to [0, 1]
    noisy_image = np.clip(noisy_image, 0., 1.)
    # Scale the noisy image back to [0, 255]
    noisy_image = noisy_image * 255
    return noisy_image

    
if __name__ == "__main__":
    opts = parse_args()
    model = opts.CNNModel
    train = True if opts.train == "True" else False
    main(train=train, model=model)
    
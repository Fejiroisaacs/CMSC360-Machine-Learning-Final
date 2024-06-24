"""
Generative Adversarial Network
Author: Fejiro Aigboro + TensorFlow tutorial - https://www.tensorflow.org/tutorials/generative/dcgan
Date: May 1st
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
from IPython import display
from helper import parse_args
from sklearn.datasets import fetch_lfw_people

def main(EPOCHS = 500, train_model=False):

    lfw_people = fetch_lfw_people(data_home="/homes/fanigboro/cmsc360/data/", color=True)

    if train_model:
        generator = make_generator_model()
        discriminator = make_discriminator_model()
    else:
        # finetuning GAN
        generator = tf.keras.models.load_model('generator_color.tf')
        discriminator = tf.keras.models.load_model('discriminator_color.tf')

    images = lfw_people.images
    train_images  = np.pad(images, ((0, 0), (0, 0), (0, 48 - 47), (0, 0)))
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    noise_dim = 100
    num_examples_to_generate = 16

    BUFFER_SIZE = 14000
    BATCH_SIZE = 256
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    train(train_dataset, EPOCHS, generator, discriminator, checkpoint, checkpoint_prefix, seed)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    generator.save("generator_color.tf", save_format='tf')
    discriminator.save("discriminator_color.tf", save_format='tf')

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(31*24*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((31, 24, 256)))
    assert model.output_shape == (None, 31, 24, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 31, 24, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 62, 24, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 62, 48, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[62, 48, 3]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output, cross_entropy):
    """Loss function for discriminator"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, cross_entropy):
    """Loss function for generator"""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer):
    """Train function from tensorflow"""
    BATCH_SIZE = 256
    noise_dim = 100
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(images, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print(f"Gen Loss: {gen_loss}, Disc Loss: {disc_loss}")
    
    
def train(dataset, epochs, generator, discriminator, checkpoint, checkpoint_prefix, seed, finetune=False):
    """Train function from tensorflow, edited by me"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, \
                cross_entropy, generator_optimizer, discriminator_optimizer)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                            epochs,
                            seed)
    if finetune:
        return generator, discriminator
    
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('GAN_images_colored/image_at_epoch_{:04d}.png'.format(epoch))
    
    
if __name__ == "__main__":
    opts = parse_args()
    epochs = opts.Epochs
    train_model = True if opts.train == "True" else False
    main(EPOCHS=epochs, train_model=train_model)
  
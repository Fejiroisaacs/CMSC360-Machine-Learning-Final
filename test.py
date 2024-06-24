"""
Finetuning GAN + comparative_stats for each cnn model and GAN generators
Author: Fejiro Aigboro + Hanna Abrahem
Date: May 13th
"""
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from GAN import train
import os
from helper import parse_args

def main(epochs=500, finetune=True) -> None:
    
    base_GAN = tf.keras.models.load_model('generator_color.tf')

    if finetune:
        lfw_people = fetch_lfw_people(data_home="/homes/fanigboro/cmsc360/data/", color=True)
        finetune_GAN(base_GAN, lfw_people, EPOCHS=epochs)
    else:
        finetuned_GAN = tf.keras.models.load_model('GeorgeBush_generator_fin.tf')
        cnn_model1 = tf.keras.models.load_model('cnn1.tf')
        cnn_model2 = tf.keras.models.load_model('cnn2.tf')
        
        comparative_stats(base_GAN, finetuned_GAN, cnn_model1, cnn_model2)
                       
            
def comparative_stats(base_GAN, finetuned_GAN, cnn_model1, cnn_model2) -> None:
    """Generate images from GAN models and evaluate with CNN models"""
    num_examples_to_generate = 1000
    noise_dim = 100
    
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    # generate images
    base_GAN_images = base_GAN(seed, training=False)
    ft_GAN_images = finetuned_GAN(seed, training=False)
    
    # getting predictions for models
    cnn1_pred_base = np.argmax(cnn_model1(base_GAN_images), axis=1)
    cnn2_pred_base = np.argmax(cnn_model2(base_GAN_images), axis=1)
    
    cnn1_pred_ft = np.argmax(cnn_model1(ft_GAN_images), axis=1)
    cnn2_pred_ft = np.argmax(cnn_model2(ft_GAN_images), axis=1)

    right_pred = [0, 0, 0, 0]
    for i in range(base_GAN_images.shape[0]):
        # id=35 is GWBush id for CNN1, id=1871 is GWBush id for CNN2
        # checking predictions and saving images
        try:
            if cnn1_pred_base[i] == 35 or cnn2_pred_base[i] == 1871:
                base_gan_img = base_GAN_images[i, :, :, :]
                if cnn1_pred_base[i] == 35 and cnn2_pred_base[i] == 1871:
                    right_pred[0] += 1
                    right_pred[1] += 1
                    save_img(base_gan_img, f"base_cnn_images/fake_bush{i}.png")
                elif cnn1_pred_base[i] == 35:
                    right_pred[0] += 1
                    save_img(base_gan_img, f"cnn1_base_images/fake_bush{i}.png")                
                elif cnn2_pred_base[i] == 1871:
                    right_pred[1] += 1
                    save_img(base_gan_img, f"cnn2_base_images/fake_bush{i}.png")
                    
            if cnn1_pred_ft[i] == 35 or cnn2_pred_ft[i] == 1871:
                ft_gan_img = ft_GAN_images[i, :, :, :]
                if cnn1_pred_ft[i] == 35 and cnn2_pred_ft[i] == 1871:
                    right_pred[2] += 1
                    right_pred[3] += 1
                    save_img(ft_gan_img, f"ft_cnn_images/fake_bush{i}.png")
                elif cnn1_pred_ft[i] == 35:
                    right_pred[2] += 1
                    save_img(ft_gan_img, f"cnn1_ft_images/fake_bush{i}.png")
                elif cnn2_pred_ft[i] == 1871:
                    right_pred[3] += 1
                    save_img(ft_gan_img, f"cnn2_ft_images/fake_bush{i}.png")
        except:
            # saving images might yield an error if i am not running on my local pc
            pass
    
    # getting accuracies
    gan_bush_accuraies = [round(pred/base_GAN_images.shape[0], 4) for pred in right_pred]
    labels = ["cnn1_base", "cnn2_base", "cnn1_ft", "cnn2_ft"]
    
    print(
        "Description:\n\
        cnn1: CNN trained with 10pics per person \n\
        cnn2: CNN trained with all images \n\
        base: Base GAN trained \n\
        ft: GAN fine tuned on George W Bush images\n"
    )
    
    for i in range(len(gan_bush_accuraies)):
        print(f"CNN predicted fake GAN image {gan_bush_accuraies[i]}% ({right_pred[i]}/{num_examples_to_generate}) of the time for: {labels[i]}") 
    
    
def save_img(image, path)-> None:
    """Save generated images"""
    plt.imshow(image)
    plt.savefig(f"test_images/{path}", format="png")
    plt.clf()


def finetune_GAN(generator, lfw_people, EPOCHS)-> None:
    """Fine tune GAN using methods similiar to inital training"""
    noise_dim = 100
    num_examples_to_generate = 16
    BUFFER_SIZE = 32
    BATCH_SIZE = 32

    names = lfw_people.target_names

    # Find the index of George W. Bush in the list of names
    idx = list(names).index('George W Bush') # idx = 1871

    # Get the images of George W. Bush
    images = lfw_people.images[lfw_people.target == idx]
    train_images  = np.pad(images, ((0, 0), (0, 0), (0, 48 - 47), (0, 0)))
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator = tf.keras.models.load_model('discriminator_color.tf')
    checkpoint_dir = './finetuning_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    ft_gen, ft_disc = train(train_dataset, EPOCHS, generator, discriminator, checkpoint, checkpoint_prefix, seed, finetune=True)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    ft_gen.save("GeorgeBush_generator_fin.tf", save_format='tf')
    ft_disc.save("GeorgeBush_discriminator_fin.tf", save_format='tf')
    
    
if __name__ == "__main__":
    opts = parse_args()
    epochs = opts.Epochs
    finetune = True if opts.train == "True" else False
    main(epochs=epochs, finetune=finetune)
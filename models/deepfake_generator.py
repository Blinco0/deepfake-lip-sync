import numpy as np   # linear algebra
import cv2
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow import keras

from keras import layers
from utils import get_face_from_video



"""
    PLACEHOLDER.....
    Generates a number
    Input is a random seed of size (100,)
    The generator is supposed to take
    
    
    
    
    image and also an audio frame extracted (its gonna match the frame from the video)
    
    we would cut off the bottom half of the image (its gonna be filled in with black)
    
    and then we use the generator 
"""
def make_number_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU()) 

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU()) 

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU()) 

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

model = make_number_generator()


def identity_encoder():
    """A model for receiving the image as a numpy array."""
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(3, 7, 1, activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.Conv2D(32, 5, (1, 2), activation='relu'),
            tf.keras.layers.Conv2D(64, 5, 2, activation='relu'),
            tf.keras.layers.Conv2D(128, 5, 2, activation='relu'),
            tf.keras.layers.Conv2D(256, 3, 2, activation='relu'),
            tf.keras.layers.Conv2D(512, 3, 2, activation='relu'),
            tf.keras.layers.Conv2D(512, 3, 1, activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="image")
    model.summary()
    return model


def audio_encoder():
    """A model for receiving the audio as a numpy array."""
    loudness = 80  # Probably the decibel of the sound. Not sure how it's like that. tbf, I actually have no clue if the variable name is c
                    # correct for this value.
    step_size = 34 # Presumably the time each frame occupies. So a frame can last like 1/30th of a second if the video is in 30FPS.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(3, 7, 1, activation='relu', input_shape=(80, step_size, 1), name="input_audio"),
        tf.keras.layers.Conv2D(32, 5, (1, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, 5, 2, activation='relu'),
        tf.keras.layers.Conv2D(128, 5, 2, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 1, activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="audio")
    model.summary()
    return model


identity_model = identity_encoder()
audio_model = audio_encoder()

x = layers.Concatenate([identity_model.layers[-1].output, audio_model.layers[-1].output])
# I just copied it from the Simpson one. The parameters, to be exact. And I have no clue what the parameters are.
combined_output = layers.Conv2DTranspose(filters=3, kernel_size=[5,5], strides=[1,1], padding="SAME",
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                            name="logits")(x)

combined_model = keras.Model(inputs=[identity_model.layers[0].input, audio_model.layers[0].input], outputs=[combined_output])
combined_model.summary()


# Definitely liable to change. Generator is a combined model.
combined_model.compile(
    optimizer='adam',
    loss='CategoricalCrossentropy',
    metrics=['accuracy']
)


def mask_image(img_path: str):
    # Read an input image as a gray image
    img = cv2.imread(img_path)

    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv2.rectangle(mask, (0,0), (255,127), 255, -1)

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(img, img, mask = mask)

    # display the mask, and the output image
    cv2.imshow('Masked Image',masked_img)
    
    # save the masked image.
    cv2.imwrite("masked", masked_img)
    cv2.waitKey(0)

mask_image("dataset/train/fake/FAKE_apahohbfek.mp4_299.png")

def extract_audio():
    project_path = get_face_from_video.get_path(os.path.dirname(__file__))


    # Get the filepaths and the metadata of it.
    file_paths, meta_paths = get_face_from_video.get_files_and_get_meta_file(project_path)

    # 
        

"""

Currently finding a way to extract audio based on frames (not sure if thats possible with purely/only moviepy)
May potentially have to extract the entire audio using moviepy library and then finding another method to split up the audio by frames.

The file_format is as follows:
{label.upper()}_{source_video}_{counter}
The counter is the position of the frame relative to the video.

For reference:

34 ms per frame


"""


def model_loss(input_real, input_z, output_channel_dim):
    g_model = generator(input_z, output_channel_dim, True)

    noisy_input_real = input_real + tf.random_normal(shape=tf.shape(input_real),
                                                     mean=0.0,
                                                     stddev=random.uniform(0.0, 0.1),
                                                     dtype=tf.float32)
    
    d_model_real, d_logits_real = discriminator(noisy_input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_model_real)*random.uniform(0.9, 1.0)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_model_fake)))
    d_loss = tf.reduce_mean(0.5 * (d_loss_real + d_loss_fake))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_model_fake)))
    return d_loss, g_loss

def model_optimizers(d_loss, g_loss):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]
    
    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=LR_D, beta1=BETA1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=LR_G, beta1=BETA1).minimize(g_loss, var_list=g_vars)  
    return d_train_opt, g_train_opt


def train(get_batches, data_shape, checkpoint_to_load=None):
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], NOISE_SIZE)
    d_loss, g_loss = model_loss(input_images, input_z, data_shape[3])
    d_opt, g_opt = model_optimizers(d_loss, g_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        iteration = 0
        d_losses = []
        g_losses = []
        
        for epoch in range(EPOCHS):        
            epoch += 1
            start_time = time.time()

            for batch_images in get_batches:
                iteration += 1
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, NOISE_SIZE))
                _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: LR_D})
                _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: LR_G})
                d_losses.append(d_loss.eval({input_z: batch_z, input_images: batch_images}))
                g_losses.append(g_loss.eval({input_z: batch_z}))

            summarize_epoch(epoch, time.time()-start_time, sess, d_losses, g_losses, input_z, data_shape)
            
            


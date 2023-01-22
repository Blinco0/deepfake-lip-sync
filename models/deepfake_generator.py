import numpy as np  # linear algebra
import cv2
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow import keras
import moviepy.editor as mp
from moviepy.editor import *

from keras import layers
from utils import get_face_from_video

WEIGHT_INIT_STDDEV = 0.02

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
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
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
    step_size = 34  # Presumably the time each frame occupies. So a frame can last like 1/30th of a second if the video is in 30FPS.

    shape_of_audio_np = (1471, 2)  # That's what scipy does?
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=2, kernel_size=7, groups=1, activation='relu', input_shape=shape_of_audio_np, name="input_audio"),
        tf.keras.layers.Conv2D(32, 5, (1, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, 5, 2, activation='relu'),
        tf.keras.layers.Conv2D(128, 5, 2, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 1, activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="audio")
    model.summary()
    return model


identity_model = identity_encoder()
audio_model = audio_encoder()

x = layers.Concatenate([identity_model.layers[-1].output, audio_model.layers[-1].output])
# I just copied it from the Simpson one. The parameters, to be exact. And I have no clue what the parameters are.
combined_output = layers.Conv2DTranspose(filters=3, kernel_size=[5, 5], strides=[1, 1], padding="SAME",
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV),
                                         name="logits")(x)

combined_model = keras.Model(inputs=[identity_model.layers[0].input, audio_model.layers[0].input],
                             outputs=[combined_output])
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
    mask = cv2.rectangle(mask, (0, 0), (255, 127), 255, -1)

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # display the mask, and the output image
    cv2.imshow('Masked Image', masked_img)

    # save the masked image.
    cv2.imwrite("masked", masked_img)
    cv2.waitKey(0)


mask_image("dataset/raw_videos/fake/FAKE_apahohbfek.mp4_299.png")


def extract_audio():
    """
    
    """
    project_path = get_face_from_video.get_path(os.path.dirname(__file__))
    ds_path = os.path.join(project_path, "dataset")
    raw_data_path = os.path.join(project_path, "raw_videos")
    print(project_path, ds_path, raw_data_path)

    # Get the filepaths and the metadata of it.
    file_paths, meta_paths = get_face_from_video.get_files_and_get_meta_file(raw_data_path)

    my_clip = VideoFileClip(r"C:\Users\Yxliu\OneDrive\Documents\dfdc_train_part_1\aassnaulhq.mp4")

    frame_time = 1 / my_clip.fps
    for i in range(0, int(my_clip.duration), frame_time):
        my_clip = my_clip.subclip(i, i + frame_time)
        audio = my_clip.audio
        audio.preview()

    #




def testing():
    seed = tf.random.normal([1, 100])
    generated_img = combined_model(seed, training=False)

    plt.imshow(generated_img[0, :, :, 0], cmap='gray')


testing()

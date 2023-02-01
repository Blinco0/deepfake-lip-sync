import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import tensorflow as tf
from tensorflow import keras
from keras import models
from moviepy.editor import *
from PIL import Image
from utils.audio_spectrogram import stft

from keras import layers
from utils import get_face_from_video

# In the future. Maybe try changing the input shape to basically use longer audio files. Right now it can only
# use the one with the duration of  1/30 second where 30 is the fps of the data used.
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


def identity_encoder():
    """A model for receiving the image as a numpy array."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(256, 256, 6), name="image_input"),
        tf.keras.layers.Conv2D(6, 7, 1, activation='relu'),
        tf.keras.layers.Conv2D(32, 5, (1, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, 5, 2, activation='relu'),
        tf.keras.layers.Conv2D(128, 5, 2, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 1, activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='sigmoid')
    ], name="image")
    model.summary()
    return model


def audio_encoder():
    """A model for receiving the audio as a numpy array."""
    loudness = 80  # Probably the decibel of the sound. Not sure how it's like that. tbf, I actually have no clue if the variable name is c
    # correct for this value.
    step_size = 34  # Presumably the time each frame occupies. So a frame can last like 1/30th of a second if the video is in 30FPS.

    shape_of_audio_np = (6, 513, 1)

    num_labels = 64
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    # norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))  May have to remove this.

    model = models.Sequential([
        layers.Input(shape=shape_of_audio_np, name="audio_input"),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ], name="audio")
    model.summary()
    return model


def generator(training=True):
    identity_model = identity_encoder()
    audio_model = audio_encoder()

    # x = layers.Concatenate([identity_model.layers[-1].output, audio_model.layers[-1].output])
    x = tf.concat([identity_model.output, audio_model.output], axis=-1)
    # I just copied it from the Simpson one. The parameters, to be exact. And I have no clue what the parameters are.
    # combined_output = layers.Conv2DTranspose(filters=3, kernel_size=[5, 5], strides=[1, 1], padding="SAME",
    #                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV),
    #                                          name="logits")(x)
    combined_output = layers.Dense(128, use_bias=False)(x)
    combined_output = layers.Dense(256, use_bias=False)(x)
    combined_output = layers.Dense(768, use_bias=False)(x)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Reshape((4, 4, 48))(combined_output)
    # assert tf.shape(combined_output) == (None, 4, 4, 48)  # Note: None is the batch size

    combined_output = layers.Conv2DTranspose(24, (5, 5), strides=(2, 2), padding='same', use_bias=False)(
        combined_output)
    # assert combined_output.output_shape == (None, 8, 8, 24)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Conv2DTranspose(12, (5, 5), strides=(8, 8), padding='same', use_bias=False)(combined_output)
    # assert tf.shape(combined_output) == (None, 64, 64, 12)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Conv2DTranspose(3, (5, 5), strides=(4, 4), padding='same', use_bias=False,
                                             activation='tanh')(combined_output)
    # assert tf.shape(combined_output) == (None, 256, 256, 3)

    combined_model = keras.Model(inputs=[identity_model.input, audio_model.input],
                                 outputs=[combined_output], name="combined_model")
    combined_model.summary()

    combined_model.compile(
        optimizer='adam',
        loss='mae'
    )
    return combined_model


# mask_image("/home/hnguyen/PycharmProjects/deepfake-lip-sync/dataset/train/fake/FAKE_aktnlyqpah.mp4_111.png")


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


def test_generate():
    print("testing")
    img = Image.open("/home/hnguyen/PycharmProjects/deepfake-lip-sync/dataset/train/fake/FAKE_aahsnkchkz_125.png")
    seed_1 = np.asarray(img)
    filepath = f"/home/hnguyen/PycharmProjects/deepfake-lip-sync/utils/audio/vmigrsncac_audio_132.wav"
    samplerate, samples = wav.read(filepath)

    seed_2 = stft(samples, 2 ** 10)
    seed_2 = np.reshape(seed_2, newshape=(6, 513, 1))
    combined_inputs = {"image_input": np.expand_dims(seed_1, axis=0),
                       "audio_input": np.expand_dims(seed_2, axis=0)}
    print(combined_inputs["image_input"].shape)
    print(combined_inputs["audio_input"].shape)

    combined = generator()
    generated_img = combined(combined_inputs, training=False)
    print(generated_img.shape)
    # print(generated_img[0].shape)
    print("done.")
    plt.imshow(generated_img[0])


if __name__ == "__main__":
    test_generate()

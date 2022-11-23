import data_loader
import numpy as np   # linear algebra
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path


import tensorflow as tf

# All lists below are supposed to be lists of numpy arrays.
x_train = []  # train set
y_train = []  # train label

x_test = []   # test set
y_test = []   # test label

# Ratio between train and test
train_ratio = 3
test_ratio = 2


def append_train(png_path: str):
    img = cv2.imread(png_path)
    file_name = png_path.split('/')[-1]
    separated = file_name.split('_')
    label = separated[0]
    img_np = np.asarray(img)
    x_train.append(img_np)
    # Categorical Data time.
    if label == "FAKE":
        y_train.append(0)
    elif label == "REAL":
        y_train.append(1)


def append_test(png_path):
    img = cv2.imread(png_path)
    file_name = png_path.split('/')[-1]
    separated = file_name.split('_')
    label = separated[0]
    img_np = np.asarray(img)
    x_test.append(img_np)
    # Categorical Data time.
    if label == "FAKE":
        y_test.append(0)
    elif label == "REAL":
        y_test.append(1)

#
# Helper function to show a list of images with their relating titles
#


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()

#
# Show some random training and test images
#


def choose_imgs_and_plot():
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, len(x_train))
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, len(x_test))
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))
    show_images(images_2_show, titles_2_show)


# save_pngs = data_loader.main()
def main():
    path_list = Path("train_and_test_set").rglob(pattern="*.png")
    for path in path_list:
        roll = random.randint(1, train_ratio + test_ratio)  # inclusive [a, b] for randint
        if roll <= test_ratio:
            append_test(str(path))
        else:
            append_train(str(path))


main()
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
print(f"x_train: {len(x_train)}, x_test {len(x_test)}")
print(f"y_train: {len(y_train)}, y_test {len(y_test)}")

# Imported the ones from MNIST database lol. Uhhh... why does it have such a high score...? Overfitting?
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, 3, 3, activation='relu', input_shape=(320, 320, 3)),
  tf.keras.layers.Conv2D(1000, 3, 3, activation='relu'),
  tf.keras.layers.MaxPool2D(3, 3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1500, activation='relu'),
  tf.keras.layers.Dense(800, activation='relu'),
  tf.keras.layers.Dense(400, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax', name='output')
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=2)
model.evaluate(x_test, y_test)

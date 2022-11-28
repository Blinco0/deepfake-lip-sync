# import utils.get_face_from_video as get_face_from_video
import numpy as np   # linear algebra
import cv2
import matplotlib.pyplot as plt
import random
# from pathlib import Path
import os
import tensorflow as tf

# All lists below are supposed to be lists of numpy arrays.
x_train = []  # train set
y_train = []  # train label

x_test = []   # test set
y_test = []   # test label

# Ratio between train and test
# train_ratio = 3
# test_ratio = 2

def append_train(img_path: str, label: str):
    """
    Add image and label to train lists
    Parameters:
        img_path: the path to the image
        label: the label of the image
    """
    img = cv2.imread(img_path)
    # img_np = np.asarray(img)
    x_train.append(img)
    # Categorical Data time.
    if label == "FAKE":
        y_train.append(0)
    elif label == "REAL":
        y_train.append(1)


def append_test(img_path, label: str):
    """
    Add image and label to test lists
    Parameters:
        img_path: the path to the image
        label: the label of the image
    """
    img = cv2.imread(img_path)
    # img_np = np.asarray(img)
    x_test.append(img)
    # Categorical Data time.
    if label == "FAKE":
        y_test.append(0)
    elif label == "REAL":
        y_test.append(1)


def load_file_from_split_dataset(dataset_path: str):
    """
    Given a path with the structure
    dataset
    |---test
    |   |---real
    |   |---fake
    |---train
    |   |---real
    |   |---fake
    Load data into its respective python list
    Assumes a UNIX-based file system
    """
    train_path = dataset_path+"/train"
    test_path = dataset_path+"/test"
    counter=0
    
    # Load train data
    for img in os.listdir(train_path+"/real"):
        counter+=1
        if counter == 5000:
            counter = 0
            break
        append_train(train_path+"/real/" + img, "REAL")

    counter = 0
    for img in os.listdir(train_path+"/fake"):
        counter+=1
        if counter == 5000:
            counter = 0
            break
        append_train(train_path+"/fake/" + img, "FAKE")

    counter = 0
    # Load test data
    for img in os.listdir(test_path+"/real"):
        counter+=1
        if counter == 1000:
            counter = 0
            break
        append_test(test_path+"/real/" + img, "REAL")

    for img in os.listdir(test_path+"/fake"):
        counter+=1
        if counter == 1000:
            counter = 0
            break
        append_test(test_path+"/fake/" + img, "FAKE")

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
    """
    Khoa's new stuff
    """
    load_file_from_split_dataset("dataset")
    # path_list = Path("train_and_test_set").rglob(pattern="*.png")
    # for path in path_list:
    #     roll = random.randint(1, train_ratio + test_ratio)  # inclusive [a, b] for randint
    #     if roll <= test_ratio:
    #         append_test(str(path))
    #     else:
    #         append_train(str(path))
    

main()

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x_train, y_train = unison_shuffled_copies(x_train, y_train)

print(f"x_train: {x_train.shape}, x_test {x_test.shape}")
print(f"y_train: {y_train.shape}, y_test {y_test.shape}")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Imported the ones from MNIST database lol. Uhhh... why does it have such a high score...? Overfitting?
# Looks like 256x256 is the standard
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
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mean_absolute_error'])
model.summary()

model.fit(x_train, y_train, epochs=7)
model.save('saved_models/khoa') # TODO: get dotenv working and make this an env variable
print("Evaluating model")
model.evaluate(x_test, y_test)

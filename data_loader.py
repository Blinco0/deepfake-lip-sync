import numpy as np
import cv2
import os
import re
import json

# Uhh, some of the videos are uncanny as hell... It's like watching a real Mandela Catalogue
# train will be a 5D numpy array. Number of videos - number of frames - row - col - rgb
# Do we need timestamp?
# AND, do we need to load all videos in a single arrays like the 5D one.
# or do we just have to have each video as its own separate numpy array, because I'm afraid that all videos might
# not have the same number of frames.

# We still have to crop out everything each frame except for the mouth...
# Also, I think that we might have to use the freeworld VAAPI video codecs.
train = []
sorted_keys = []
mp4_files = []


# Remember to use regex to grab the metadata and the json...
# Ask Khoa on if we need to separate each video batch into their own directory...
def get_files_and_get_meta_file(directory):
    files = []
    meta_files = []
    vid_pattern = r".*\.mp4"
    metafile_pattern = r".*\.json"
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if re.match(pattern=vid_pattern, string=filename):
                # Join the filename and directory to get a complete relative filepath
                filepath = os.path.join(directory, filename)
                sorted_keys.append(filename)
                files.append(filepath)
                print(filepath)
            elif re.match(pattern=metafile_pattern, string=filename):
                metafile_path = os.path.join(directory, filename)
                meta_files.append(metafile_path)
    return files, meta_files


# Get the labels from the metadata.json. Real is 1, Fake is 0. Categorical Data into Integers.
def get_labels(metafile_path):
    # Uh oh... We need to line up the labels key entry with the directory one... DONE with the use of sorted keys
    # labels_np will be a 1D numpy array after casting the returning labels[] as a nparray
    labels = []
    with open(metafile_path) as f:
        labels_dict = json.load(f)
        for mp4_file in sorted_keys:
            meta_data = labels_dict[mp4_file]
            label = meta_data["label"]
            if label.upper() == "FAKE":
                labels.append(0)
            elif label.upper() == "REAL":
                labels.append(1)
    return labels


def capture_video(vid_dest):
    cap = cv2.VideoCapture(vid_dest)

    # Check if the video can be turned into a stream successfully.
    # If not, probably check to make sure the destination is correct.
    frames = []
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        exit(1)

    # Do we have to store every single frame on it?  Yes I think. Ask Khoa later today.
    while cap.isOpened():
        # Get the boolean if frame is found and the frame itself
        ret, frame = cap.read()
        # Check to see if frame is found. Otherwise, the video is considered to have gone through all frames.
        # Nice that frame is also a matrix
        if ret is True:
            frame_np = np.asarray(frame)
            cv2.imshow("Frame", frame)
            frames.append(frame_np)
            # Wait for 25 miliseconds
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    frames = np.asarray(frames)
    # Hmm, do I need to add frames to the train one? What did I mean by that???
    print(frames.shape)


if __name__ == "__main__":
    directory = "train"
    mp4_files, metafile_path = get_files_and_get_meta_file(directory)
    metafile_path = metafile_path[0]  # There should be only one metafile in the training set/
    labels_list = get_labels(metafile_path)
    print(len(labels_list))
    for mp4_file in mp4_files:
        capture_video(mp4_file)

    # Yeah... There is a metadata.json here... I'm too lazy to get it out, so... Imma use regex...
    train_np = np.asarray(train)
    labels_np = np.asarray(labels_list)

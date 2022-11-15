import numpy as np
import cv2
import glob, os
import re
import json

# label will be a 1D numpy array.
labels = []
# train will be a 5D numpy array. Number of videos - number of frames - row - col - rgb
# Do we need timestamp?
# AND, do we need to load all videos in a single arrays like the 5D one.
# or do we just have to have each video as its own separate numpy array, because I'm afraid that all videos might
# not have the same number of frames.
train = []

mp4_files = []


# Remember to use regex to grab the metadata...
# Ask Khoa on if we need to separate each video batch into their own directory...
def get_files(directory):
    files = []
    return files


# Get the labels from the metadata.json. Real is 1, Fake is 0. Categorical Data into Integers.
def get_labels(directory):
    # Uh oh... We need to line up the labels key entry with the directory one...
    with open(directory) as f:
        labels_dict = json.load(f)
    pass


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
    # Hmm, do I need to add frames to the train one?
    print(frames.shape)


mp4_files = get_files("train")
for mp4_file in mp4_files:
    capture_video(mp4_file)
# capture_video("train/aabdnomlru.mp4")

# Yeah... There is a metadata.json here... I'm too lazy to get it out, so... Imma use regex...
train_np = np.asarray(train)
labels = np.asarray(labels)

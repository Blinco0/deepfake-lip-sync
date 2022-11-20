import numpy as np
import cv2
import os
import re
import json
import time

# Uhh, some of the videos are uncanny as hell... It's like watching a real Mandela Catalogue
# Train is a list containing 3D numpy arrays for the deepfake discriminator:
# x coord - y coord - rgb values

# TODO:
#       Maybe refactor the code for getting the label. Don't want to open the json file for every single frame....


train = []
labels = []
sorted_keys = []
mp4_files = []
front_face_detector = cv2.CascadeClassifier("cascade-files/haarcascade_frontalface_alt2.xml")
RESIZE_SIZE = (320, 320)  # Resize size for the cropped face

# For profile picture detection (including side faces... We might need it later)...
# profile_face_detector = cv2.CascadeClassifier("cascade-files/haarcascade_profileface.xml")

if front_face_detector.empty():
    print("Unable to open the haarcascade mouth detection xml file...")
    exit(1)


def get_files_and_get_meta_file(directory):
    """
    Get the file paths for every video of .mp4 format as well as the file path of the metadata.json file.
    Assume that the metadata file is in the same directory as those mp4 videos
    :param directory: the directory where all the mp4 files and the metadata.json are
    :return: None
    """
    file_paths = []
    meta_files = []
    if os.name == "nt":
        # For Windows
        vid_pattern = r"^.*\\\\.*\.mp4$"
        metafile_pattern = r"^.*\\\\.*\.json$"
    else:
        # For Linux
        vid_pattern = r"^.*\.mp4$"
        metafile_pattern = r"^.*\.json$"

    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if re.match(pattern=vid_pattern, string=filename):
                # Join the filename and directory to get a complete relative filepath
                filepath = os.path.join(directory, filename)
                sorted_keys.append(filename)
                file_paths.append(filepath)
                print(filepath)
            elif re.match(pattern=metafile_pattern, string=filename):
                metafile_path = os.path.join(directory, filename)
                meta_files.append(metafile_path)
    return file_paths, meta_files


def get_meta_dict(metafile_path):
    with open(metafile_path) as f:
        meta_dict = json.load(f)
    return meta_dict


def capture_video(vid_dest, meta_dict):
    """
    Go through the video using its path and process every frames in that video
    :param meta_dict: the dictionary form of the metadata.json
    :param vid_dest: the video's file path
    :param meta_dict: the dictionary form of the metadata.json
    :return: None
    """
    cap = cv2.VideoCapture(vid_dest)
    if os.name == "nt":
        # For Windows
        pattern = r"^.*\\\\(.*\.mp4)$"
    else:
        # For Linux
        pattern = r"^.*/(.*\.mp4)$"
    source_video = re.match(pattern=pattern, string=vid_dest)[1]
    label = meta_dict[source_video]["label"]

    # Check if the video can be turned into a stream successfully.
    # If not, probably check to make sure the destination is correct.
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        exit(2)

    while cap.isOpened():
        # Get the boolean if frame is found afid the frame itself
        ret, frame = cap.read()
        # Check to see if frame is found. Otherwise, the video is considered to have gone through all frames.
        # Nice that frame is also a matrix
        if ret is True:
            detect_face_and_add_labels(frame, label=label)
            # Wait for 25 miliseconds
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


def detect_face_and_add_labels(frame, label):
    """
    Crop and resize only the frontal face detection the pretrained model uses.
    Will also label the frame as either 0 (FAKE) or 1 (REAL) according to the metadata file.
    :param frame: a frame of the video.
    :param label: the label FAKE or REAL of the video the frame belongs to
    :return: None
    """
    # Apparently, this colorspace is damn good for computer vision stuff. YCrBr that is. But it's not working so
    # a different colorspace is needed.
    # frame_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = front_face_detector.detectMultiScale(frame_bgr, minNeighbors=6,
                                                 minSize=(125, 125), scaleFactor=1.15)
    # profile_faces = profile_face_detector.detectMultiScale(frame_bgr, minNeighbors=6,
    # minSize=(150, 150), maxSize=(500, 500), scaleFactor=1.1)
    for (x, y, w, h) in faces:
        frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x, y), pt2=(x + w, y + h),
                                  color=(0, 255, 0), thickness=2)
        cropped = frame_bgr[y:y + h, x:x + w]
        cropped = cv2.resize(cropped, RESIZE_SIZE)
        cropped_np = np.asarray(cropped)
        # Only use the cropped and resized frame to append to the list of frames.
        train.append(cropped_np)
        # Add in the labels to each frame in accordance with the frame's source video label in the metadata.json
        get_labels_for_frame(label=label)
        # for (x, y, w, h) in profile_faces:
        # frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x, y), pt2=(x + w, y + h),
        # color=(0, 0, 255), thickness=2)
        cv2.imshow("Facial detection cropped", cropped)


def get_labels_for_frame(label):
    """
    Get the label for each cropped frame using the source video's own label in the metafile_path parameter
    :param label: the label FAKE or REAL of the video the frame belongs to
    :return None
    """
    if label.upper() == "FAKE":
        labels.append(0)
    elif label.upper() == "REAL":
        labels.append(1)


def main():
    directory = "/home/hnguyen/PycharmProjects/deepfake-lip-sync/train"
    mp4_file_paths, metafile_path = get_files_and_get_meta_file(directory)
    metafile_path = metafile_path[0]  # There should be only one metafile in the training set
    meta_dictionary = get_meta_dict(metafile_path)
    for i in range(len(mp4_file_paths)):
        start_time = time.time()
        capture_video(mp4_file_paths[i], meta_dictionary)
        print(f"----- Video {mp4_file_paths[i]} done. {i + 1} out of {len(mp4_file_paths)}"
              f"{time.time() - start_time} seconds -----")

    # train and labels don't have to be turned into numpy arrays.
    # train_np = np.asarray(train)
    # labels_test = np.asarray(labels)
    #
    train_set = train[:(len(train) // 2)]
    train_label_set = labels[:(len(labels) // 2)]
    test_set = train[(len(train) // 2):]
    test_label_set = labels[(len(labels) // 2):]
    return train_set, train_label_set, test_set, test_label_set

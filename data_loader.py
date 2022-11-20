import numpy as np
import cv2
import os
import re
import json

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
    vid_pattern = r".*\.mp4"
    metafile_pattern = r".*\.json"
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


def get_labels_for_frame(metafile_path, source_video_path):
    """
    Get the label for each cropped frame using the source video's own label in the metafile_path parameter
    :param metafile_path: the path to the meta file containing the labels for all videos in the training set.
    :param source_video_path: path of the source video that the frame belongs to.
    :return:None
    """
    pattern = r"^.*/(.*\.mp4)$"
    with open(metafile_path) as f:
        labels_dict = json.load(f)
        source_video = re.match(pattern=pattern, string=source_video_path)[1]
        meta_data = labels_dict[source_video]
        label = meta_data["label"]
        if label.upper() == "FAKE":
            labels.append(0)
        elif label.upper() == "REAL":
            labels.append(1)


def capture_video(vid_dest, metafile_path):
    """
    Go through the video using its path and process every frames in that video
    :param vid_dest: the video's file path
    :param metafile_path: the metadata's file path
    :return: None
    """
    cap = cv2.VideoCapture(vid_dest)

    # Check if the video can be turned into a stream successfully.
    # If not, probably check to make sure the destination is correct.
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        exit(2)

    while cap.isOpened():
        # Get the boolean if frame is found and the frame itself
        ret, frame = cap.read()
        # Check to see if frame is found. Otherwise, the video is considered to have gone through all frames.
        # Nice that frame is also a matrix
        if ret is True:
            detect_face_and_add_labels(frame, vid_dest, metafile_path)
            # Wait for 25 miliseconds
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


def detect_face_and_add_labels(frame, vid_dest, metafile_path):
    """
    Crop and resize only the frontal face detection the pretrained model uses.
    Will also label the frame as either 0 (FAKE) or 1 (REAL) according to the metadata file.
    :param frame: a frame of the video.
    :param vid_dest: the filepath to the video
    :param metafile_path: the path of the metafile
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
        cropped = frame_bgr[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, RESIZE_SIZE)
        cropped_np = np.asarray(cropped)
        # Only use the cropped and resized frame to append to the list of frames.
        train.append(cropped_np)
        # Add in the labels to each frame in accordance with the frame's source video label in the metadata.json
        get_labels_for_frame(metafile_path=metafile_path, source_video_path=vid_dest)
    # for (x, y, w, h) in profile_faces:
        # frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x, y), pt2=(x + w, y + h),
                                  # color=(0, 0, 255), thickness=2)
        cv2.imshow("Facial detection cropped", cropped)


def main():
    directory = "train"
    mp4_file_paths, metafile_path = get_files_and_get_meta_file(directory)
    metafile_path = metafile_path[0]  # There should be only one metafile in the training set
    for mp4_file in mp4_file_paths:
        capture_video(mp4_file, metafile_path)

    # train and labels don't have to be turned into numpy arrays.
    # train_np = np.asarray(train)
    # labels_test = np.asarray(labels)
    #
    train_set = train[:(len(train) // 2)]
    train_label_set = labels[:(len(labels) // 2)]
    test_set = train[(len(train) // 2):]
    test_label_set = labels[(len(labels) // 2):]
    return train_set, train_label_set, test_set, test_label_set



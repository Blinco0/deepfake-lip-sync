import json
import random
import re
import time

import cv2
from moviepy.editor import *


# Uhh, some videos are uncanny as hell... It's like watching a real Mandela Catalogue
# Train is a list containing 3D numpy arrays for the deepfake discriminator:
# x coord - y coord - rgb values


def detect_if_structure_exists(dataset_path):
    """
    Checek if the file structure exists and if not, make the folder appropriately
    :return: false
    """
    first_levels = [dataset_path]
    second_levels = ["raw_videos", "test", "valid"]
    third_levels = ["real", "fake"]

    for level in first_levels:
        for second_level in second_levels:
            for third_level in third_levels:
                path = os.path.join(level, second_level, third_level)
                if os.path.exists(path) is False:
                    os.makedirs(path, exist_ok=True)


def get_path(script_path):
    """
    Get the path to the current script. Crop it out to get the path of the project.
    Assume that the project name is deepfake-lip-sync.
    :return the absolute path of the project.
    """
    if os.name == "nt":
        pattern = r"^(.*\\\\deepfake-lip-sync).*"
    else:
        pattern = r"(.*/deepfake-lip-sync).*"
    match = re.match(pattern=pattern, string=script_path)
    return match[1]


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
                # print(filepath)  # Optional. Uncomment if you want to check if the output is printed correctly.
            elif re.match(pattern=metafile_pattern, string=filename):
                metafile_path = os.path.join(directory, filename)
                meta_files.append(metafile_path)
    return file_paths, meta_files


def get_meta_dict(metafile_path):
    with open(metafile_path) as f:
        meta_dict = json.load(f)
    return meta_dict


def capture_video(dataset_path: str, vid_dest: str, meta_dict: dict):
    """
    Go through the video using its path and process every frame in that video.
    May also extract the audio piece too.
    :param dataset_path: the path to the dataset
    :param meta_dict: the dictionary form of the metadata.json
    :param vid_dest: the video's file path
    :param meta_dict: the dictionary form of the metadata.json
    :return: None
    """
    cap = cv2.VideoCapture(vid_dest)
    # Start the audio, too.
    audio = VideoFileClip(vid_dest).audio

    frame_time = 1 / cap.get(cv2.CAP_PROP_FPS)
    counter = 0
    if os.name == "nt":
        # For Windows
        pattern = r"^.*\\\\(.*)\.mp4$"
    else:
        # For Linux
        pattern = r"^.*/(.*)\.mp4$"
    source_video = re.match(pattern=pattern, string=vid_dest)[1]
    label = meta_dict[f"{source_video}.mp4"]["label"]

    # Check if the video can be turned into a stream successfully.
    # If not, probably check to make sure the destination is correct.
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        exit(2)

    while cap.isOpened():
        # Get the boolean ret if face is found from the frame itself.
        # ret is for saying if a frame can be extracted out of the video.
        ret, frame = cap.read()
        # audio_frame, val = player.get_frame()
        # Check to see if frame is found. Otherwise, the video is considered to have gone through all frames.
        # Nice that frame is also a matrix.
        if ret is True:
            detect_face_add_labels_get_audio(frame=frame, audio=audio, source_video_name=source_video, counter=counter,
                                             frame_time=frame_time, dataset_path=dataset_path, label=label, )
            # Wait for 25 miliseconds
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        counter += 1
    # print(counter)  # Guaranteed 300. Will delete it later.


def detect_face_add_labels_get_audio(frame, audio, source_video_name: str,
                                     counter: int, frame_time, dataset_path, label):
    """
    Crop and resize only the frontal face detection the pretrained model uses.
    Will also label the frame as either 0 (FAKE) or 1 (REAL) according to the metadata file.
    Will also extract the original audio related to the frame to the sync video.
    :param dataset_path: the path to the dataset
    :param frame: a frame of the video.
    :param label: the label FAKE or REAL of the video the frame belongs to
    :param source_video_name: the name of the video the frame belongs to
    :param counter: the index of the frame in the video, very useful for extracting audio out of.
    :param audio: the audio part of the source video extracted from the video
    :param frame_time: The number of time of a frame. Got by dividing 1 / fps
    :return: None
    """
    # Apparently, this colorspace is damn good for computer vision stuff. YCrBr that is. But it's not working so
    # a different colorspace is needed. Scratch that, we need to use the RGB one.
    # frame_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = front_face_detector.detectMultiScale(frame_bgr, minNeighbors=6,
                                                 minSize=(125, 125), scaleFactor=1.15)
    # profile_faces = profile_face_detector.detectMultiScale(frame_bgr, minNeighbors=6,
    # minSize=(150, 150), maxSize=(500, 500), scaleFactor=1.1)

    # Get the audio subclip and save it!
    start = counter * frame_time
    frame_audio = audio.subclip(start, start + frame_time)

    for (x, y, w, h) in faces:
        # Cropping only the faces. Please read documentation for it....
        frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x - 1, y - 1), pt2=(x + w, y + h),
                                  color=(0, 255, 0), thickness=1)
        cropped = frame_bgr[y:y + h, x:x + w]
        cropped = cv2.resize(cropped, RESIZE_SIZE)
        # <REAL or FAKE>_<source_video>_<frame_index>
        # Roll to see if it goes to the test or the raw_videos
        roll = random.randint(1, train_ratio + test_ratio + valid_ratio)  # inclusive [a, b] for randint
        if roll <= test_ratio:
            dest = f"test"
        elif roll <= train_ratio + test_ratio:
            dest = f"raw_videos"
            # what about valid?
        else:
            dest = f"valid"
        path = os.path.join(dataset_path, dest, label.lower(), f"{label.upper()}_{source_video_name}_{counter}.png")
        audio_path = os.path.join(dataset_path, dest, label.lower(),
                                  f"{label.upper()}_{source_video_name}_{counter}.wav")

        cv2.imwrite(path, cropped)
        frame_audio.write_audiofile(filename=audio_path, codec="pcm_s16le", verbose=False, logger=None)

        # for (x, y, w, h) in profile_faces:
        # frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x, y), pt2=(x + w, y + h),
        # color=(0, 0, 255), thickness=2)
        # cv2.imshow("Facial detection cropped", cropped)  # imshow is the bottleneck...


def main():
    project_path = get_path(os.path.dirname(__file__))
    ds_path = os.path.join(project_path, "dataset")
    raw_data_path = os.path.join(project_path, "raw_videos")

    detect_if_structure_exists(ds_path)
    mp4_file_paths, metafile_path = get_files_and_get_meta_file(raw_data_path)
    metafile_path = metafile_path[0]  # There should be only one metafile in the training set
    meta_dictionary = get_meta_dict(metafile_path)
    num_loops = 40
    for i in range(num_loops):
        start_time = time.time()
        capture_video(ds_path, mp4_file_paths[i], meta_dictionary)
        print(
            f"----- Video {mp4_file_paths[i]} done. {i + 1} out of {num_loops} out of a total of {len(mp4_file_paths)}"
            f". {time.time() - start_time} seconds -----")
    return


if __name__ == "__main__":
    train = []
    labels = []
    sorted_keys = []
    mp4_files = []

    front_face_detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml'))
    RESIZE_SIZE = (256, 256)  # Resize size for the cropped face

    # For profile picture detection (including side faces... We might need it later)...
    # profile_face_detector = cv2.CascadeClassifier("cascade-files/haarcascade_profileface.xml")
    if front_face_detector.empty():
        print("Unable to open the haarcascade mouth detection xml file...")
        exit(1)

    # Ratio between raw_videos and test
    train_ratio = 3
    test_ratio = 2
    valid_ratio = 1

    main()

import os
import glob
import re


def choosing_data_for_batch(batch_num: int, data_path: str):
    """
    Go through the data_path folder containing images and audios extracted from videos, and
    generate a JSON File with each entry being:
        - key: batch_index
        - val: The list of of tuples (image_name_for_generating, image_name_for_reference)
    :param batch_num: Number of batches you want to generate for the JSON file
    :param data_path: The path containing images and audios
    :return: None
    """
    if not os.path.exists(data_path):
        raise FileExistsError(f"{data_path} does not exist")
    list_of_fps = []
    list_of_fps_choose = []
    if os.name == "nt":
        # For Windows
        img_pattern = r"^(.*\\\\.*)\.png$"
    else:
        # For Linux
        img_pattern = r"^(.*)\.png$"
    # Add name of frames to
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if match := re.match(pattern = img_pattern, string=filename):
                list_of_fps

    for batch_index in batch_num:
        pass

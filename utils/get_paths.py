import os
import re


def get_path(script_path):
    """
    Get the path to the current script. Crop it out to get the path of the project.
    Assume that the project name is deepfake-lip-sync.
    :return the absolute path of the project.
    """
    if os.name == "nt":
        pattern = r"^(.*\\deepfake-lip-sync).*"
    else:
        pattern = r"(.*/deepfake-lip-sync).*"
    match = re.match(pattern=pattern, string=script_path)
    return match.group(1)



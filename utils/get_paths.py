import os
import re
#TODO: Change get_path to get_project_path or sth.


def get_path(script_path=os.path.dirname(__file__)):
    """
    Get the path to the current script. Crop it out to get the path of the project. Default
    argument is the path to this file.
    Assume that the project name is deepfake-lip-sync.
    :return the absolute path of the project.
    """
    if os.name == "nt":
        pattern = r"^(.*\\deepfake-lip-sync).*"
    else:
        pattern = r"(.*/deepfake-lip-sync).*"
    match = re.match(pattern=pattern, string=script_path)
    return match.group(1)






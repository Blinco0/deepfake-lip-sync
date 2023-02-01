from moviepy.editor import *
from utils.get_face_from_video import get_path, get_files_and_get_meta_file
from scipy.io import wavfile
import numpy as np
# Compilation of changes
# Change directory name to raw_videos
# Audio extraction into numpy array is possible! Now we have to implement it in our data preprocessing file!!!!


def extract_audio():
    """

    """
    project_path = get_path(os.path.dirname(__file__))
    raw_data_path = os.path.join(project_path, "raw_videos")
    # Get the filepaths and the metadata of it.
    file_paths, meta_paths = get_files_and_get_meta_file(raw_data_path)
    #testing_path = rd.choice(file_paths)
    testing_path = "/home/hnguyen/PycharmProjects/deepfake-lip-sync/raw_videos/vmigrsncac.mp4"
    print(testing_path)
    my_clip = VideoFileClip(testing_path)
    audio = my_clip.audio

    # Testing the write_audiofile
    audio.write_audiofile(filename=
                                f"sdasdsad.wav",
                                codec="pcm_s16le")
    # Count the number of frames. Assume 30 FPS, but with -1 for some reason. So 0.033 is a go.
    frame_time = 1 / my_clip.fps
    # Also test if there is actually 300 frames in the vid and not 299
    print(my_clip.fps, my_clip.duration)
    for i in range(0, round(my_clip.fps * my_clip.duration)):
        start = i * frame_time
        print(f"vmigrsncac_audio_{i+1}.wav {start} - {start + frame_time}") # Okay, so frame does line up with its duration!!!
        frame_audio = audio.subclip(start, start + frame_time)
        frame_audio.write_audiofile(filename=
                                    f"vmigrsncac_audio_{i+1}.wav",
                                    codec="pcm_s16le")
        # filename can either be fullpath or relative path. file name must include the extension too.
        # codec have to have no leading or trailing whitespace. Bruh.

    # So, data extraction:
        # Also to extract the audio for every video though.
        # When looping through the single loop that uses opencv, do similar thing as the loop above, particualrly
        # in the subclip part.
        # Save it to same folder as the frame in the wav format.
        # Use scipy to grab that file and read it.


# Just save to scipy and do it .
def turn_mp3_to_numpy(file_path):
    output = wavfile.read(file_path)
    numpy_arr = np.array(output[1]) # 0 for the sampling rate and 1 for the actual array
    return numpy_arr


if __name__ == "__main__":
    extract_audio()
    example_audio_numpy = turn_mp3_to_numpy("audio/vmigrsncac_audio_237.wav")
    print(example_audio_numpy)
    # for i in range(300):
    #     example_audio_numpy = turn_mp3_to_numpy(f"vmigrsncac_audio_{i+1}.wav")
    #     print(example_audio_numpy.shape)
    ex_vid_audio = turn_mp3_to_numpy("sdasdsad.wav")
    print(ex_vid_audio.shape)


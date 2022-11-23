# deepfake-lip-sync
A Generative Adversarial Network that deepfakes a person's lip to a given audio source

## File structure
Make sure you process the dataset to get this file structure:\
deepfake-lip-sync\
|---dataset\
|&emsp;&nbsp;|---test\
|&emsp;&nbsp;|&emsp;&nbsp;|---real\
|&emsp;&nbsp;|&emsp;&nbsp;|---fake\
|&emsp;&nbsp;|---train\
|&emsp;&nbsp;|&emsp;&nbsp;|---real\
|&emsp;&nbsp;|&emsp;&nbsp;|---fake\
|&emsp;&nbsp;|---valid\
|&emsp;&nbsp;|&emsp;&nbsp;|---real\
|&emsp;&nbsp;|&emsp;&nbsp;|---fake\
Within each of the test, train and validation folders, place the 320x320 images into their respective folders.

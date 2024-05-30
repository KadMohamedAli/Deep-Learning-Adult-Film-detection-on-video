# Deep-Learning-Pornography-detection-on-video
A deep learning approch for porn detection on videos using image and sound.

Models available at : https://drive.google.com/drive/folders/1CEwlOINEgbiC2PGFgF5fpSExXv9BONcw?usp=sharing

Source have the files to create, train and evaluate the model, also all the preprocessing steps.

predict.ipynb : let you test the model on a video file, or a directory of files.

Models was trained on LSPD images and images and sound extracted from videos , and tested on LSPD and NPDI-2K and a custom dataset composed of porn video from LSPD and NPDI-2k and videos from local TV.

99% accuracy score on LSPD (221 videos).
95% accuracy score on NPDI-2k (773 videos)
99% accuracy score on custom dataset (352 videos)

Tested videos was extracted from the correspending dataset and split to 1 minutes videos, we also deleted manualy uncorreclty labelled videos for the testing.

Results are available on /source/VideoClassification_both.ipynb


## Tested Versions

This script was tested with the following versions of libraries:

- OpenCV version: 4.9.0
- NumPy version: 1.26.4
- SciPy version: 1.13.1
- scikit-learn version: 1.5.0
- pydub version: Not available
- moviepy version: 2.0.0.dev2
- IPython version: 8.24.0
- Pillow (PIL) version: 10.3.0
- TensorFlow version: 2.10.0
- Python version: 3.10.14 | packaged by Anaconda, Inc. | (main, Mar 21 2024, 16:20:14) [MSC v.1916 64 bit (AMD64)]
- Anaconda is not installed or not found in PATH

## Installation

Ensure you have the necessary libraries installed. Use the following commands to install them:

```bash
pip install opencv-python-headless==4.9.0
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install scikit-learn==1.5.0
pip install pydub  # No version specified
pip install moviepy==2.0.0.dev2
pip install ipython==8.24.0
pip install Pillow==10.3.0
pip install tensorflow==2.10.0

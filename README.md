# Deep Learning Approach for Porn Detection on Videos Using Image and Sound

This project implements a deep learning approach for detecting pornographic content in videos by analyzing both images and sounds.

## Overview

A Jupyter notebook is provided to test the models on individual video files or directories of files. Additionally, this repository contains all the necessary files to preprocess data, train models, and evaluate their performance.(not organized so not really usable for the moment)
### Available Models

The trained models are available for download at the following link:
[Download Models](https://drive.google.com/drive/folders/1CEwlOINEgbiC2PGFgF5fpSExXv9BONcw?usp=sharing)

### Key Files

- **predict.ipynb**: A notebook to test the model on a video file or a directory of video files.
- **/source/VideoClassification_both.ipynb**: Contains the evaluation results of the models.

### Training and Testing

The models were trained using the LSPD dataset and videos with extracted images and sound. Testing was conducted on the LSPD [1], NPDI-2K [2], and a custom dataset comprising pornographic videos from both LSPD and NPDI-2K and videos from local TV.

#### Performance

- **LSPD**: 99% accuracy (221 videos)
- **NPDI-2K**: 95% accuracy (773 videos)
- **Custom Dataset**: 99% accuracy (352 videos)

Videos used for testing were extracted from the corresponding datasets and split into 1-minute segments. Incorrectly labeled videos were manually removed for accurate testing.




## Datasets
**[1]**: Duy, P., Nguyen, T., Nguyen, Q., Tran, H., Khac, N.-K., and Vu, L. (2022). LSPD: A large-scale pornographic dataset for detection and classification. International Journal of Intelligent Engineering and Systems, 15, 198.
**[2]**: Moreira, D., Avila, S., Perez, M., Moraes, D., Testoni, V., Valle, E., Goldenstein, S., and Rocha, A. (2016). Pornography classification: The hidden clues in video space–time. Forensic Science International, 268, 46–61.




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

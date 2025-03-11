# TSA Project

## Overview
This project implements a GhostNet-based feature extractor for image classification tasks. It includes functionalities for image preprocessing and feature extraction, leveraging TensorFlow and Keras.

## Project Structure
```
TSA
├── src
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── image_preprocessing.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Installation
To set up the project, ensure you have Python installed. Then, install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage
1. Place your dataset in a directory.
2. Update the `preprocessed_directory` variable in `src/main.py` with the path to your dataset.
3. Run the main script:

```
python src/main.py
```

## Functions
- **Feature Extraction**: The project includes a GhostNet-based model for extracting features from images.
- **Image Preprocessing**: Functions to load, resize, and normalize images for model input.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
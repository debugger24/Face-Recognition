# Face Recognition

Developed using openface (https://cmusatyalab.github.io/openface/)

# Functions

### generate_embedding.py

This is used to generate the embedding of all the images present in `data/train`. The embedding will be vector of length of 128. Embedding of all the images are stored in a dictionary in `embedding` directory.

### embedding.py

This file has `getRep` function that returns the embedding of the image.

### predict.py

This file has `getPrediction` function predicts the person using embedding passed to it.

### webcam.py

It captures the image from webcam and call `getRep` function from `embedding` and then call `getPrediction` to get the predicted result.
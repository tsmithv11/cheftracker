# Count the number of scenes a person appears in in a movie

_Note: This is very hacky and a work in progress_

_This is modified from Jason Brownlee's tutorial: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/_

This code takes a movie file and goes scene by scene and counts the number of scenes with each person from a trained model. It checks how many CPU cores your machine has and runs parrellel processes for each core. It also skips over some number of frames (currently set to `15`) to save time.

## Setup details

You need `facenet_keras.h5` in your directory. Find it in the tutorial linked above.

First, make sure you create a trained model using a folder with subfolders for each person saved as a pickle file (currently set to `svc_model.sav`) and encoder classes (currently set to `classes.npy`).

Make sure to change `{video/location.mp4}` to your video location.

# Count the number of scenes a person appears in in a movie

_Note: This is very hacky and a work in progress_

Make sure you create a trained model using a folder with subfolders for each person first.

This code takes a movie file and goes scene by scene and counts the number of scenes with each person from a trained model. It checks how many CPU cores your machine has and runs parrellel processes for each core. It also skips over some number of frames (currently set to 15) to save time. 

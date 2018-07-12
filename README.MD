# Dog breeds neural network

## Project for training neural network. 
Data is taken from [here](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## Training is splited into steps:
- extrecting and augumenting data images (augumentation.py)
- retraining pretrained model with prepared data (retrain.sh)
- testing trained model with random image (test.sh)
- converting .pb file into .tflite file (convert.sh)

# Webcam face recognition

This python project uses the [OpenCV](http://opencv.org/) library to recognize the person visible in webcam.

## Faces to recognize
The script uses the webcam to find a face, and then if it doesn't find it in the existing images, it ads it to the folder. All the images are saved with a corresponding .json file which saves the name of the person.

## Install
1. Download/clone this repo
2. pip install -r ./requirements.txt
3. Run: python face_recognizer.py

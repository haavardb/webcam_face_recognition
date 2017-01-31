# Webcam face recognition

This python project uses the [OpenCV](http://opencv.org/) library to recognize the person visible in webcam.

## Faces to recognize
Before you can run the recognizer you have to give it some images to train with. Upload these to the faces folder with the pattern "subject{number}."
Each subject you want to recognize should also have a corresponding .json file. subject{number}.json

`{
  "first_name":"foo",
  "last_name":"bar"
}`

## Install
1. Download/clone this repo
2. pip install -r ./requirements.txt
4. Upload images to faces folder
3. Run: python face_recognizer.py

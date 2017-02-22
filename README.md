# Webcam face recognition

This python project uses the [OpenCV](http://opencv.org/) library to recognize the person visible in webcam.

## Faces to recognize
The script uses the webcam to find a face, and then if it doesn't find it in the existing images, it ads it to the folder. All the images are saved with a corresponding .json file which saves the name of the person.

## Install
1. Download/clone this repo
2. Install [OpenCV](http://opencv.org/)
3. pip install -r ./requirements.txt
4. create a 'google_cloud.json' file with the content fetched from [Google Cloud API console](https://support.google.com/cloud/answer/6158857?hl=en).
5. Run: python face_recognizer.py

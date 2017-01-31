#!/usr/bin/python

import cv2, os
import numpy as np
import sys
import logging as log
import datetime as dt
import json
from playsound import playsound
from gtts import gTTS
from time import sleep
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()

# Captur video from webcam
video_capture = cv2.VideoCapture(0)

# Path to the Images, used to train the recognizer
path = './faces'

# Minimum confident that the recognizer has to have to make a match
# Closer to 0, the more confident
# I found 43 to be a good number, but this may vary..
minConf = 43

def get_images_and_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)  if not f.endswith('.json')]
    images = []
    labels = []
    for image_path in image_paths:

        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
    # return the images list and labels list
    return images, labels

# Call the get_images_and_labels function and get the face images and the
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray
    )

    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        cv_img = frame.astype(np.uint8)
        cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

        cv2.imshow("input", cv_gray)

        predict_image = np.array(cv_gray, 'uint8')
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])

        if conf < minConf:
            with open(path+'/subject'+str(nbr_predicted)+'.json') as data_file:
                data = json.load(data_file)
                print "Predicted as: {}, conf: {}".format(data["first_name"].encode('utf-8'), conf)

                sayText = "Heisann " + data["first_name"].encode('utf-8')
                tts = gTTS(text=sayText, lang='no')
                tts.save("hello.mp3")
                playsound('hello.mp3')
        else:
            print "No match found..."

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

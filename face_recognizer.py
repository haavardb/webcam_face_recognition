#!/usr/bin/python

import cv2, os
import numpy as np
import sys
import logging as log
import datetime as dt
import json
import progressbar
import speech_recognition as sr
import random

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
# I found 40 to be a good number, but this may vary..
minConf = 40

secondsBetweenOutput = 10

# Google cloud credentials
with open('google_cloud.json', 'r') as outfile:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = outfile.read()

def get_images_and_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)  if not f.endswith('.json')]
    images = []
    labels = []
    print "Training with sample images.."
    with progressbar.ProgressBar(max_value=len(image_paths)) as bar:
        count = 1
        for image_path in image_paths:
            bar.update(count)
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            image = np.array(image_pil, 'uint8')
            nbr = int(os.path.split(image_path)[1].split(".")[0])
            faces = faceCascade.detectMultiScale(image)
            count = count+1
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
        # return the images list and labels list
        return images, labels

def seconds_between(d1, d2):
    return abs((d2 - d1).seconds)

def main():

    fucking_mac_files_paths = [os.path.join(path, f) for f in os.listdir(path)  if f.endswith('.DS_Store')]
    for fucking_mac_files_path in fucking_mac_files_paths:
        os.remove(fucking_mac_files_path)

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

        uniqueSubjects = set(labels)

        for (x, y, w, h) in faces:

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3)

            cv_img = frame.astype(np.uint8)
            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

            cv2.imshow("input", cv_gray)

            predict_image = np.array(cv_gray, 'uint8')
            # nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])

            if 'lastMatch' not in locals():
                lastMatch = 0
                matchTime = dt.datetime.utcnow()

            if 'hasMatch' not in locals():
                hasMatch = False
                faceFoundTime = dt.datetime.utcnow()

            print "Predicted as: {}, conf: {}".format(nbr_predicted, conf)

            if conf < minConf:
                faceFoundTime = dt.datetime.utcnow()
                secBtw = seconds_between(matchTime, dt.datetime.utcnow())
                if (lastMatch != nbr_predicted) or (secBtw > secondsBetweenOutput):
                    # print "Predicted as: {}".format(nbr_predicted)
                    with open(path+'/'+str(nbr_predicted)+'.json') as data_file:
                        data = json.load(data_file)
                        print "Predicted as: {}, conf: {}".format(data["first_name"].encode('utf-8'), conf)
                        sayText = "Hei " + data["first_name"].encode('utf-8') + "!"
                        tts = gTTS(text=sayText, lang='no')
                        tts.save("hello.mp3")
                        playsound('hello.mp3')
                        lastMatch = nbr_predicted
                        matchTime = dt.datetime.utcnow()
                        # subjectNo = len(uniqueSubjects)+1
                        imgName = os.path.join(path, str(nbr_predicted)+'.'+str(random.randint(1, 100000))+'.png')
                        # cv2.imwrite(imgName, predict_image)
                        cv2.imwrite(imgName, cv_img[y: y + h, x: x + w])
            else:
                secBtwFaceFound = seconds_between(faceFoundTime, dt.datetime.utcnow())
                print "No match found for {} seconds".format(secBtwFaceFound)
                if secBtwFaceFound > 3:
                    # sayText = "Sorry love, but i don't recognize your ugly fucking face."
                    # tts = gTTS(text=sayText, lang='en-au')
                    sayText = "Sorry ass, men jeg kjenner ikke igjen det stygge trynet ditt, men jeg skal huske deg til neste gang. Hva heter du?"
                    tts = gTTS(text=sayText, lang='no')
                    tts.save("not_found.mp3")
                    playsound('not_found.mp3')
                    subjectNo = len(uniqueSubjects)+1

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    cv2.imwrite(os.path.join(path, '%d.'+str(random.randint(1, 100000))+'.png') % subjectNo, cv_img)
                    # cv2.imwrite(os.path.join(path, '%d.'+str(random.randint(1, 100000))+'.png') % subjectNo, cv_img[y: y + h, x: x + w])

                    # cv2.imwrite(imgName, )

                    theName=raw_input('Input:')
                    with open(os.path.join(path, '%d.json') % subjectNo, 'wb') as outfile:
                        json.dump({'first_name': theName}, outfile)
                        outfile.close()

                    # r = sr.Recognizer()
                    # with sr.Microphone() as source:
                    #     audio = r.listen(source)
                    #     theName = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, language="nb-NO")
                    #     print("Google Cloud Speech thinks you said " + theName)
                    #
                    #     sayText = "Heisann " + theName.encode('utf-8')
                    #     tts = gTTS(text=sayText, lang='no')
                    #     tts.save("not_found.mp3")
                    #     playsound('not_found.mp3')
                    #     with open(os.path.join(path, '%d.json') % subjectNo, 'w') as outfile:
                    #         json.dump({'first_name': theName}, outfile)

                        main()
                        break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

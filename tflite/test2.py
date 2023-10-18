import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf  # TensorFlow is required for Keras to work
import numpy as np
import math
import time

#capture object
cap = cv2.VideoCapture(0) #id number
detector = HandDetector(maxHands=2) #for now single hand (data collection)

offset = 50
imgSize = 300

prevKey = ord(".")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
interpreter = tf.lite.Interpreter(model_path="models/model1/asl_model.tflite")
interpreter.allocate_tensors()

# Load the labels
class_names = open("models/model2/labels.txt", "r").readlines()

print("classes: ", class_names)

while True:
    success,img = cap.read()
    predictionBaseImg = img.copy()
    pImg = img
    hands,img = detector.findHands(img)

    if hands:
        for i, hand in enumerate(hands):
            #hand = hands[0]
            x,y,w,h = hand['bbox'] #get the bounding box

            backgroundImage = np.ones((imgSize, imgSize, 3),np.uint8)*255
            predictionBackroundImage = np.ones((imgSize, imgSize, 3),np.uint8)*255
            
            croppedImage = img[y-offset:y+h+offset, x-offset:x+w+offset]
            croppedPredictionImage = predictionBaseImg[y-offset:y+h+offset, x-offset:x+w+offset]
            

            aspectRatio = h/w

            if croppedImage.size > 0:
                newW = w
                newH = h
                wGap = 0
                hGap = 0

                if aspectRatio > 1:
                    newW = math.floor(imgSize/aspectRatio)
                    newH = imgSize
                    wGap = wGap = math.floor((imgSize-newW)/2)
                else:
                    newH = math.floor(imgSize*aspectRatio)
                    newW = imgSize
                    hGap = math.floor((imgSize-newH)/2)
                

                resizedImg = cv2.resize(croppedImage, (newW, newH))
                backgroundImage[hGap:newH+hGap, wGap:newW+wGap] = resizedImg

                resizedPredictImg = cv2.resize(croppedPredictionImage, (newW, newH))
                predictionBackroundImage[hGap:newH+hGap, wGap:newW+wGap] = resizedPredictImg
                
                image = predictionBackroundImage.reshape(1, 300, 300, 3)

                # Predicts the model
                prediction = model.predict(image)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                

                cv2.putText(img, f'ASL Sign: {class_name[2:-1]} ({str(np.round(confidence_score * 100))[:-2]}%)',(x, y+h+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)



                # Print prediction and confidence score
                print("Class:", class_name[2:], end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    cv2.imshow("Image", img)
    cv2.waitKey(1) #1ms delays
    

# # import cv2
# # import numpy as np
# # import math

# # import tensorflow as tf

# # # capture object
# # cap = cv2.VideoCapture(0)  # id number

# # # Disable scientific notation for clarity
# # np.set_printoptions(suppress=True)

# # # Load the TensorFlow Lite model
# # interpreter = tf.lite.Interpreter(model_path="models/model1/asl_model.tflite")
# # interpreter.allocate_tensors()

# # # Load the labels
# # class_names = open("models/model1/labels.txt", "r").readlines()

# # print("classes: ", class_names)

# # while True:
# #     success, img = cap.read()
# #     predictionBaseImg = img.copy()

# #     # Resize the image to the desired input size of the model
# #     img = cv2.resize(img, (300, 300))

# #     image = img.reshape(1, 300, 300, 3)

# #     # Prepare input tensor for inference
# #     input_data = (image.astype(np.float32) / 255.0)  # Normalize if necessary

# #     # Set input tensor
# #     input_details = interpreter.get_input_details()
# #     interpreter.set_tensor(input_details[0]['index'], input_data)

# #     # Run inference
# #     interpreter.invoke()

# #     # Get output tensor
# #     output_details = interpreter.get_output_details()
# #     output_data = interpreter.get_tensor(output_details[0]['index'])

# #     # Process output
# #     index = np.argmax(output_data)
# #     class_name = class_names[index]
# #     confidence_score = output_data[0][index]

# #     # Print prediction and confidence score
# #     print("Class:", class_name[2:], end="")
# #     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

# #     # Display the image with the prediction
# #     cv2.putText(img, f'ASL Sign: {class_name[2:-1]} ({str(np.round(confidence_score * 100))[:-2]}%)',
# #                 (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

# #     cv2.imshow("Image", img)
# #     cv2.waitKey(1)  # 1ms delays



# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time

# import tensorflow as tf

# #capture object
# cap = cv2.VideoCapture(0) #id number
# detector = HandDetector(maxHands=2)

# offset = 50
# imgSize = 300

# prevKey = ord(".")

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path="models/model1/asl_model.tflite")
# interpreter.allocate_tensors()

# # Load the labels
# class_names = open("models/model1/labels.txt", "r").readlines()

# print("classes: ", class_names)

# while True:
#     success,img = cap.read()
#     predictionBaseImg = img.copy()
#     pImg = img
#     hands,img = detector.findHands(img)

#     height, width, channels = pImg.shape

#     if hands:
#         for i, hand in enumerate(hands):
#             x,y,w,h = hand['bbox'] #get the bounding box
#             predictionBackroundImage = np.ones((imgSize, imgSize, 3),np.uint8)*255
            
#             croppedImage = img[y-offset:y+h+offset, x-offset:x+w+offset]
#             # croppedPredictionImage = predictionBaseImg[y-offset:y+h+offset, x-offset:x+w+offset]
#             croppedPredictionImage = predictionBaseImg
#             croppedImage = pImg
            
#             aspectRatio = height/width

#             if croppedImage.size > 0:
                
#                 newW = width
#                 newH = height


#                 wGap = 0
#                 hGap = 0

#                 if aspectRatio > 1:
#                     newW = math.floor(imgSize/aspectRatio)
#                     newH = imgSize
#                     wGap = wGap = math.floor((imgSize-newW)/2)
#                 else:
#                     newH = math.floor(imgSize*aspectRatio)
#                     newW = imgSize
#                     hGap = math.floor((imgSize-newH)/2)
                

#                 resizedImg = cv2.resize(croppedImage, (newW, newH))

#                 resizedPredictImg = cv2.resize(croppedPredictionImage, (newW, newH))
#                 predictionBackroundImage[hGap:newH+hGap, wGap:newW+wGap] = resizedPredictImg

#                 cv2.imshow("resized image", predictionBackroundImage)
#                 cv2.waitKey(1) #1ms delays
                
#                 image = predictionBackroundImage.reshape(1, 300, 300, 3)

#                 # Prepare input tensor for inference
#                 input_data = (image.astype(np.float32) / 255.0)  # Normalize if necessary

#                 # Set input tensor
#                 input_details = interpreter.get_input_details()
#                 interpreter.set_tensor(input_details[0]['index'], input_data)

#                 # Run inference
#                 interpreter.invoke()

#                 # Get output tensor
#                 output_details = interpreter.get_output_details()
#                 output_data = interpreter.get_tensor(output_details[0]['index'])

#                 # Process output
#                 index = np.argmax(output_data)
#                 print(f"found: {index}")
#                 class_name = class_names[index]
#                 confidence_score = output_data[0][index]
#                 # (Your existing code for displaying text)
#                 cv2.putText(img, f'ASL Sign: {class_name[2:-1]} ({str(np.round(confidence_score * 100))[:-2]}%)',(x, y+h+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)



#                 # Print prediction and confidence score
#                 print("Class:", class_name[2:], end="")
#                 print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     cv2.imshow("Image", img)
#     cv2.waitKey(1) #1ms delays




import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

import tensorflow as tf

#capture object
cap = cv2.VideoCapture(0) #id number
detector = HandDetector(maxHands=2) #for now single hand (data collection)

offset = 50
imgSize = 300

prevKey = ord(".")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="models/model1/asl_model.tflite")
interpreter.allocate_tensors()

# Load the labels
class_names = open("models/model1/labels.txt", "r").readlines()

print("classes: ", class_names)

while True:
    success,img = cap.read()
    predictionBaseImg = img.copy()
    pImg = img
    hands,img = detector.findHands(img)

    if hands:
        for i, hand in enumerate(hands):
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

                # Prepare input tensor for inference
                input_data = (image.astype(np.float32) / 255.0)  # Normalize if necessary

                # Set input tensor
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get output tensor
                output_details = interpreter.get_output_details()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Process output
                index = np.argmax(output_data)
                print(f"found: {index}")
                class_name = class_names[index]
                confidence_score = output_data[0][index]
                # (Your existing code for displaying text)
                cv2.putText(img, f'ASL Sign: {class_name[2:-1]} ({str(np.round(confidence_score * 100))[:-2]}%)',(x, y+h+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)



                # Print prediction and confidence score
                print("Class:", class_name[2:], end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    cv2.imshow("Image", img)
    cv2.waitKey(1) #1ms delays
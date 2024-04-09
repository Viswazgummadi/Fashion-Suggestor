import os
import cv2
import math
import argparse
import csv

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def Predict(image_path):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    frame = cv2.imread(image_path)
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        return None

    for faceBox in faceBoxes:
        face = frame[max(0,faceBox[1]-20):min(faceBox[3]+20,frame.shape[0]-1), max(0,faceBox[0]-20):min(faceBox[2]+20, frame.shape[1]-1)]

        # Ensure the face region has valid dimensions
        if face.size == 0:
            print("Invalid face region")
            return None

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        if gender == 'Male':
            return 0
        else:
            return 1
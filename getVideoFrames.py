import cv2
import os
from os import listdir
import random
import re

videoFolderPath = "/Users/oscarwan/mimaBall"
imageDataPath = "/Users/oscarwan/bballDetection/images"
if not os.path.exists(imageDataPath):
    print(1)
    os.makedirs(imageDataPath + "/training")
for videos in os.listdir(videoFolderPath):
    if videos.endswith(".mp4") or videos.endswith(".MP4"): 
        #print(videos)
        print(videos)
    
        cap = cv2.VideoCapture(videoFolderPath + "/" + videos)
        allFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        capCount = 0
        while capCount < 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, allFrames))
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(imageDataPath + "/training/" + re.sub(r'[\W_]+', '', videos) + "_" + str(capCount) + ".jpg", frame)
                capCount += 1


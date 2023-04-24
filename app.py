import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import os
import cvzone

fps = cvzone.FPS()
cap = cv2.VideoCapture(0) # capture using default camera
cap.set(10,200) # set brightness of captured image to 200
hd = HandDetector(detectionCon=0.65) # setting confidence threshold to 65%


# Create a list of hand gesture images by reading each image file from the 'Fingers' folder using OpenCV's imread method
overlaylist=[]
folderpath = 'Fingers'
list = os.listdir(folderpath)
for imgpath in list:
    image = cv2.imread(f'{folderpath}/{imgpath}')
    overlaylist.append(image)

# print (overlaylist)


# Loop continuously until user interrupt
while True:
    # Read current frame from video capture device
    _, img = cap.read()
    
    # Resize the frame to desired output size
    img = cv2.resize(img, (1150, 900))
    
    # Update the FPS counter based on the current frame
    fps.update(img, pos=(890, 40), scale=2, color=(166, 89, 166))
    
    # Detect hands in the current frame
    hand, imgs = hd.findHands(img)

    # hand - is a list of dictionaries containing information about each detected hand, and imgs is a list of images showing the hand detection results.
    # print(hand)
    # print(imgs)

    if hand:
        inputhand = hand[0]
        bbox = inputhand["bbox"]
        lmlist = inputhand['lmList']
        handtype = inputhand['type']

        # fingersUp() method from the HandDetector class is used to determine how many fingers are extended. The result is assigned to the fingersup variable.
        fingersup = hd.fingersUp(inputhand)

        # fingersup returns an array of 0 and 1s, we count the 1s to get the total count of raised fingers
        totalfingers = fingersup.count(1)

        # setting the overlay image based on current state of the hand
        h, w, c = overlaylist[totalfingers - 1].shape
        # print(h,w,c)
        img[0:h, 0:w] = overlaylist[totalfingers - 1]

        cv2.rectangle(img, (0, 200), (170, 425), (166, 89, 166), cv2.FILLED) 
        cv2.putText(img, str(totalfingers), (45, 350), cv2.FONT_HERSHEY_PLAIN, 7, (255, 255, 255),5)

        # print(totalfingers)




    cv2.imshow('FRAME',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
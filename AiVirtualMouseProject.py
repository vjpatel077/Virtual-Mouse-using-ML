import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

################################################
wCam, hCam = 1080, 640
frameR = 100  # Frame Reduction
smoothening = 7

################################################
pTime = 0
ploX, ploxY = 0, 0  # Previous Location of X and Y
clocX, clocY = 0, 0  # Current Location of X and Y

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()  # This will give the size of the screen
# print(wScr, hScr) # 1536, 864

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPositon(img)
    print(lmList)

    # 2. Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen Values
            clocX = ploX + (x3 - ploX) / smoothening
            clocY = ploxY + (y3 - ploxY) / smoothening

            # 7. Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            ploX, ploxY = clocX, clocY

        # 8. Both Index and thumb are up: Clicking Mode
        if fingers[1] == 1 and fingers[0] == 1:
            # 9. Find the distance between fingers
            length, img, lineInfo = detector.findDistance(8, 4, img)
            print(length)
            # 10. Left Click mouse if distance is short
            if length < 90:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click(button='left')
                pyautogui.sleep(1)
        # 11. Both Index and thumb are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 1:
            # 12. Find the distance between fingers
            length, img, lineInfo = detector.findDistance(4, 12, img)
            print(length)
            # 13. Right Click mouse if distance is short
            if length < 20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click(button='right')
                pyautogui.sleep(1)

    # 14. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 15. Display

    cv2.imshow("Image", img)
    cv2.waitKey(1)

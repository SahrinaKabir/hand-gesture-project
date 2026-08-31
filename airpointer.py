import cv2
import pyautogui
import time

from hand_detector import HandDetector

# ----------------------------
# Settings
# ----------------------------

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

screenWidth, screenHeight = pyautogui.size()

detector = HandDetector()

frameReduction = 100
smoothening = 7

prevX, prevY = 0, 0
currX, currY = 0, 0

pTime = 0

# ----------------------------
# Click Settings
# ----------------------------

lastClickTime = 0
clickDelay = 0.4

lastRightClickTime = 0
rightClickDelay = 0.4

# ----------------------------
# Drag Settings
# ----------------------------

pinchStartTime = None
pinching = False
isDragging = False

dragHoldTime = 0.3
dragThreshold = 35

# ----------------------------
# Scroll Settings
# ----------------------------

prevScrollY = 0
scrollThreshold = 15
scrollSpeed = 60

# ----------------------------
# Main Loop
# ----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame, landmarkList, bbox = detector.findHands(frame)
    
    fingers = []

    if landmarkList:
       fingers = detector.fingersUp(landmarkList)
    #    print(fingers)

    h, w, _ = frame.shape

    cv2.rectangle(
        frame,
        (frameReduction, frameReduction),
        (w - frameReduction, h - frameReduction),
        (255, 0, 255),
        2
    )

    if landmarkList:
        
        currentTime = time.time()


        # ----------------------------
        # Mouse / Scroll
        # ----------------------------

        x = landmarkList[8][1]
        y = landmarkList[8][2]

        # Only Index Finger Up -> Move Mouse
        if (
    fingers[1] == 1 and
    fingers[2] == 0 and
    fingers[3] == 0 and
    fingers[4] == 0
):

            x = max(frameReduction, min(w - frameReduction, x))
            y = max(frameReduction, min(h - frameReduction, y))

            screenX = (
                (x - frameReduction)
                / (w - 2 * frameReduction)
            ) * screenWidth

            screenY = (
                (y - frameReduction)
                / (h - 2 * frameReduction)
            ) * screenHeight

            currX = prevX + (screenX - prevX) / smoothening
            currY = prevY + (screenY - prevY) / smoothening

            pyautogui.moveTo(currX, currY)

            prevX = currX
            prevY = currY

            cv2.circle(frame, (x, y), 12, (255, 0, 255), -1)
                    
        # ==================================================
        # SCROLL MODE
        # ==================================================

        if (
    fingers[1] == 1 and
    fingers[2] == 1 and
    fingers[3] == 0 and
    fingers[4] == 0
):

            cv2.putText(
                frame,
                "SCROLL MODE",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            scrollY = landmarkList[8][2]

            if prevScrollY != 0:

                delta = prevScrollY - scrollY

                if abs(delta) > scrollThreshold:

                    pyautogui.scroll(int(delta * scrollSpeed / 10))

            prevScrollY = scrollY

        else:

            prevScrollY = 0

        # ==================================================
        # LEFT CLICK / DRAG
        # ==================================================

        leftDistance = detector.distance(landmarkList, 4, 8)

        if leftDistance < dragThreshold:

            cv2.circle(
                frame,
                (landmarkList[4][1], landmarkList[4][2]),
                12,
                (0, 0, 255),
                -1
            )

            if not pinching:
                pinching = True
                pinchStartTime = currentTime

            holdTime = currentTime - pinchStartTime

            if holdTime >= dragHoldTime and not isDragging:

                pyautogui.mouseDown()

                isDragging = True

            if isDragging:

                cv2.putText(
                    frame,
                    "DRAGGING",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        else:

            if pinching:

                holdTime = currentTime - pinchStartTime

                if isDragging:

                    pyautogui.mouseUp()

                    isDragging = False

                elif (
                    holdTime < dragHoldTime
                    and currentTime - lastClickTime > clickDelay
                ):

                    pyautogui.click()

                    lastClickTime = currentTime

                pinching = False
                pinchStartTime = None

       # ==================================================
        # RIGHT CLICK
        # ==================================================

        if fingers == [1, 1, 1, 0, 0]:

            cv2.putText(
                frame,
                "RIGHT CLICK",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if currentTime - lastRightClickTime > rightClickDelay:

                pyautogui.rightClick()

                lastRightClickTime = currentTime

        # ----------------------------
        # Coordinates
        # ----------------------------

        cv2.putText(
            frame,
            f"X:{int(currX)}  Y:{int(currY)}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ----------------------------
    # FPS
    # ----------------------------

    cTime = time.time()

    fps = 1 / (cTime - pTime) if pTime else 0

    pTime = cTime

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("AirPointer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
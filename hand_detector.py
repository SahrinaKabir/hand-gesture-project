import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(self):

        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # Fingertip landmark IDs
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        landmarkList = []
        bbox = None

        if result.hand_landmarks:

            h, w, _ = img.shape

            hand = result.hand_landmarks[0]

            xList = []
            yList = []

            for idx, landmark in enumerate(hand):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                xList.append(x)
                yList.append(y)

                landmarkList.append([idx, x, y])

                if draw:

                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

                    cv2.putText(
                        img,
                        str(idx),
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (0, 255, 255),
                        1
                    )

            xmin = min(xList)
            xmax = max(xList)
            ymin = min(yList)
            ymax = max(yList)

            bbox = (xmin, ymin, xmax, ymax)

            if draw:

                cv2.rectangle(
                    img,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (255, 0, 255),
                    2
                )

                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),
                    (0,17)
                ]

                for start, end in connections:

                    x1, y1 = landmarkList[start][1], landmarkList[start][2]
                    x2, y2 = landmarkList[end][1], landmarkList[end][2]

                    cv2.line(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

        return img, landmarkList, bbox

    def distance(self, landmarkList, p1, p2):

        if len(landmarkList) == 0:
            return 0

        x1, y1 = landmarkList[p1][1], landmarkList[p1][2]
        x2, y2 = landmarkList[p2][1], landmarkList[p2][2]

        return math.hypot(x2 - x1, y2 - y1)

    def fingersUp(self, landmarkList):
        """
        Returns:
        [Thumb, Index, Middle, Ring, Little]

        1 = Finger Up
        0 = Finger Down
        """

        if len(landmarkList) != 21:
            return []

        fingers = []

        # -------------------------
        # Thumb
        # -------------------------
        # Assumes mirrored webcam (cv2.flip(frame, 1))
        if landmarkList[4][1] > landmarkList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # -------------------------
        # Other four fingers
        # -------------------------
        for tip in self.tipIds[1:]:

            if landmarkList[tip][2] < landmarkList[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
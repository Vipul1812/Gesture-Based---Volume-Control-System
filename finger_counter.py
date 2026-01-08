import cv2
import mediapipe as mp
import math


class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.draw = mp.solutions.drawing_utils
        self.blue_style = self.draw.DrawingSpec(color=(255, 0, 0), thickness=3)

    def _dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def count_fingers(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        actions = {
            "volume": None,
            "brightness": None,
            "gesture": "NONE",
            "distance_cm": 0
        }

        h, w, _ = frame.shape

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lms, handed in zip(result.multi_hand_landmarks,
                                       result.multi_handedness):

                label = handed.classification[0].label

                thumb = hand_lms.landmark[4]
                index = hand_lms.landmark[8]

                dist = self._dist(thumb, index)
                dist_cm = round(dist * 100, 2)

                if label == "Right":
                    actions["volume"] = dist
                    actions["gesture"] = "VOLUME"
                    actions["distance_cm"] = dist_cm

                if label == "Left":
                    actions["brightness"] = dist
                    actions["gesture"] = "BRIGHTNESS"
                    actions["distance_cm"] = dist_cm

                for i, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    color = (0, 255, 0) if i in [4, 8] else (255, 255, 0)
                    cv2.circle(frame, (cx, cy), 6, color, 3)

                self.draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.blue_style,
                    self.blue_style
                )

        return frame, actions

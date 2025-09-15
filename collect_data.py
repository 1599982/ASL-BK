# collect_data.py
import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

label = input("Introduce la letra que vas a capturar (A-Z): ").upper()
save_dir = os.path.join(DATA_DIR, label)
os.makedirs(save_dir, exist_ok=True)

counter = 0

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                npy_path = os.path.join(save_dir, f"{counter}.npy")
                np.save(npy_path, landmarks)
                counter += 1

        cv2.imshow("Recolectando datos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

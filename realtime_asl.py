from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64

app = Flask(__name__)
CORS(app)

# Cargar modelo entrenado
with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    """Extrae los landmarks en formato [x,y,z,...]"""
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data)

@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.json
    frame_data = data.get("frame")
    if not frame_data:
        return jsonify({"hands": []})

    # Convertir base64 a numpy array
    header, encoded = frame_data.split(",", 1)
    decoded = base64.b64decode(encoded)
    nparr = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    hands_results = []
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                data = extract_landmarks(hand_landmarks).reshape(1, -1)
                pred = model.predict(data)[0]

                hand_label = 1 if handedness.classification[0].label == "Left" else 2
                hands_results.append({"hand": hand_label, "letter": str(pred)})

    return jsonify({"hands": hands_results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

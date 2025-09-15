from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Cargar modelo entrenado
with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Estado global de predicciones
last_predictions = []

def extract_landmarks(hand_landmarks):
    """Extrae los landmarks en formato [x,y,z,...]"""
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data)

def gen_frames():
    global last_predictions
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            predictions = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # Extraer landmarks y predecir
                    data = extract_landmarks(hand_landmarks).reshape(1, -1)
                    pred = model.predict(data)[0]

                    # Determinar si es izquierda o derecha
                    label = handedness.classification[0].label  # "Left" o "Right"
                    if label == "Left":
                        color = (0, 0, 255)  # 🔴 rojo en BGR
                    else:
                        color = (255, 0, 0)  # 🔵 azul en BGR

                    # Dibujar landmarks con color personalizado
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=color, thickness=2)
                    )

                    # Guardar predicción
                    predictions.append({
                        "hand": 1 if label == "Left" else 2,
                        "letter": str(pred)
                    })

            last_predictions = predictions

            # Codificar frame en JPEG para el stream
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/prediction")
def prediction():
    return jsonify({"hands": last_predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import os
from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
from flask_cors import CORS
import threading
import queue


app = Flask(__name__)  # Correction de _name_ -> __name__
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": "*",
        "expose_headers": "*"
    }
})

# Queue pour la communication entre les threads
prediction_queue = queue.Queue()

# Charger le modèle
model = None
try:
    with open('model.p', 'rb') as model_file:
        model_dict = pickle.load(model_file)
    model = model_dict['model']
except FileNotFoundError:
    print("Erreur : Le fichier 'model.p' est introuvable.")

# Initialisation de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Dictionnaire pour les labels
labels_dict = {0: 'hello', 1: 'yes', 2:'no', 3:'peace'}

# Initialisation du moteur de synthèse vocale
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def process_frame(frame):
    data_aux = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    results = hands.process(frame_rgb)
    prediction_result = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            min_x, min_y = min(x_), min(y_)
            data_aux = [(x - min_x, y - min_y) for x, y in zip(x_, y_)]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if model:
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux)
                predicted_character = labels_dict[int(prediction[0])]

                prediction_result = predicted_character

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    return frame, prediction_result


def generate_frames():
    cap = cv2.VideoCapture(0)
    previous_prediction = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame, prediction = process_frame(frame)

        if prediction and prediction != previous_prediction:
            prediction_queue.put(prediction)
            previous_prediction = prediction
            threading.Thread(target=speak, args=(prediction,)).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_prediction')
def get_prediction():
    try:
        prediction = prediction_queue.get_nowait()
        return jsonify({"prediction": prediction})
    except queue.Empty:
        return jsonify({"prediction": None})


if __name__ == "__main__":  # Correction de _name_ -> __name__
    print("\nStarting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)

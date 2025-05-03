import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Charger le modèle
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

# Vérifier si la caméra est bien ouverte
if not cap.isOpened():
    print("Erreur d'ouverture de la caméra.")
    exit()

# Initialisation de MediaPipe pour la détection de mains
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Dictionnaire pour les labels
labels_dict = {0: 'hello', 1: 'yes', 2: 'no',3:'peace'}

# Initialisation du moteur de synthèse vocale
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

previous_prediction = None

while True:
    data_aux = []

    ret, frame = cap.read()

    if not ret:
        print("Erreur de capture vidéo")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

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

        # Normalisation des coordonnées
        min_x, min_y = min(x_), min(y_)
        data_aux = [(x - min_x, y - min_y) for x, y in zip(x_, y_)]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        data_aux = np.asarray(data_aux).reshape(1, -1)  # Convertir en tableau 2D


        prediction = model.predict(data_aux)
        predicted_character = labels_dict[int(prediction[0])]

        if predicted_character != previous_prediction:
            speak(predicted_character)
            previous_prediction = predicted_character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : la caméra n'a pas pu être ouverte.")
    exit()

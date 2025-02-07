import mediapipe as mp
print(mp.__version__)

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)
# Importation depuis le module interne _framework_bindings.image
from mediapipe.python._framework_bindings import image
# from mediapipe.framework.formats import image
from mediapipe import ImageFormat

import cv2
import numpy as np

# Chemin vers le modèle
model_path = 'model/pose_landmarker_heavy.task'

# Création des options en mode VIDEO
options = PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.2,
    min_tracking_confidence=0.2,
    output_segmentation_masks=True
)

# Ouvrir la vidéo (ou utiliser 0 pour la webcam)
cap = cv2.VideoCapture("videos/test_.mp4")
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo.")
    exit()

# Connexions entre landmarks (inspirées de MediaPipe Pose)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32)
]

# Couleurs pour landmarks et connexions
color_con = [(16, 199, 22), (0, 222, 205), (186, 38, 14)]
color_land = [(0, 39, 222), (0, 39, 222), (0, 39, 222)]

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        # Conversion de la frame de BGR à RGB et garantie que le tableau est contigu
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # Récupération du timestamp en millisecondes
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        print("timestamp_ms : ", timestamp_ms)

        print("frame_rgb : ", frame_rgb)

        
        # Créez l'objet MediaPipe Image en passant d'abord le tableau de données,
        # puis le format d'image (3 correspond à SRGB).
        mp_image_obj = image.Image(data=frame_rgb, image_format=ImageFormat.SRGB)

        print("la", mp_image_obj)

        # Détection des poses pour la frame vidéo
        pose_landmarker_result = landmarker.detect_for_video(mp_image_obj, timestamp_ms)

        height, width, _ = frame.shape

        if pose_landmarker_result.pose_landmarks:
            for i, pose in enumerate(pose_landmarker_result.pose_landmarks):
                # Dessiner un cercle pour chaque landmark
                for landmark in pose:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), radius=4,
                                color=color_land[i % len(color_land)], thickness=-1)

                # Dessiner les connexions entre les landmarks
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(pose) and end_idx < len(pose):
                        start_landmark = pose[start_idx]
                        end_landmark = pose[end_idx]
                        x1 = int(start_landmark.x * width)
                        y1 = int(start_landmark.y * height)
                        x2 = int(end_landmark.x * width)
                        y2 = int(end_landmark.y * height)
                        cv2.line(frame, (x1, y1), (x2, y2),
                                color=color_con[i % len(color_con)], thickness=2)

        cv2.imshow('Pose Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

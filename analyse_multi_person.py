import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialisation des modèles
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Charger YOLOv8 pré-entraîné pour détecter les personnes
model = YOLO("model/yolov8n.pt")  # Utilise YOLOv8 nano (plus rapide) ou remplace par 'yolov8s.pt' pour plus de précision

# Charger la vidéo
cap = cv2.VideoCapture("videos/test_3.mp4")

# Récupérer les dimensions de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('videos/output.mp4', cv2.VideoWqriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Initialiser Mediapipe Pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.2
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)[0]

        for detection in results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection.tolist()
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                person_roi = frame_rgb[y1:y2, x1:x2]
                if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                    person_results = pose.process(person_roi)
                    if person_results.pose_landmarks:
                        # Spécifications personnalisées
                        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
                        connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

                        # Dessiner sur une copie de la ROI pour créer une vignette
                        roi_copy = person_roi.copy()
                        mp_drawing.draw_landmarks(
                            roi_copy,
                            person_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=landmark_drawing_spec,
                            connection_drawing_spec=connection_drawing_spec
                        )
                        # Créer la vignette (150x150 pixels)
                        thumbnail = cv2.resize(roi_copy, (150, 150))
                        # Superposer la vignette dans le coin supérieur droit
                        thumb_h, thumb_w = thumbnail.shape[:2]
                        frame[10:10+thumb_h, frame_width - 10 - thumb_w:frame_width - 10] = thumbnail

                        # (Optionnel) Vous pouvez aussi dessiner les landmarks directement sur le frame complet :
                        # mp_drawing.draw_landmarks(frame, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        #                           landmark_drawing_spec=landmark_drawing_spec,
                        #                           connection_drawing_spec=connection_drawing_spec)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO + Mediapipe Multi-Person Pose Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #     # Utiliser YOLO pour détecter les personnes
    #     results = model(frame_rgb)[0]  # YOLO retourne une liste d'objets détectés

    #     for detection in results.boxes.data:
    #         x1, y1, x2, y2, conf, cls = detection.tolist()
            
    #         # Vérifier si l'objet détecté est une personne (classe 0 dans YOLO)
    #         if int(cls) == 0:  
    #             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    #             # Extraire la région de la personne détectée
    #             person_roi = frame_rgb[y1:y2, x1:x2]

    #             # Vérifier que la découpe ne soit pas vide
    #             if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
    #                 person_results = pose.process(person_roi)

    #                 if person_results.pose_landmarks:
    #                     # Convertir les coordonnées pour les remettre dans l'image principale
    #                     for lm in person_results.pose_landmarks.landmark:
    #                         lm_x = int((lm.x * (x2 - x1) + x1)/3)
    #                         lm_y = int((lm.y * (y2 - y1) + y1)/3)

    #                         # # Dessiner les landmarks sur l'image principale
    #                         cv2.circle(frame, (lm_x, lm_y), 5, (0, 255, 0), -1)

    #                     # Dessiner les connexions des points
    #                     mp_drawing.draw_landmarks(
    #                         frame, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    #                     )

    #             # Dessiner la boîte de détection autour de la personne
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             # cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     # Affichage et enregistrement
    #     cv2.imshow("YOLO + Mediapipe Multi-Person Pose Detection", frame)
    #     out.write(frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

cap.release()
out.release()
cv2.destroyAllWindows()

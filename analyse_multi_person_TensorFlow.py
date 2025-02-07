import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# utilisation CPU seul car pb compatibilité cuDNN/CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Initialisation de Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Charger le modèle TensorFlow (SSD MobileNet v2)
model_path = "model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
detect_fn = tf.saved_model.load(model_path)

# Ouvrir la vidéo
cap = cv2.VideoCapture("videos/test_5.mp4")

# Récupérer la taille de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('videos/output_tf.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Initialisation de Mediapipe Pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convertir en uint8 (correctif pour TensorFlow)
        frame_rgb = frame_rgb.astype(np.uint8)

        # Préparer l'image pour TensorFlow Object Detection
        input_tensor = tf.convert_to_tensor(frame_rgb)[tf.newaxis, ...]

        # Détection d'objets
        detections = detect_fn(input_tensor)

        # Extraire les boîtes de détection et les classes
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['detection_classes'] = detections['detection_classes'].astype(int)

        # Boucle sur les objets détectés
        for i in range(num_detections):
            if detections['detection_scores'][i] > 0.5 and detections['detection_classes'][i] == 1:  # Classe 1 = Personne
                y1, x1, y2, x2 = detections['detection_boxes'][i]
                x1, y1, x2, y2 = int(x1 * frame_width), int(y1 * frame_height), int(x2 * frame_width), int(y2 * frame_height)

                # Découper la région de la personne détectée
                person_roi = frame_rgb[y1:y2, x1:x2]

                # Vérifier que la découpe ne soit pas vide
                if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                    person_results = pose.process(person_roi)

                    if person_results.pose_landmarks:
                        # Convertir les coordonnées pour les remettre dans l'image principale
                        for lm in person_results.pose_landmarks.landmark:
                            lm_x = int(lm.x * (x2 - x1) + x1)
                            lm_y = int(lm.y * (y2 - y1) + y1)

                            # Dessiner les landmarks sur l'image principale
                            cv2.circle(frame, (lm_x, lm_y), 5, (0, 255, 0), -1)

                        # Dessiner les connexions des points
                        mp_drawing.draw_landmarks(
                            frame, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )

                # Dessiner la boîte autour de la personne
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {detections['detection_scores'][i]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Affichage et enregistrement
        cv2.imshow("TensorFlow + Mediapipe Multi-Person Pose Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

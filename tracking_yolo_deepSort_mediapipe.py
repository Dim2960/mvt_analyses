import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # Tracker DeepSORT
from supervision import BoxAnnotator
from supervision.detection.core import Detections

# --- Imports pour MediaPipe Tasks (PoseLandmarker) ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)
# Import pour créer un objet image pour MediaPipe
from mediapipe.python._framework_bindings import image
from mediapipe import ImageFormat

# --------------------- CONFIGURATION ---------------------
VIDEO_SOURCE = "videos/test_12.mp4"  # Vidéo ou 0 pour webcam
CONFIDENCE_THRESHOLD = 0.8           # Seuil de confiance pour YOLO

# --------------------- INITIALISATION YOLO & DEEPSORT ---------------------
model = YOLO("model/yolo11x.pt")

max_age = [400, 200, 600, 800, 100]
max_cosine_distance = [0.23, 0.17, 0.29, 0.26, 0.13]
n_init =[50, 25, 75, 100, 5, 10]
nn_budget = [300, 100, 500, 700, 50, 150]
max_iou_distance= [0.9, 0.5]

# Initialiser DeepSORT pour kata et combat
tracker = DeepSort(
    max_age= max_age[4],             # Nombre maximum de frames sans mise à jour avant la suppression d’un track
    n_init= n_init[1],               # Nombre minimum d’images pour confirmer un track
    max_cosine_distance= max_cosine_distance[1],  # Seuil de distance pour l’association des features
    nn_budget= nn_budget[1],          # Limite du budget de voisinage (peut être None)
    override_track_class= 0,  # Ici, on restreint le tracking à une classe en particulier 0:person
    half=False,
    bgr=True, 
    max_iou_distance= max_iou_distance[0]
)

cap = cv2.VideoCapture(VIDEO_SOURCE)
annotator = BoxAnnotator(thickness=2)

# --- Initialisation du VideoWriter pour enregistrer le résultat ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_file = "videos/result_" + VIDEO_SOURCE.replace("videos/", "")  # Nom du fichier de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore # Codec pour MP4
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# --------------------- INITIALISATION DU POSE LANDMARKER ---------------------
model_path = 'model/pose_landmarker_lite.task'
options = PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=3,  # Chaque ROI devrait contenir une seule personne mais sinon l'idtrck le plus petit est prioritaire et est affiché sur toutes les box
    min_pose_detection_confidence=0.75,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.95,
    output_segmentation_masks=True
)

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

color_con = [(16, 199, 22), (0, 222, 205), (186, 38, 14)]
color_land = [(0, 39, 222), (0, 39, 222), (0, 39, 222)]

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- Détection avec YOLO ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections_list = []
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if int(cls) == 0:
                largeur = x2 - x1
                hauteur = y2 - y1
                detections_list.append([[x1, y1, largeur, hauteur], conf, int(cls)])

        # --- Mise à jour du tracker DeepSORT ---
        tracks = tracker.update_tracks(detections_list, frame=frame)
        track_boxes = []
        track_ids = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()  # [x1, y1, x2, y2]
            track_boxes.append(bbox)
            track_ids.append(track_id)

        # --- Annotation des boîtes de suivi ---
        annotated_frame = frame.copy()
        if track_boxes:
            detections_obj = Detections(
                xyxy=np.array(track_boxes),
                class_id=np.array(track_ids, dtype=int)
            )
            annotated_frame = annotator.annotate(scene=annotated_frame, detections=detections_obj)
            for bbox, tid in zip(track_boxes, track_ids):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(annotated_frame, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- Analyse de la pose sur chaque ROI ---
            # Récupération du timestamp de la frame (en ms)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            roi_timestamp = timestamp_ms  # Initialise le timestamp pour chaque ROI

            for bbox, tid in zip(track_boxes, track_ids):
                # Extraire les coordonnées de la boîte de suivi en tant qu'entiers
                x1, y1, x2, y2 = map(int, bbox)
                # Vérifier si la boîte dépasse les dimensions du cadre
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                # Extraire la région d'intérêt (ROI) de l'image annotée selon les coordonnées de la boîte
                roi = annotated_frame[y1:y2, x1:x2]
                # Vérifier que la ROI n'est pas vide
                if roi.size == 0:
                    continue

                # Convertir la ROI de l'espace de couleur BGR à RGB pour MediaPipe
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                # Assurer que le tableau numpy est stocké de manière contiguë en mémoire
                roi_rgb = np.ascontiguousarray(roi_rgb)
                # Créer un objet image MediaPipe à partir de la ROI RGB
                mp_image_obj = image.Image(data=roi_rgb, image_format=ImageFormat.SRGB)

                # Détecter la pose dans la ROI en utilisant le pose landmarker et un timestamp pour les modes vidéo
                pose_landmarker_result = landmarker.detect_for_video(mp_image_obj, roi_timestamp)
                # Incrémenter le timestamp pour garantir des appels avec des timestamps strictement croissants
                roi_timestamp += 1

                roi_height, roi_width, _ = roi.shape
                if pose_landmarker_result.pose_landmarks:  # Vérifie si des landmarks de pose ont été détectés
                    for i, pose in enumerate(pose_landmarker_result.pose_landmarks):  # Itère sur chaque ensemble de landmarks détecté avec son indice
                        for landmark in pose:  # Parcourt chaque point clé dans la pose
                            x_lm = int(landmark.x * roi_width)  # Calcule la coordonnée x en fonction de la largeur de la ROI
                            y_lm = int(landmark.y * roi_height)  # Calcule la coordonnée y en fonction de la hauteur de la ROI
                            cv2.circle(roi, (x_lm, y_lm), radius=4,  # Dessine un cercle sur le landmark dans la ROI
                                        color=color_land[int(tid) % len(color_land)], thickness=-1)  # Couleur choisie cycliquement
                        for connection in POSE_CONNECTIONS:  # Parcourt chaque couple de landmarks à connecter
                            start_idx, end_idx = connection  # Extrait les indices de début et de fin pour la connexion
                            if start_idx < len(pose) and end_idx < len(pose):  # Vérifie que les indices sont valides pour la pose
                                start_landmark = pose[start_idx]  # Obtient le landmark de départ
                                end_landmark = pose[end_idx]  # Obtient le landmark d'arrivée
                                x1_line = int(start_landmark.x * roi_width)
                                y1_line = int(start_landmark.y * roi_height)
                                x2_line = int(end_landmark.x * roi_width)
                                y2_line = int(end_landmark.y * roi_height)
                                cv2.line(roi, (x1_line, y1_line), (x2_line, y2_line),
                                        color=color_con[int(tid) % len(color_con)], thickness=2)
                        break  # Affiche un seul groupe de landmarks

                annotated_frame[y1:y2, x1:x2] = roi  # Réinsère le ROI modifié dans le cadre annoté

        # Afficher la frame annotée
        cv2.imshow("YOLO11x + DeepSORT + Pose Landmarker", annotated_frame)
        
        # --- Écriture de la frame annotée dans le fichier vidéo ---
        video_writer.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
video_writer.release()  # Libère le VideoWriter
cv2.destroyAllWindows()

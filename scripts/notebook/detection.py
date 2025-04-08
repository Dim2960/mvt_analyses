import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from supervision import BoxAnnotator
from supervision.detection.core import Detections

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.python._framework_bindings import image
from mediapipe import ImageFormat

# Configuration initiale
VIDEO_SOURCE = "videos/test_13.mp4"
CONFIDENCE_THRESHOLD = 0.5 #0.8
MODEL_PATH = "model/yolo12x.pt"
POSE_MODEL_PATH = 'model/pose_landmarker_heavy.task'
OUTPUT_FILE = "videos/result_" + VIDEO_SOURCE.replace("videos/", "")

# Paramètres pour le tracker DeepSort
TRACKER_PARAMS = {
    'max_age': 400,  # Nombre maximum de frames pendant lesquels un objet peut être perdu avant d'être supprimé.
    'n_init': 50,  # Nombre de détections consécutives nécessaires pour initialiser une piste.
    'max_cosine_distance': 0.17,  # Distance cosinus maximale pour associer des détections à des pistes.
    'nn_budget': 300,  # Taille du budget pour le voisin le plus proche.
    'override_track_class': 0,  # Classe de l'objet à suivre (0 pour suivre la class person).
    'half': False,  # Indique si les images doivent être réduites de moitié.
    'bgr': True,  # Indique si les images sont en format BGR.
    'max_iou_distance': 0.9  # Distance IoU maximale pour associer des détections à des pistes.
}

# Paramètres pour le modèle PoseLandmarker de MediaPipe
LANDMARK_PARAMS = {
    'base_options': mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),  # Chemin vers le modèle PoseLandmarker.
    'running_mode': RunningMode.VIDEO,  # Mode d'exécution du modèle (VIDEO pour le traitement vidéo).
    'num_poses': 3,  # Nombre maximum de poses à détecter dans une image.
    'min_pose_detection_confidence': 0.75,  # Seuil de confiance minimum pour détecter une pose.
    'min_pose_presence_confidence': 0.5,  # Seuil de confiance minimum pour considérer qu'une pose est présente.
    'min_tracking_confidence': 0.95,  # Seuil de confiance minimum pour suivre une pose détectée.
    'output_segmentation_masks': True  # Indique si le modèle doit générer des masques de segmentation pour les poses détectées.
}


# Connexions entre les points clés des poses
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


# Couleurs pour les connexions et les points clés des poses
COLOR_CON = [
    (22, 199, 16),    # Vert clair
    (205, 222, 0),    # Cyan
    (0, 255, 255),    # Jaune
    (0, 165, 255),    # Orange
    (255, 255, 0),    # Cyan clair
    (128, 0, 128),    # Violet
    (255, 0, 255),    # Magenta
    (128, 128, 128),  # Gris
    (14, 38, 186),    # Rouge foncé
    (0, 0, 255),      # Rouge
    (0, 255, 0),      # Vert
    (0, 215, 255),    # Or
    (30, 105, 210),   # Chocolat
    (60, 20, 220),    # Rouge cerise
    (0, 128, 0),      # Vert foncé
    (255, 0, 0),      # Bleu
    (128, 0, 0),      # Bleu foncé
    (0, 128, 128),    # Vert olive
    (0, 0, 128),      # Marron
    (128, 128, 0),    # Turquoise
    (0, 0, 0),        # Noir
    (255, 255, 255)   # Blanc
]

COLOR_LAND = [
    (0, 39, 222), 
    (0, 39, 222), 
    (0, 39, 222)
]


def initialize_models():
    """
    Initialise les modèles YOLO et PoseLandmarker.

    Returns:
        yolo_model: Modèle YOLO pour la détection d'objets.
        pose_landmarker: Modèle PoseLandmarker pour la détection de poses.
    """
    yolo_model = YOLO(MODEL_PATH)
    pose_landmarker_options = PoseLandmarkerOptions(
        base_options=LANDMARK_PARAMS['base_options'],
        running_mode=LANDMARK_PARAMS['running_mode'],
        num_poses=LANDMARK_PARAMS['num_poses'],
        min_pose_detection_confidence=LANDMARK_PARAMS['min_pose_detection_confidence'],
        min_pose_presence_confidence=LANDMARK_PARAMS['min_pose_presence_confidence'],
        min_tracking_confidence=LANDMARK_PARAMS['min_tracking_confidence'],
        output_segmentation_masks=LANDMARK_PARAMS['output_segmentation_masks']
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_landmarker_options)
    return yolo_model, pose_landmarker


def initialize_tracker():
    """
    Initialise le tracker DeepSort.

    Returns:
        tracker: Tracker DeepSort configuré.
    """
    tracker = DeepSort(
        max_age=TRACKER_PARAMS['max_age'],
        n_init=TRACKER_PARAMS['n_init'],
        max_cosine_distance=TRACKER_PARAMS['max_cosine_distance'],
        nn_budget=TRACKER_PARAMS['nn_budget'],
        override_track_class=TRACKER_PARAMS['override_track_class'],
        half=TRACKER_PARAMS['half'],
        bgr=TRACKER_PARAMS['bgr'],
        max_iou_distance=TRACKER_PARAMS['max_iou_distance']
    )
    return tracker


def initialize_video_capture_and_writer():
    """
    Initialise la capture vidéo et l'écriture vidéo.

    Returns:
        cap: Objet de capture vidéo.
        video_writer: Objet d'écriture vidéo.
    """
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    video_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))
    return cap, video_writer


def detect_objects(model, frame):
    """
    Détecte les objets dans un frame avec YOLO.

    Args:
        model: Modèle YOLO.
        frame: Frame vidéo.

    Returns:
        detections_list: Liste des détections avec leurs coordonnées, confiance et classe.
    """
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections_list = []
    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) == 0:
            largeur = x2 - x1
            hauteur = y2 - y1
            detections_list.append([[x1, y1, largeur, hauteur], conf, int(cls)])
    return detections_list


def update_tracks(tracker, detections_list, frame):
    """
    Met à jour les pistes avec DeepSort.

    Args:
        tracker: Tracker DeepSort.
        detections_list: Liste des détections.
        frame: Frame vidéo.

    Returns:
        track_boxes: Liste des boîtes de suivi.
        track_ids: Liste des IDs de suivi.
    """
    tracks = tracker.update_tracks(detections_list, frame=frame)
    track_boxes = []
    track_ids = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        track_boxes.append(bbox)
        track_ids.append(track_id)
    return track_boxes, track_ids


def annotate_frame(frame, track_boxes, track_ids, annotator):
    """
    Annote les boîtes de suivi sur le frame.

    Args:
        frame: Frame vidéo.
        track_boxes: Liste des boîtes de suivi.
        track_ids: Liste des IDs de suivi.
        annotator: Annotateur pour les boîtes de détection.

    Returns:
        annotated_frame: Frame annoté.
    """
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
    return annotated_frame


def detect_poses(landmarker, roi, roi_timestamp):
    """
    Détecte les poses dans une région d'intérêt (ROI).

    Args:
        landmarker: Modèle PoseLandmarker.
        roi: Région d'intérêt.
        roi_timestamp: Timestamp pour la détection de poses.

    Returns:
        pose_landmarker_result: Résultat de la détection de poses.
    """
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_rgb = np.ascontiguousarray(roi_rgb)
    mp_image_obj = image.Image(data=roi_rgb, image_format=ImageFormat.SRGB)
    pose_landmarker_result = landmarker.detect_for_video(mp_image_obj, roi_timestamp)
    return pose_landmarker_result


def draw_poses(roi, pose_landmarker_result, track_id):
    """
    Dessine les poses détectées sur la région d'intérêt (ROI).

    Args:
        roi: Région d'intérêt.
        pose_landmarker_result: Résultat de la détection de poses.
        track_id: ID de suivi.

    Returns:
        roi: Région d'intérêt avec les poses dessinées.
    """
    roi_height, roi_width, _ = roi.shape
    if pose_landmarker_result.pose_landmarks:
        for i, pose in enumerate(pose_landmarker_result.pose_landmarks):
            for landmark in pose:
                x_lm = int(landmark.x * roi_width)
                y_lm = int(landmark.y * roi_height)
                cv2.circle(roi, (x_lm, y_lm), radius=4,
                            color=COLOR_LAND[int(track_id) % len(COLOR_LAND)], thickness=-1)
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pose) and end_idx < len(pose):
                    start_landmark = pose[start_idx]
                    end_landmark = pose[end_idx]
                    x1_line = int(start_landmark.x * roi_width)
                    y1_line = int(start_landmark.y * roi_height)
                    x2_line = int(end_landmark.x * roi_width)
                    y2_line = int(end_landmark.y * roi_height)
                    cv2.line(roi, (x1_line, y1_line), (x2_line, y2_line),
                            color=COLOR_CON[int(track_id) % len(COLOR_CON)], thickness=2)
            break
    return roi


def process_video():
    """
    Traite la vidéo en détectant, suivant et annotant les objets et les poses.
    """
    yolo_model, pose_landmarker = initialize_models()
    tracker = initialize_tracker()
    cap, video_writer = initialize_video_capture_and_writer()
    annotator = BoxAnnotator(thickness=2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detections_list = detect_objects(yolo_model, frame)
        track_boxes, track_ids = update_tracks(tracker, detections_list, frame)
        annotated_frame = annotate_frame(frame, track_boxes, track_ids, annotator)

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        roi_timestamp = timestamp_ms

        for bbox, tid in zip(track_boxes, track_ids):
            x1, y1, x2, y2 = map(int, bbox)
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue

            roi = annotated_frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            pose_landmarker_result = detect_poses(pose_landmarker, roi, roi_timestamp)
            roi_timestamp += 1
            roi = draw_poses(roi, pose_landmarker_result, tid)
            annotated_frame[y1:y2, x1:x2] = roi

        cv2.imshow("YOLOv8n + DeepSORT + Pose Landmarker", annotated_frame)
        video_writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()

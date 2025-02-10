import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import du tracker DeepSORT
from supervision import BoxAnnotator
from supervision.detection.core import Detections

# --------------------- CONFIGURATION ---------------------
VIDEO_SOURCE = "videos/test_8.mp4"  # Chemin de la vidéo ou 0 pour la webcam
# CONFIDENCE_THRESHOLD = 0.6        # Seuil de confiance pour YOLO - combat
CONFIDENCE_THRESHOLD = 0.8       # Seuil de confiance pour YOLO - Kata

# --------------------- INITIALISATION ---------------------
# Charger le modèle YOLOv8
model = YOLO("model/yolo11x.pt")  # Vous pouvez utiliser yolov8s.pt pour plus de rapidité

# # Initialiser DeepSORT pour combat
# tracker = DeepSort(
#     max_age=50,             # Nombre maximum de frames sans mise à jour avant la suppression d’un track
#     n_init=10,               # Nombre minimum d’images pour confirmer un track
#     max_cosine_distance=0.17,  # Seuil de distance pour l’association des features
#     nn_budget=300,          # Limite du budget de voisinage (peut être None)
#     override_track_class=0,  # Ici, on ne restreint pas le tracking à une classe en particulier
#     half = False
# )

# Initialiser DeepSORT pour kata
tracker = DeepSort(
    max_age=400,             # Nombre maximum de frames sans mise à jour avant la suppression d’un track
    n_init=30,               # Nombre minimum d’images pour confirmer un track
    max_cosine_distance=0.27,  # Seuil de distance pour l’association des features
    nn_budget=300,          # Limite du budget de voisinage (peut être None)
    override_track_class=0,  # Ici, on ne restreint pas le tracking à une classe en particulier
    half = False
)

# Initialiser OpenCV
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Création de l'annotateur de boîtes (Supervision)
annotator = BoxAnnotator(thickness=2)

# --------------------- BOUCLE PRINCIPALE ---------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- Détection avec YOLO ---
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Filtrer uniquement les personnes (classe 0 dans COCO)
    detections_list = []
    for det in results.boxes.data:
        # Chaque 'det' contient [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) == 0:  # Classe 0 = Personne
            # deep_sort_realtime attend une liste au format [x1, y1, x2, y2, confidence, class]
            detections_list.append([[x1, y1, x2/4, y2/2], conf, int(cls)])

    
    # --- Mise à jour du tracker DeepSORT ---
    # La méthode update_tracks renvoie une liste d’objets Track.
    tracks = tracker.update_tracks(detections_list, frame=frame)

    # Rassembler les boîtes et les IDs pour chaque track confirmé
    track_boxes = []
    track_ids = []
    for track in tracks:
        # On ne traite que les tracks confirmés
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        # La méthode to_ltrb() retourne [x1, y1, x2, y2]
        bbox = track.to_ltrb()
        track_boxes.append(bbox)
        track_ids.append(track_id)
    
    # --- Annotation et affichage ---
    annotated_frame = frame.copy()
    if len(track_boxes) > 0:
        # On crée un objet Detections pour l'annotateur Supervision
        confidences = np.ones(len(track_boxes))  # DeepSORT ne fournit pas de score de tracking à ce stade
        detections_obj = Detections(
            xyxy=np.array(track_boxes),
            confidence=confidences,
            class_id=np.array(track_ids, dtype=int)  # Ici, on utilise l’ID du track pour colorer ou annoter
        )
        annotated_frame = annotator.annotate(
            scene=annotated_frame,
            detections=detections_obj
        )
        # Ajouter manuellement le label (ID) à côté de chaque boîte
        for bbox, tid in zip(track_boxes, track_ids):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.putText(annotated_frame, f"ID {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 + DeepSORT", annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer les ressources
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # Tracker DeepSORT
from supervision import BoxAnnotator
from supervision.detection.core import Detections

# --------------------- CONFIGURATION ---------------------
VIDEO_SOURCE = "videos/test_8.mp4"  # Chemin de la vidéo ou 0 pour la webcam
CONFIDENCE_THRESHOLD = 0.8          # Seuil de confiance pour YOLO

# --------------------- INITIALISATION ---------------------
# Charger le modèle YOLOv8
model = YOLO("model/yolo11x.pt")  # Vous pouvez utiliser yolov8s.pt pour plus de rapidité

# Initialiser DeepSORT
tracker = DeepSort(
    max_age=400,             # Nombre maximum de frames sans mise à jour avant la suppression d’un track
    n_init=100,               # Nombre minimum d’images pour confirmer un track
    max_cosine_distance=0.4,  # Seuil de distance pour l’association des features
    nn_budget=400,          # Limite du budget de voisinage (peut être None)
    override_track_class=None,  # Ici, on ne restreint pas le tracking à une classe en particulier
    half = True
)

# Initialiser OpenCV
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Création de l'annotateur de boîtes (Supervision)
annotator = BoxAnnotator(thickness=2)

# Initialiser le soustracteur de fond (background subtraction) avec MOG2
backSub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=50, detectShadows=True)

# --------------------- BOUCLE PRINCIPALE ---------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- Suppression du fond ---
    # Appliquer le soustracteur pour obtenir un masque du premier plan
    fg_mask = backSub.apply(frame)
    # Seuil pour éliminer les ombres (les zones grises) et obtenir un masque binaire
    _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    # Optionnel : Appliquer une ouverture pour réduire le bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Détection avec YOLO ---
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Préparer la liste des détections pour DeepSORT
    # Nous filtrons ici uniquement les détections de personnes (classe 0)
    # et nous vérifions que la zone détectée correspond bien à un objet en mouvement
    detections_list = []
    for det in results.boxes.data:
        # Chaque 'det' contient [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) == 0:  # Classe 0 = Personne
            # Convertir les coordonnées en entiers et s'assurer qu'elles sont dans les limites du frame
            x1_int = max(0, int(x1))
            y1_int = max(0, int(y1))
            x2_int = min(frame.shape[1], int(x2))
            y2_int = min(frame.shape[0], int(y2))

            # Extraire la région d'intérêt (ROI) correspondante dans le masque de foreground
            roi_mask = fg_mask[y1_int:y2_int, x1_int:x2_int]
            if roi_mask.size == 0:
                continue

            # Calculer le ratio de pixels actifs dans le ROI
            white_pixels = cv2.countNonZero(roi_mask)
            roi_area = (x2_int - x1_int) * (y2_int - y1_int)
            if roi_area <= 0:
                continue
            ratio = white_pixels / float(roi_area)

            # On accepte la détection si le ratio de pixels actifs est supérieur à un seuil (ici 0.2)
            if ratio > 0.0:
                # Remarque : on passe ici la boîte complète sans modification (pas de division)
                detections_list.append([[x1, y1, x2, y2], conf, int(cls)])

    # --- Mise à jour du tracker DeepSORT ---
    # La méthode update_tracks renvoie une liste d’objets Track
    tracks = tracker.update_tracks(detections_list, frame=frame)

    # Rassembler les boîtes et les IDs pour chaque track confirmé
    track_boxes = []
    track_ids = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()  # Format [x1, y1, x2, y2]
        track_boxes.append(bbox)
        track_ids.append(track_id)

    # --- Annotation et affichage ---
    annotated_frame = frame.copy()

    # Afficher également le masque de foreground pour le débogage (optionnel)
    cv2.imshow("Foreground Mask", fg_mask)

    if len(track_boxes) > 0:
        # Créer un objet Detections pour l'annotateur Supervision
        confidences = np.ones(len(track_boxes))  # DeepSORT ne fournit pas de score de tracking
        detections_obj = Detections(
            xyxy=np.array(track_boxes),
            confidence=confidences,
            class_id=np.array(track_ids, dtype=int)  # Utiliser l’ID du track pour l’annotation
        )
        annotated_frame = annotator.annotate(
            scene=annotated_frame,
            detections=detections_obj
        )
        # Ajouter le label (ID) à côté de chaque boîte
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

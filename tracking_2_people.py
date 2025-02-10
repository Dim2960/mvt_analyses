import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace
from types import SimpleNamespace

from supervision import BoxAnnotator
# Importer la classe Detections attendue par BoxAnnotator
from supervision.detection.core import Detections

# --------------------- CONFIGURATION ---------------------
VIDEO_SOURCE = "videos/test_8.mp4"  # Chemin de la vidéo ou 0 pour la webcam
CONFIDENCE_THRESHOLD = 0.4          # Seuil de confiance pour YOLO

# --------------------- INITIALISATION ---------------------
# Charger le modèle YOLOv8
model = YOLO("model/yolov8x.pt")  # Vous pouvez utiliser yolov8s.pt pour plus de précision

# Préparer les arguments nécessaires pour ByteTrack
args = Namespace(
    track_buffer=3000,
    track_high_thresh=0.4,
    track_low_thresh=0.2,
    match_thresh=0.65,
    new_track_thresh=0.9,
    fuse_score=False
)
tracker = BYTETracker(args, frame_rate=30)

# Initialiser OpenCV
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Création de l'annotateur de boîtes
annotator = BoxAnnotator(thickness=2)



# --------------------- BOUCLE PRINCIPALE ---------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break


    # Détection avec YOLO
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Filtrer uniquement les personnes (classe 0 dans COCO)
    detections_list = []
    for det in results.boxes.data:
        # Chaque 'det' contient [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) == 0:  # Classe 0 = Personne
            detections_list.append([x1, y1, x2, y2, conf])
    
    # Convertir en numpy array (ou tableau vide si aucune détection)
    detections_arr = np.array(detections_list) if len(detections_list) > 0 else np.empty((0, 5))

    # Préparer un objet avec les attributs attendus par BYTETracker.update :
    # - conf: scores de confiance
    # - xywh: boîtes englobantes au format [x, y, width, height]
    # - cls: numéro de classe (ici, uniquement 0 pour les personnes)
    if detections_arr.shape[0] > 0:
        boxes = detections_arr[:, :4]       # x1, y1, x2, y2
        confs = detections_arr[:, 4]        # scores de confiance
        classes = np.zeros((detections_arr.shape[0],))  # Tous à 0 (seulement des personnes)
        
        # Conversion en format xywh : (x, y, width, height)
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = boxes[:, 0]              # x
        xywh[:, 1] = boxes[:, 1]              # y
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = x2 - x1
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = y2 - y1

        detection_results = SimpleNamespace(conf=confs, xywh=xywh, cls=classes)
    else:
        detection_results = SimpleNamespace(
            conf=np.empty((0,)),
            xywh=np.empty((0, 4)),
            cls=np.empty((0,))
        )

    # Mise à jour du tracker avec l'objet de détection et la forme de l'image
    tracks = tracker.update(detection_results, frame.shape)

    # Rassembler les boîtes et les IDs pour chaque track
    track_boxes = []
    track_ids = []
    for i, track in enumerate(tracks):
        # Chaque 'track' est supposé contenir [x1, y1, x2, y2, track_id, ...]
        x1, y1, x2, y2, tid = track[:5]
        track_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        # Si tid est None ou NaN, on utilise l'indice i comme identifiant par défaut
        if tid is None or (isinstance(tid, float) and np.isnan(tid)):
            track_ids.append(i)
        else:
            track_ids.append(int(tid))
    
    # Copier l'image pour l'annotation
    annotated_frame = frame.copy()
    if len(track_boxes) > 0:
        # IMPORTANT : On ajoute ici un vecteur de confiance factice pour que l'objet Detections soit complet.
        confidences = np.ones(len(track_boxes))
        # Créer un objet Detections à partir des boîtes ET en fournissant un attribut 'class_id'
        # qui sera utilisé pour déterminer la couleur dans l'annotation.
        detections_obj = Detections(
            xyxy=np.array(track_boxes),
            confidence=confidences,
            class_id=np.array(track_ids)
        )
        # On passe directement l'instance de la palette en lookup personnalisée

        annotated_frame = annotator.annotate(
            scene=annotated_frame,
            detections=detections_obj
        )
        # Ajouter manuellement les labels (IDs) à côté de chaque boîte
        for bbox, tid in zip(track_boxes, track_ids):
            x1, y1, x2, y2 = bbox
            cv2.putText(annotated_frame, f"ID {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



    # Afficher l'image annotée
    cv2.imshow("YOLOv8 + ByteTrack", annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer les ressources
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialisation de Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("videos/test_5.mp4")

# Définition du codec et du fichier de sortie vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('videos/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Activer la détection de plusieurs personnes
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
        results = pose.process(frame_rgb)

        # Vérifier si des personnes ont été détectées
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Affichage et enregistrement
        cv2.imshow("Mediapipe Multi-Person Pose Detection", frame)
        out.write(frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()  # Libérer le fichier vidéo
cv2.destroyAllWindows()

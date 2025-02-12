# Person Detection and Tracking with Pose Estimation

## 🌟 Description

Ce projet combine YOLO pour la détection de personnes, DeepSORT pour le suivi des personnes, et MediaPipe pour la détection de poses. Il traite une vidéo en détectant, suivant et annotant les personnes et les poses, puis enregistre le résultat dans un fichier vidéo.

Dans un cas d'usage, l'objectif serait de réaliser un suivi précis des combattants de judo dans une vidéo, afin de détecter et d'analyser les mouvements et techniques qu'ils exécutent. Ce projet vise à fournir des informations détaillées sur les actions des judokas, permettant ainsi une analyse statistique des techniques et de la performance globale lors d'un championnat, par exemple.

🚨 **Attention Limitation :** le nombre de personne détecté en terme de position n'est pas ajustable automatiquement est un max doit être defini dans les paramètres de mediapipe.landmarker

## 💡 Fonctionnalités

- Détection de personne avec YOLO.
- Suivi avec DeepSORT.
- Détection de poses avec MediaPipe.
- Annotation des personnes et des poses sur les frames vidéo.
- Enregistrement de la vidéo annotée.

## 💪 Prérequis

- Python 3.10.9
- OpenCV
- NumPy
- Ultralytics YOLO
- DeepSORT
- Supervision
- MediaPipe

## 🛠 Installation

1. Clonez le dépôt :

    ```bash
    git clone https://github.com/Dim2960/mvt_analyses.git
    cd mvt_analyses
    ```

2. Installez les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

## 🔍 Utilisation

1. Placez votre vidéo source dans le dossier \`videos\` et nommez-la \`test_17.mp4\`.

2. Exécutez le script principal :

    ```bash
    python main.py
    ```

3. Le script traitera la vidéo et enregistrera le résultat dans le dossier \`videos\` avec le préfixe \`result_\`.

## ⚙️ Configuration

Vous pouvez configurer les paramètres des modèles et du tracker en modifiant les dictionnaires \`MODEL_PARAMS\` et \`TRACKER_PARAMS\` dans le script principal.

Cette exemple est optimisé pour un combat de judo.

### Paramètres des modèles

```python
MODEL_PARAMS = {
    'yolo_model_path': 'model/yolo11x.pt',
    'pose_model_path': 'model/pose_landmarker_lite.task',
    'running_mode': RunningMode.VIDEO,
    'num_poses': 3,
    'min_pose_detection_confidence': 0.75,
    'min_pose_presence_confidence': 0.5,
    'min_tracking_confidence': 0.95,
    'output_segmentation_masks': True
}
```

### Paramètres du tracker

```python
TRACKER_PARAMS = {
    'max_age': 400,
    'n_init': 50,
    'max_cosine_distance': 0.17,
    'nn_budget': 300,
    'override_track_class': 0,
    'half': False,
    'bgr': True,
    'max_iou_distance': 0.9
}
```

## 🗃️ Structure du projet

```
mvt_analyses/
│
├── videos/
│   ├── test_17.mp4
│   ├── test_xx.mp4
│   ├── result_test_17.mp4
│   ├── result_test_xx.mp4
│   └── ...
│
├── model/
│   ├── yolo11x.pt
│   ├── pose_landmarker_lite.task
│   └── ...
│
├── detection.py
├── requirements.txt
├── LICENCE
└── README.md
```

## 📚 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [Ultralytics YOLO](https://github.com/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/)

## 🌐 Contribution

N'hésitez pas à contribuer au projet en soumettant des pull requests ou en signalant des bugs.

---
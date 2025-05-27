# Camera_LidarFusion

Système de détection d'objets et mesure de distance combinant YOLOv8n avec un LiDAR RPLIDAR A1M8 pour créer un système de perception temps réel.

## 🎯 Objectifs du Projet

Le projet Camera_LidarFusion vise à développer un système autonome de détection et d'alerte basé sur la fusion de données LiDAR et caméra pour des applications de sécurité et de surveillance.

## Fonctionnalités

- Détection d'objets en temps réel via la caméra de l'ordinateur
- Mesure de distance précise avec le LIDAR RPLIDAR A1M8
- Fusion des données pour associer objets détectés et distances
- Interface utilisateur intuitive pour visualiser les résultats
- Outils de calibration pour aligner le LIDAR et la caméra

## Prérequis

- Python 3.8 ou supérieur
- LIDAR RPLIDAR A1M8 connecté
- Webcam ou caméra intégrée

## Installation

1. Cloner ce dépôt
2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application :

```bash
python app.py
```

## Structure du Projet

```
projet_detection/
├── src/
│   ├── camera.py       # Gestion de la caméra et détection YOLOv8
│   ├── lidar.py        # Communication avec le LIDAR
│   ├── fusion.py       # Fusion des données caméra et LIDAR
│   ├── dashboard.py    # Interface utilisateur
│   └── calibrate.py    # Outils de calibration
├── config/
│   └── config.json     # Configuration du système
├── app.py              # Point d'entrée de l'application
├── requirements.txt    # Dépendances
└── README.md           # Documentation
```

## Calibration

Le système propose deux méthodes de calibration :

1. **Calibration manuelle** : L'utilisateur mesure et entre la distance et l'angle entre le LIDAR et la caméra
2. **Calibration semi-automatique** : Utilisation d'un objet de référence pour calculer automatiquement la transformation

## Extensions Futures

- Intégration GPS
- Support pour capteurs IMU
- Enregistrement vidéo et logs d'anomalies

## Licence

Ce projet est sous licence MIT.

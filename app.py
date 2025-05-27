#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application principale pour le système de détection d'objets et mesure de distance.
Combine la détection via YOLOv8 et la mesure de distance via LIDAR RPLIDAR A1M8.
"""

import os
import json
import logging
import sys
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detection_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DetectionSystem")

# Vérification des dossiers nécessaires
config_dir = Path("config")
if not config_dir.exists():
    config_dir.mkdir(exist_ok=True)
    logger.info(f"Dossier {config_dir} créé")

# Vérification du fichier de configuration
config_file = config_dir / "config.json"
if not config_file.exists():
    default_config = {
        "camera": {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "lidar": {
            "port": "COM3",  # À modifier selon votre système
            "baudrate": 115200
        },
        "calibration": {
            "angle_cam_lidar": 0.0,  # Angle en degrés entre caméra et LIDAR
            "distance_cam_lidar": 0.1,  # Distance en mètres entre caméra et LIDAR
            "camera_fov": 60.0  # Champ de vision horizontal de la caméra en degrés
        },
        "detection": {
            "confidence_threshold": 0.5,
            "classes": []  # Vide = toutes les classes
        },
        "interface": {
            "type": "streamlit"  # ou "tkinter"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=4)
    logger.info(f"Fichier de configuration par défaut créé: {config_file}")

# Import des modules du projet
try:
    from src.camera import CameraModule
    from src.lidar import LidarModule
    from src.fusion import FusionModule
    from src.dashboard import Dashboard
    from src.calibrate import CalibrationModule
    
    logger.info("Tous les modules ont été importés avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation des modules: {e}")
    sys.exit(1)


def main():
    """Point d'entrée principal de l'application"""
    logger.info("Démarrage du système de détection d'objets et mesure de distance")
    
    # Chargement de la configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info("Configuration chargée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        sys.exit(1)
    
    # Initialisation des modules
    try:
        camera_module = CameraModule(config["camera"])
        lidar_module = LidarModule(config["lidar"])
        calibration_module = CalibrationModule(config["calibration"])
        fusion_module = FusionModule(camera_module, lidar_module, calibration_module)
        
        # Choix de l'interface selon la configuration
        if config["ui"]["interface_type"].lower() == "streamlit":
            logger.info("Utilisation de l'interface Streamlit")
            print("Pour lancer l'interface Streamlit, exécutez: streamlit run src/dashboard.py")
            sys.exit(0)
        else:
            # Interface Tkinter par défaut
            dashboard = Dashboard(camera_module, lidar_module, fusion_module, calibration_module)
            dashboard.run()
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du système: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
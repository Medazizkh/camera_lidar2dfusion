#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de calibration pour aligner le LIDAR et la caméra.
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger("DetectionSystem.Calibration")

class CalibrationModule:
    """Classe pour gérer la calibration entre le LIDAR et la caméra."""
    
    def __init__(self, config):
        """Initialise le module de calibration avec la configuration spécifiée.
        
        Args:
            config (dict): Configuration pour la calibration
        """
        self.config = config
        self.config_file = Path("config/config.json")
        
        # Paramètres de calibration
        self.angle_cam_lidar = config.get("angle_cam_lidar", 0.0)  # Angle en degrés
        self.distance_cam_lidar = config.get("distance_cam_lidar", 0.1)  # Distance en mètres
        self.camera_fov = config.get("camera_fov", 60.0)  # Champ de vision horizontal en degrés
        
        # État de la calibration
        self.is_calibrated = False
        self.calibration_points = []
        
        logger.info("Module de calibration initialisé")
    
    def get_param(self, param_name):
        """Récupère un paramètre de calibration.
        
        Args:
            param_name (str): Nom du paramètre
            
        Returns:
            float: Valeur du paramètre
        """
        if param_name == "angle_cam_lidar":
            return self.angle_cam_lidar
        elif param_name == "distance_cam_lidar":
            return self.distance_cam_lidar
        elif param_name == "camera_fov":
            return self.camera_fov
        else:
            logger.warning(f"Paramètre de calibration inconnu: {param_name}")
            return None
    
    def set_param(self, param_name, value):
        """Définit un paramètre de calibration.
        
        Args:
            param_name (str): Nom du paramètre
            value (float): Nouvelle valeur
            
        Returns:
            bool: True si le paramètre a été modifié, False sinon
        """
        if param_name == "angle_cam_lidar":
            self.angle_cam_lidar = float(value)
            logger.info(f"Angle caméra-LIDAR défini à {self.angle_cam_lidar} degrés")
            return True
        elif param_name == "distance_cam_lidar":
            self.distance_cam_lidar = float(value)
            logger.info(f"Distance caméra-LIDAR définie à {self.distance_cam_lidar} mètres")
            return True
        elif param_name == "camera_fov":
            self.camera_fov = float(value)
            logger.info(f"Champ de vision de la caméra défini à {self.camera_fov} degrés")
            return True
        else:
            logger.warning(f"Paramètre de calibration inconnu: {param_name}")
            return False
    
    def save_calibration(self):
        """Sauvegarde les paramètres de calibration dans le fichier de configuration.
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Lecture du fichier de configuration existant
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Mise à jour des paramètres de calibration
            config["calibration"]["angle_cam_lidar"] = self.angle_cam_lidar
            config["calibration"]["distance_cam_lidar"] = self.distance_cam_lidar
            config["calibration"]["camera_fov"] = self.camera_fov
            
            # Écriture du fichier de configuration
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info("Paramètres de calibration sauvegardés")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des paramètres de calibration: {e}")
            return False
    
    def manual_calibration(self, angle, distance):
        """Effectue une calibration manuelle.
        
        Args:
            angle (float): Angle entre la caméra et le LIDAR en degrés
            distance (float): Distance entre la caméra et le LIDAR en mètres
            
        Returns:
            bool: True si la calibration a réussi, False sinon
        """
        try:
            self.angle_cam_lidar = float(angle)
            self.distance_cam_lidar = float(distance)
            
            logger.info(f"Calibration manuelle: angle={self.angle_cam_lidar}°, distance={self.distance_cam_lidar}m")
            
            # Sauvegarde des paramètres
            self.save_calibration()
            
            self.is_calibrated = True
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la calibration manuelle: {e}")
            return False
    
    def add_calibration_point(self, camera_detection, lidar_angle, lidar_distance):
        """Ajoute un point de calibration.
        
        Args:
            camera_detection (dict): Détection de la caméra
            lidar_angle (float): Angle LIDAR correspondant en degrés
            lidar_distance (float): Distance LIDAR correspondante en mètres
            
        Returns:
            int: Nombre de points de calibration actuels
        """
        point = {
            "camera": {
                "center": camera_detection["center"],
                "angle": camera_detection["horizontal_angle"]
            },
            "lidar": {
                "angle": lidar_angle,
                "distance": lidar_distance
            }
        }
        
        self.calibration_points.append(point)
        logger.info(f"Point de calibration ajouté: {len(self.calibration_points)} points au total")
        
        return len(self.calibration_points)
    
    def clear_calibration_points(self):
        """Efface tous les points de calibration."""
        self.calibration_points = []
        logger.info("Points de calibration effacés")
    
    def semi_auto_calibration(self):
        """Effectue une calibration semi-automatique à partir des points de calibration.
        
        Returns:
            bool: True si la calibration a réussi, False sinon
        """
        if len(self.calibration_points) < 3:
            logger.warning("Pas assez de points de calibration (minimum 3 requis)")
            return False
        
        try:
            # Calcul de l'angle moyen entre la caméra et le LIDAR
            angle_diffs = []
            for point in self.calibration_points:
                camera_angle = point["camera"]["angle"]
                lidar_angle = point["lidar"]["angle"]
                
                # Calcul de la différence d'angle (en tenant compte du passage de 359° à 0°)
                diff = (lidar_angle - camera_angle) % 360
                if diff > 180:
                    diff -= 360
                
                angle_diffs.append(diff)
            
            # Calcul de la moyenne des différences d'angle
            self.angle_cam_lidar = np.mean(angle_diffs)
            
            # Estimation de la distance entre la caméra et le LIDAR
            # (simplifié, une estimation plus précise nécessiterait une optimisation)
            self.distance_cam_lidar = 0.1  # Valeur par défaut
            
            logger.info(f"Calibration semi-automatique: angle={self.angle_cam_lidar}°, distance={self.distance_cam_lidar}m")
            
            # Sauvegarde des paramètres
            self.save_calibration()
            
            self.is_calibrated = True
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la calibration semi-automatique: {e}")
            return False
    
    def draw_calibration_target(self, frame, target_size=0.3):
        """Dessine une cible de calibration sur l'image.
        
        Args:
            frame (numpy.ndarray): Image sur laquelle dessiner
            target_size (float): Taille relative de la cible (0-1)
            
        Returns:
            numpy.ndarray: Image avec la cible
        """
        if frame is None:
            return None
        
        # Création d'une copie de l'image
        output = frame.copy()
        
        # Dimensions de l'image
        height, width = output.shape[:2]
        
        # Centre de l'image
        center_x = width // 2
        center_y = height // 2
        
        # Taille de la cible
        target_radius = int(min(width, height) * target_size / 2)
        
        # Dessin de la cible
        cv2.circle(output, (center_x, center_y), target_radius, (0, 255, 0), 2)
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.line(output, (center_x - target_radius, center_y), (center_x + target_radius, center_y), (0, 255, 0), 2)
        cv2.line(output, (center_x, center_y - target_radius), (center_x, center_y + target_radius), (0, 255, 0), 2)
        
        # Ajout de texte
        cv2.putText(output, "Placez l'objet de calibration au centre", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de fusion des données caméra et LIDAR.
"""

import logging
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger("DetectionSystem.Fusion")

class FusionModule:
    """Classe pour fusionner les données de la caméra et du LIDAR."""
    
    def __init__(self, camera_module, lidar_module, calibration_module):
        """Initialise le module de fusion.
        
        Args:
            camera_module: Module de gestion de la caméra
            lidar_module: Module de gestion du LIDAR
            calibration_module: Module de calibration
        """
        self.camera = camera_module
        self.lidar = lidar_module
        self.calibration = calibration_module
        
        logger.info("Module de fusion initialisé")
    
    def associate_detections_with_distances(self, detections=None):
        """Associe les détections de la caméra avec les distances du LIDAR.
        
        Args:
            detections (list, optional): Liste des détections. Si None, utilise les dernières détections.
            
        Returns:
            dict: Dictionnaire associant les indices de détection aux distances
        """
        if detections is None:
            detections = self.camera.current_detections
        
        if not detections:
            return {}
        
        # Récupération des paramètres de calibration
        angle_cam_lidar = self.calibration.get_param("angle_cam_lidar")
        
        # Dictionnaire pour stocker les distances associées aux détections
        distances = {}
        
        for i, detection in enumerate(detections):
            # Récupération de l'angle horizontal de l'objet détecté
            obj_angle = detection["horizontal_angle"]
            
            # Conversion de l'angle caméra en angle LIDAR
            lidar_angle = self._camera_angle_to_lidar_angle(obj_angle, angle_cam_lidar)
            
            # Récupération de la distance pour cet angle
            distance = self.lidar.get_distance_at_angle(lidar_angle)
            
            if distance is not None:
                distances[i] = distance
                logger.debug(f"Objet {detection['class_name']} à {distance:.2f}m (angle: {lidar_angle:.1f}°)")
        
        return distances
    
    def _camera_angle_to_lidar_angle(self, camera_angle, angle_cam_lidar):
        """Convertit un angle caméra en angle LIDAR.
        
        Args:
            camera_angle (float): Angle horizontal dans le repère caméra (en degrés)
            angle_cam_lidar (float): Angle entre la caméra et le LIDAR (en degrés)
            
        Returns:
            float: Angle dans le repère LIDAR (en degrés)
        """
        # L'angle LIDAR est l'angle caméra plus l'angle de décalage entre les deux capteurs
        lidar_angle = (camera_angle + angle_cam_lidar) % 360
        return lidar_angle
    
    def process_frame(self, frame=None):
        """Traite une image pour détecter les objets et associer les distances.
        
        Args:
            frame (numpy.ndarray, optional): Image à traiter. Si None, capture une nouvelle image.
            
        Returns:
            tuple: (image avec annotations, détections, distances associées)
        """
        # Capture d'une image si nécessaire
        if frame is None:
            frame = self.camera.get_frame()
            
        if frame is None:
            logger.warning("Aucune image disponible pour le traitement")
            return None, [], {}
        
        # Détection des objets
        detections = self.camera.detect_objects(frame)
        
        # Association des distances
        distances = self.associate_detections_with_distances(detections)
        
        # Dessin des détections avec les distances
        annotated_frame = self.camera.draw_detections(frame, detections, with_distance=True, distances=distances)
        
        return annotated_frame, detections, distances
    
    def get_3d_positions(self, detections=None, distances=None):
        """Calcule les positions 3D des objets détectés.
        
        Args:
            detections (list, optional): Liste des détections
            distances (dict, optional): Dictionnaire associant les indices de détection aux distances
            
        Returns:
            dict: Dictionnaire associant les indices de détection aux positions 3D (x, y, z)
        """
        if detections is None:
            detections = self.camera.current_detections
            
        if distances is None:
            distances = self.associate_detections_with_distances(detections)
        
        # Récupération des paramètres de calibration
        angle_cam_lidar = self.calibration.get_param("angle_cam_lidar")
        distance_cam_lidar = self.calibration.get_param("distance_cam_lidar")
        
        # Dictionnaire pour stocker les positions 3D
        positions_3d = {}
        
        for i, detection in enumerate(detections):
            if i in distances:
                # Récupération de la distance
                distance = distances[i]
                
                # Récupération de l'angle horizontal
                obj_angle = detection["horizontal_angle"]
                
                # Conversion en angle LIDAR
                lidar_angle = self._camera_angle_to_lidar_angle(obj_angle, angle_cam_lidar)
                
                # Conversion en radians
                lidar_angle_rad = np.radians(lidar_angle)
                
                # Calcul des coordonnées dans le repère LIDAR
                x_lidar = distance * np.cos(lidar_angle_rad)
                y_lidar = distance * np.sin(lidar_angle_rad)
                z_lidar = 0  # On suppose que tous les objets sont sur le même plan horizontal
                
                # Transformation des coordonnées du repère LIDAR au repère caméra
                # (simplifié, une transformation plus précise nécessiterait une matrice de transformation complète)
                x_cam = x_lidar - distance_cam_lidar * np.cos(np.radians(angle_cam_lidar))
                y_cam = y_lidar - distance_cam_lidar * np.sin(np.radians(angle_cam_lidar))
                z_cam = z_lidar
                
                positions_3d[i] = (x_cam, y_cam, z_cam)
        
        return positions_3d
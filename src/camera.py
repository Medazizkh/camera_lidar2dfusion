#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de gestion de la caméra et détection d'objets avec YOLOv8.
"""

import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO
from pathlib import Path

logger = logging.getLogger("DetectionSystem.Camera")

class CameraModule:
    """Classe pour gérer la caméra et la détection d'objets avec YOLOv8."""
    
    def __init__(self, config):
        """Initialise le module de caméra avec la configuration spécifiée.
        
        Args:
            config (dict): Configuration pour la caméra et la détection
        """
        self.config = config
        self.device_id = config.get("device_id", 0)
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.fps = config.get("fps", 30)
        
        # Initialisation de la caméra
        self.cap = None
        
        # Initialisation du modèle YOLOv8
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("Modèle YOLOv8n chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle YOLOv8: {e}")
            raise
        
        # Paramètres de détection
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.classes = config.get("classes", None)  # None = toutes les classes
        
        # Variables pour stocker les résultats de détection
        self.current_frame = None
        self.current_detections = []
        self.last_detection_time = 0
        
        logger.info("Module caméra initialisé")
    
    def start(self):
        """Démarre la capture vidéo."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise Exception(f"Impossible d'ouvrir la caméra {self.device_id}")
                
            logger.info(f"Caméra démarrée: ID={self.device_id}, {self.width}x{self.height} @ {self.fps}fps")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de la caméra: {e}")
            return False
    
    def stop(self):
        """Arrête la capture vidéo."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Caméra arrêtée")
    
    def get_frame(self):
        """Capture une image depuis la caméra.
        
        Returns:
            numpy.ndarray: Image capturée ou None en cas d'erreur
        """
        if not self.cap or not self.cap.isOpened():
            logger.warning("Tentative de capture d'image sans caméra active")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Échec de la capture d'image")
            return None
        
        self.current_frame = frame
        return frame
    
    def detect_objects(self, frame=None):
        """Détecte les objets dans l'image fournie ou dans la dernière image capturée.
        
        Args:
            frame (numpy.ndarray, optional): Image à analyser. Si None, utilise la dernière image capturée.
            
        Returns:
            list: Liste des détections (bounding boxes, classes, scores)
        """
        if frame is None:
            frame = self.current_frame
            
        if frame is None:
            logger.warning("Aucune image disponible pour la détection")
            return []
        
        try:
            # Exécution de la détection avec YOLOv8
            results = self.model(frame, conf=self.confidence_threshold, classes=self.classes)
            
            # Extraction des résultats
            detections = []
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Coordonnées de la boîte englobante (x1, y1, x2, y2)
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    # Classe de l'objet
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id]
                    # Score de confiance
                    conf = float(box.conf[0].item())
                    
                    # Calcul du centre de la boîte
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Calcul de l'angle horizontal par rapport au centre de l'image
                    # (utile pour l'association avec les données LIDAR)
                    img_center_x = frame.shape[1] / 2
                    horizontal_angle = self._pixel_to_angle(center_x, img_center_x, frame.shape[1])
                    
                    detection = {
                        "bbox": bbox,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "center": (center_x, center_y),
                        "horizontal_angle": horizontal_angle
                    }
                    detections.append(detection)
            
            self.current_detections = detections
            self.last_detection_time = time.time()
            
            return detections
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'objets: {e}")
            return []
    
    def _pixel_to_angle(self, pixel_x, center_x, image_width):
        """Convertit une position en pixels en angle horizontal.
        
        Args:
            pixel_x (float): Position horizontale en pixels
            center_x (float): Position centrale de l'image en pixels
            image_width (int): Largeur totale de l'image en pixels
            
        Returns:
            float: Angle en degrés (-FOV/2 à +FOV/2)
        """
        # Récupération du champ de vision horizontal de la caméra depuis la configuration
        camera_fov = self.config.get("camera_fov", 60.0)  # Valeur par défaut: 60 degrés
        
        # Calcul de l'angle
        normalized_pos = (pixel_x - center_x) / (image_width / 2)
        angle = normalized_pos * (camera_fov / 2)
        
        return angle
    
    def draw_detections(self, frame=None, detections=None, with_distance=False, distances=None):
        """Dessine les boîtes de détection sur l'image.
        
        Args:
            frame (numpy.ndarray, optional): Image sur laquelle dessiner. Si None, utilise la dernière image capturée.
            detections (list, optional): Liste des détections. Si None, utilise les dernières détections.
            with_distance (bool): Si True, affiche les distances associées aux objets
            distances (dict, optional): Dictionnaire associant les indices de détection aux distances
            
        Returns:
            numpy.ndarray: Image avec les détections dessinées
        """
        if frame is None:
            frame = self.current_frame
            
        if frame is None:
            logger.warning("Aucune image disponible pour dessiner les détections")
            return None
            
        if detections is None:
            detections = self.current_detections
            
        # Création d'une copie de l'image pour ne pas modifier l'original
        output = frame.copy()
        
        # Couleurs pour les différentes classes (générées aléatoirement mais cohérentes)
        np.random.seed(42)  # Pour avoir les mêmes couleurs à chaque fois
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
        # Dessiner chaque détection
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            
            # Sélection de la couleur en fonction de la classe
            color = colors[det["class_id"] % len(colors)].tolist()
            
            # Dessin de la boîte englobante
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Préparation du texte à afficher
            if with_distance and distances and i in distances:
                distance = distances[i]
                label = f"{cls_name} {conf:.2f} - {distance:.2f}m"
            else:
                label = f"{cls_name} {conf:.2f}"
            
            # Dessin du texte
            cv2.putText(output, label, (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
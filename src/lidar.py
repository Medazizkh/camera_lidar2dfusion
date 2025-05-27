#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de gestion du LIDAR RPLIDAR A1M8.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar
from threading import Thread, Lock

logger = logging.getLogger("DetectionSystem.Lidar")

class LidarModule:
    """Classe pour gérer le LIDAR RPLIDAR A1M8."""
    
    def __init__(self, config):
        """Initialise le module LIDAR avec la configuration spécifiée.
        
        Args:
            config (dict): Configuration pour le LIDAR
        """
        self.config = config
        self.port = config.get("port", "COM3")  # Port par défaut pour Windows
        self.baudrate = config.get("baudrate", 115200)
        
        # Initialisation du LIDAR
        self.lidar = None
        
        # Variables pour stocker les données du scan
        self.scan_data = []
        self.scan_data_lock = Lock()  # Pour l'accès thread-safe aux données
        
        # Thread pour la lecture continue des données
        self.scan_thread = None
        self.running = False
        
        logger.info("Module LIDAR initialisé")
    
    def start(self):
        """Démarre la connexion au LIDAR et la lecture des données."""
        try:
            # Connexion au LIDAR
            self.lidar = RPLidar(self.port, baudrate=self.baudrate)
            
            # Récupération des informations du LIDAR
            info = self.lidar.get_info()
            health = self.lidar.get_health()
            logger.info(f"LIDAR connecté: {info}")
            logger.info(f"État du LIDAR: {health}")
            
            # Démarrage du thread de lecture
            self.running = True
            self.scan_thread = Thread(target=self._scan_thread, daemon=True)
            self.scan_thread.start()
            
            logger.info(f"Lecture LIDAR démarrée sur le port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du LIDAR: {e}")
            if self.lidar:
                self.lidar.stop()
                self.lidar.disconnect()
                self.lidar = None
            return False
    
    def stop(self):
        """Arrête la connexion au LIDAR."""
        self.running = False
        
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=1.0)
        
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.disconnect()
                logger.info("LIDAR déconnecté")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du LIDAR: {e}")
            finally:
                self.lidar = None
    
    def _scan_thread(self):
        """Thread pour la lecture continue des données du LIDAR."""
        try:
            # Réinitialisation des données de scan
            with self.scan_data_lock:
                self.scan_data = []
            
            # Démarrage du scan
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                
                # Mise à jour des données de scan
                with self.scan_data_lock:
                    self.scan_data = scan
        except Exception as e:
            logger.error(f"Erreur dans le thread de lecture LIDAR: {e}")
            self.running = False
    
    def get_scan_data(self):
        """Récupère les dernières données de scan.
        
        Returns:
            list: Liste des points de scan (qualité, angle, distance)
        """
        with self.scan_data_lock:
            return self.scan_data.copy() if self.scan_data else []
    
    def get_distances_by_angle(self, start_angle=0, end_angle=360, step=1):
        """Récupère les distances pour une plage d'angles spécifiée.
        
        Args:
            start_angle (float): Angle de début en degrés
            end_angle (float): Angle de fin en degrés
            step (float): Pas entre les angles en degrés
            
        Returns:
            dict: Dictionnaire associant les angles aux distances
        """
        distances = {}
        scan_data = self.get_scan_data()
        
        # Normalisation des angles
        start_angle = start_angle % 360
        end_angle = end_angle % 360
        
        # Si l'angle de fin est inférieur à l'angle de début, on ajoute 360°
        if end_angle < start_angle:
            end_angle += 360
        
        # Création d'une liste d'angles à rechercher
        angles_to_find = np.arange(start_angle, end_angle, step) % 360
        
        # Pour chaque point du scan
        for _, angle, distance in scan_data:
            # Normalisation de l'angle
            angle_deg = angle % 360
            
            # Recherche de l'angle le plus proche dans la liste des angles à trouver
            closest_angle_idx = np.abs(angles_to_find - angle_deg).argmin()
            closest_angle = angles_to_find[closest_angle_idx]
            
            # Si l'angle est suffisamment proche
            if abs(angle_deg - closest_angle) < step / 2:
                # Conversion de la distance en mètres (le LIDAR renvoie en mm)
                distance_m = distance / 1000.0
                
                # Mise à jour de la distance pour cet angle
                if closest_angle not in distances or distance_m < distances[closest_angle]:
                    distances[closest_angle] = distance_m
        
        return distances
    
    def get_distance_at_angle(self, target_angle, tolerance=2.0):
        """Récupère la distance pour un angle spécifique.
        
        Args:
            target_angle (float): Angle cible en degrés
            tolerance (float): Tolérance en degrés pour la recherche de l'angle
            
        Returns:
            float: Distance en mètres ou None si aucune donnée n'est disponible
        """
        scan_data = self.get_scan_data()
        
        # Normalisation de l'angle cible
        target_angle = target_angle % 360
        
        # Recherche des points proches de l'angle cible
        matching_points = []
        for _, angle, distance in scan_data:
            # Normalisation de l'angle
            angle_deg = angle % 360
            
            # Calcul de la différence d'angle (en tenant compte du passage de 359° à 0°)
            angle_diff = min(abs(angle_deg - target_angle), 360 - abs(angle_deg - target_angle))
            
            if angle_diff <= tolerance:
                # Conversion de la distance en mètres
                distance_m = distance / 1000.0
                matching_points.append((angle_diff, distance_m))
        
        if not matching_points:
            return None
        
        # Tri des points par proximité d'angle
        matching_points.sort(key=lambda x: x[0])
        
        # Retourne la distance du point le plus proche de l'angle cible
        return matching_points[0][1]
    
    def visualize_scan(self, ax=None, max_distance=5.0):
        """Visualise les données du scan LIDAR.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes matplotlib pour le dessin
            max_distance (float): Distance maximale à afficher en mètres
            
        Returns:
            matplotlib.axes.Axes: Axes matplotlib avec le dessin
        """
        scan_data = self.get_scan_data()
        
        # Création d'un nouvel axe si nécessaire
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        # Effacement de l'axe
        ax.clear()
        
        # Configuration de l'axe
        ax.set_theta_zero_location('N')  # 0° au nord
        ax.set_theta_direction(-1)  # Sens horaire
        ax.set_rlim(0, max_distance)
        ax.set_title('Scan LIDAR')
        
        # Extraction des angles et distances
        angles = []
        distances = []
        
        for _, angle, distance in scan_data:
            # Conversion de l'angle en radians
            angle_rad = np.radians(angle)
            # Conversion de la distance en mètres
            distance_m = distance / 1000.0
            
            if distance_m <= max_distance:
                angles.append(angle_rad)
                distances.append(distance_m)
        
        # Dessin des points
        if angles and distances:
            ax.scatter(angles, distances, s=5, c='blue', alpha=0.7)
        
        return ax
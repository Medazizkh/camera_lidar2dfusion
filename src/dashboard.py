#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'interface utilisateur pour le système de détection d'objets et mesure de distance.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, Frame, Label, Button, Entry, Scale, HORIZONTAL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

logger = logging.getLogger("DetectionSystem.Dashboard")

class Dashboard:
    """Classe pour l'interface utilisateur du système de détection et mesure de distance."""
    
    def __init__(self, camera_module, lidar_module, fusion_module, calibration_module):
        """Initialise le dashboard avec les modules nécessaires.
        
        Args:
            camera_module: Module de gestion de la caméra
            lidar_module: Module de gestion du LIDAR
            fusion_module: Module de fusion des données
            calibration_module: Module de calibration
        """
        self.camera = camera_module
        self.lidar = lidar_module
        self.fusion = fusion_module
        self.calibration = calibration_module
        
        # Variables pour le contrôle de l'interface
        self.running = False
        self.calibration_mode = False
        
        # Création de la fenêtre principale
        self.root = None
        self.camera_canvas = None
        self.lidar_canvas = None
        self.log_text = None
        
        # Figure pour le scan LIDAR
        self.lidar_figure = plt.figure(figsize=(5, 5))
        self.lidar_ax = self.lidar_figure.add_subplot(111, projection='polar')
        
        # Thread pour la mise à jour de l'interface
        self.update_thread = None
        
        logger.info("Dashboard initialisé")
    
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        self.root = tk.Tk()
        self.root.title("Système de Détection d'Objets et Mesure de Distance")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Création d'un style pour les widgets
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))
        style.configure("TLabel", font=("Arial", 10))
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame pour la vidéo et le scan LIDAR
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame pour la vidéo
        camera_frame = ttk.LabelFrame(display_frame, text="Vidéo avec Détection")
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas pour la vidéo
        self.camera_canvas = tk.Canvas(camera_frame, bg="black")
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour le scan LIDAR
        lidar_frame = ttk.LabelFrame(display_frame, text="Scan LIDAR")
        lidar_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas pour le scan LIDAR
        self.lidar_canvas = FigureCanvasTkAgg(self.lidar_figure, lidar_frame)
        self.lidar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame pour les contrôles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Boutons de contrôle
        start_button = ttk.Button(controls_frame, text="Démarrer", command=self.start_system)
        start_button.pack(side=tk.LEFT, padx=5)
        
        stop_button = ttk.Button(controls_frame, text="Arrêter", command=self.stop_system)
        stop_button.pack(side=tk.LEFT, padx=5)
        
        # Frame pour la calibration
        calibration_frame = ttk.LabelFrame(main_frame, text="Calibration")
        calibration_frame.pack(fill=tk.X, pady=5)
        
        # Contrôles de calibration
        ttk.Label(calibration_frame, text="Angle caméra-LIDAR (degrés):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.angle_entry = ttk.Entry(calibration_frame, width=10)
        self.angle_entry.grid(row=0, column=1, padx=5, pady=5)
        self.angle_entry.insert(0, str(self.calibration.get_param("angle_cam_lidar")))
        
        ttk.Label(calibration_frame, text="Distance caméra-LIDAR (mètres):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.distance_entry = ttk.Entry(calibration_frame, width=10)
        self.distance_entry.grid(row=1, column=1, padx=5, pady=5)
        self.distance_entry.insert(0, str(self.calibration.get_param("distance_cam_lidar")))
        
        # Boutons de calibration
        manual_calib_button = ttk.Button(calibration_frame, text="Calibration Manuelle", command=self.manual_calibration)
        manual_calib_button.grid(row=0, column=2, padx=5, pady=5)
        
        auto_calib_button = ttk.Button(calibration_frame, text="Mode Calibration Auto", command=self.toggle_calibration_mode)
        auto_calib_button.grid(row=1, column=2, padx=5, pady=5)
        
        add_point_button = ttk.Button(calibration_frame, text="Ajouter Point", command=self.add_calibration_point)
        add_point_button.grid(row=0, column=3, padx=5, pady=5)
        
        run_auto_calib_button = ttk.Button(calibration_frame, text="Lancer Calibration Auto", command=self.run_auto_calibration)
        run_auto_calib_button.grid(row=1, column=3, padx=5, pady=5)
        
        # Frame pour les logs
        log_frame = ttk.LabelFrame(main_frame, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Zone de texte pour les logs
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar pour les logs
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Ajout d'un handler de logging pour afficher les logs dans l'interface
        self.log_handler = LogTextHandler(self.log_text)
        logger.addHandler(self.log_handler)
        
        logger.info("Interface utilisateur configurée")
    
    def run(self):
        """Lance l'interface utilisateur."""
        self.setup_ui()
        self.root.mainloop()
    
    def start_system(self):
        """Démarre le système de détection et mesure de distance."""
        if self.running:
            logger.warning("Le système est déjà en cours d'exécution")
            return
        
        # Démarrage de la caméra
        if not self.camera.start():
            messagebox.showerror("Erreur", "Impossible de démarrer la caméra")
            return
        
        # Démarrage du LIDAR
        if not self.lidar.start():
            self.camera.stop()
            messagebox.showerror("Erreur", "Impossible de démarrer le LIDAR")
            return
        
        # Démarrage du thread de mise à jour
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Système démarré")
    
    def stop_system(self):
        """Arrête le système de détection et mesure de distance."""
        if not self.running:
            logger.warning("Le système n'est pas en cours d'exécution")
            return
        
        # Arrêt du thread de mise à jour
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # Arrêt de la caméra et du LIDAR
        self.camera.stop()
        self.lidar.stop()
        
        logger.info("Système arrêté")
    
    def _update_loop(self):
        """Boucle de mise à jour de l'interface."""
        try:
            while self.running:
                # Capture et traitement d'une image
                frame = self.camera.get_frame()
                
                if frame is not None:
                    # Mode de calibration
                    if self.calibration_mode:
                        frame = self.calibration.draw_calibration_target(frame)
                    
                    # Traitement de l'image
                    annotated_frame, detections, distances = self.fusion.process_frame(frame)
                    
                    # Mise à jour de l'affichage de la caméra
                    self._update_camera_display(annotated_frame)
                
                # Mise à jour de l'affichage du LIDAR
                self._update_lidar_display()
                
                # Pause pour limiter l'utilisation du CPU
                time.sleep(0.03)  # ~30 FPS
        except Exception as e:
            logger.error(f"Erreur dans la boucle de mise à jour: {e}")
            self.running = False
    
    def _update_camera_display(self, frame):
        """Met à jour l'affichage de la caméra.
        
        Args:
            frame (numpy.ndarray): Image à afficher
        """
        if frame is None or self.camera_canvas is None:
            return
        
        # Redimensionnement de l'image pour l'affichage
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Conversion de l'image OpenCV (BGR) en image PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Redimensionnement de l'image pour s'adapter au canvas
            img_width, img_height = pil_img.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Conversion en PhotoImage pour l'affichage dans le canvas
            self.photo = ImageTk.PhotoImage(image=pil_img)
            
            # Mise à jour du canvas
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.photo, anchor=tk.CENTER
            )
    
    def _update_lidar_display(self):
        """Met à jour l'affichage du scan LIDAR."""
        if self.lidar_ax is None or self.lidar_canvas is None:
            return
        
        # Visualisation du scan LIDAR
        self.lidar.visualize_scan(self.lidar_ax)
        
        # Mise à jour du canvas
        self.lidar_canvas.draw()
    
    def manual_calibration(self):
        """Effectue une calibration manuelle avec les valeurs saisies."""
        try:
            angle = float(self.angle_entry.get())
            distance = float(self.distance_entry.get())
            
            if self.calibration.manual_calibration(angle, distance):
                messagebox.showinfo("Calibration", "Calibration manuelle effectuée avec succès")
            else:
                messagebox.showerror("Erreur", "Échec de la calibration manuelle")
        except ValueError:
            messagebox.showerror("Erreur", "Valeurs de calibration invalides")
    
    def toggle_calibration_mode(self):
        """Active ou désactive le mode de calibration."""
        self.calibration_mode = not self.calibration_mode
        
        if self.calibration_mode:
            logger.info("Mode de calibration activé")
            messagebox.showinfo("Calibration", "Mode de calibration activé. Placez un objet au centre de l'image et cliquez sur 'Ajouter Point'.")
        else:
            logger.info("Mode de calibration désactivé")
    
    def add_calibration_point(self):
        """Ajoute un point de calibration."""
        if not self.calibration_mode:
            messagebox.showwarning("Calibration", "Activez d'abord le mode de calibration")
            return
        
        # Capture d'une image
        frame = self.camera.get_frame()
        if frame is None:
            messagebox.showerror("Erreur", "Impossible de capturer une image")
            return
        
        # Détection des objets
        detections = self.camera.detect_objects(frame)
        
        if not detections:
            messagebox.showwarning("Calibration", "Aucun objet détecté dans l'image")
            return
        
        # Recherche de l'objet le plus proche du centre
        center_x = frame.shape[1] / 2
        center_y = frame.shape[0] / 2
        
        closest_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            det_center_x, det_center_y = detection["center"]
            distance = ((det_center_x - center_x) ** 2 + (det_center_y - center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_detection = detection
        
        # Récupération de l'angle LIDAR correspondant
        lidar_angle = 0  # Angle par défaut
        lidar_distance = 0  # Distance par défaut
        
        # Demande à l'utilisateur de confirmer l'angle et la distance LIDAR
        lidar_dialog = tk.Toplevel(self.root)
        lidar_dialog.title("Données LIDAR")
        lidar_dialog.geometry("300x150")
        lidar_dialog.transient(self.root)
        lidar_dialog.grab_set()
        
        ttk.Label(lidar_dialog, text="Angle LIDAR (degrés):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        lidar_angle_entry = ttk.Entry(lidar_dialog, width=10)
        lidar_angle_entry.grid(row=0, column=1, padx=5, pady=5)
        lidar_angle_entry.insert(0, "0")
        
        ttk.Label(lidar_dialog, text="Distance LIDAR (mètres):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        lidar_distance_entry = ttk.Entry(lidar_dialog, width=10)
        lidar_distance_entry.grid(row=1, column=1, padx=5, pady=5)
        lidar_distance_entry.insert(0, "0")
        
        def confirm_lidar_data():
            nonlocal lidar_angle, lidar_distance
            try:
                lidar_angle = float(lidar_angle_entry.get())
                lidar_distance = float(lidar_distance_entry.get())
                lidar_dialog.destroy()
                
                # Ajout du point de calibration
                num_points = self.calibration.add_calibration_point(
                    closest_detection, lidar_angle, lidar_distance)
                
                messagebox.showinfo("Calibration", f"Point de calibration ajouté ({num_points} points au total)")
            except ValueError:
                messagebox.showerror("Erreur", "Valeurs LIDAR invalides")
        
        ttk.Button(lidar_dialog, text="Confirmer", command=confirm_lidar_data).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Attente de la fermeture de la boîte de dialogue
        self.root.wait_window(lidar_dialog)
    
    def run_auto_calibration(self):
        """Lance la calibration semi-automatique."""
        if self.calibration.semi_auto_calibration():
            # Mise à jour des champs de saisie
            self.angle_entry.delete(0, tk.END)
            self.angle_entry.insert(0, str(self.calibration.get_param("angle_cam_lidar")))
            
            self.distance_entry.delete(0, tk.END)
            self.distance_entry.insert(0, str(self.calibration.get_param("distance_cam_lidar")))
            
            messagebox.showinfo("Calibration", "Calibration semi-automatique effectuée avec succès")
        else:
            messagebox.showerror("Erreur", "Échec de la calibration semi-automatique")
    
    def on_closing(self):
        """Gère la fermeture de l'application."""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application?"):
            self.stop_system()
            self.root.destroy()


class LogTextHandler(logging.Handler):
    """Handler de logging pour afficher les logs dans un widget Text."""
    
    def __init__(self, text_widget):
        """Initialise le handler avec le widget Text spécifié.
        
        Args:
            text_widget: Widget Text pour l'affichage des logs
        """
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        """Affiche un message de log dans le widget Text.
        
        Args:
            record: Enregistrement de log
        """
        msg = self.format(record) + '\n'
        
        # Insertion du message dans le widget Text
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)  # Défilement automatique


# Version Streamlit de l'interface
def streamlit_dashboard():
    """Interface utilisateur avec Streamlit."""
    import streamlit as st
    import json
    from pathlib import Path
    
    # Chargement de la configuration
    config_file = Path("config/config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        st.error("Fichier de configuration introuvable")
        return
    
    # Titre de l'application
    st.title("Système de Détection d'Objets et Mesure de Distance")
    
    # Initialisation des modules
    from camera import CameraModule
    from lidar import LidarModule
    from fusion import FusionModule
    from calibrate import CalibrationModule
    
    # Création des modules
    camera_module = CameraModule(config["camera"])
    lidar_module = LidarModule(config["lidar"])
    calibration_module = CalibrationModule(config["calibration"])
    fusion_module = FusionModule(camera_module, lidar_module, calibration_module)
    
    # Sidebar pour les contrôles
    st.sidebar.header("Contrôles")
    
    # Boutons de contrôle
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.sidebar.button("Démarrer" if not st.session_state.running else "Arrêter"):
        st.session_state.running = not st.session_state.running
        
        if st.session_state.running:
            # Démarrage de la caméra et du LIDAR
            camera_module.start()
            lidar_module.start()
            st.sidebar.success("Système démarré")
        else:
            # Arrêt de la caméra et du LIDAR
            camera_module.stop()
            lidar_module.stop()
            st.sidebar.info("Système arrêté")
    
    # Calibration
    st.sidebar.header("Calibration")
    
    # Paramètres de calibration
    angle_cam_lidar = st.sidebar.number_input(
        "Angle caméra-LIDAR (degrés)",
        value=float(calibration_module.get_param("angle_cam_lidar")),
        step=0.1
    )
    
    distance_cam_lidar = st.sidebar.number_input(
        "Distance caméra-LIDAR (mètres)",
        value=float(calibration_module.get_param("distance_cam_lidar")),
        step=0.01
    )
    
    # Bouton de calibration manuelle
    if st.sidebar.button("Calibration Manuelle"):
        if calibration_module.manual_calibration(angle_cam_lidar, distance_cam_lidar):
            st.sidebar.success("Calibration manuelle effectuée avec succès")
        else:
            st.sidebar.error("Échec de la calibration manuelle")
    
    # Mode de calibration
    if 'calibration_mode' not in st.session_state:
        st.session_state.calibration_mode = False
    
    if st.sidebar.checkbox("Mode Calibration", value=st.session_state.calibration_mode):
        st.session_state.calibration_mode = True
        st.sidebar.info("Mode de calibration activé. Placez un objet au centre de l'image.")
    else:
        st.session_state.calibration_mode = False
    
    # Affichage principal
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Vidéo avec Détection")
        video_placeholder = st.empty()
    
    with col2:
        st.header("Scan LIDAR")
        lidar_placeholder = st.empty()
    
    # Boucle principale
    if st.session_state.running:
        # Capture et traitement d'une image
        frame = camera_module.get_frame()
        
        if frame is not None:
            # Mode de calibration
            if st.session_state.calibration_mode:
                frame = calibration_module.draw_calibration_target(frame)
            
            # Traitement de l'image
            annotated_frame, detections, distances = fusion_module.process_frame(frame)
            
            # Affichage de l'image
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Création d'une figure pour le scan LIDAR
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        lidar_module.visualize_scan(ax)
        
        # Affichage du scan LIDAR
        lidar_placeholder.pyplot(fig)
    
    # Logs
    st.header("Logs")
    log_placeholder = st.empty()
    
    # Affichage des logs (simulé)
    logs = [
        "Système initialisé",
        "Caméra connectée",
        "LIDAR connecté"
    ]
    
    log_placeholder.text("\n".join(logs))


if __name__ == "__main__":
    # Pour exécuter l'interface Streamlit directement
    streamlit_dashboard()
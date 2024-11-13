#!/usr/bin/env python
import cv2
import numpy as np
import cv2.aruco as aruco
 
class MarkerFinder:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.aruco_dict = None
        self.parameters = aruco.DetectorParameters_create()
        self.markers = []
 
    def marker_param(self, marker_size, aruco_dict_type):
       
        # Imposta il dizionario ArUco
        self.aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_type))
        self.marker_size = marker_size
 
    def detect_markers_poses(self, rgb_image, trans_camera_pose, max_distance):
        # Rileva i marker ArUco nell'immagine
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
       
        if ids is not None:
            self.markers = []
            for i, corner in enumerate(corners):
                # Stima la posa del marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, self.marker_size, self.camera_matrix, self.dist_coeffs)
                position = tvec[0][0]
                rotation, _ = cv2.Rodrigues(rvec[0][0])
                quaternion = tf.transformations.quaternion_from_matrix(rotation)
               
                marker_info = {
                    'id': int(ids[i]),
                    'position': position,
                    'rotation': quaternion
                }
                self.markers.append(marker_info)
               
                # Disegna il marker sull'immagine
                aruco.drawDetectedMarkers(rgb_image, [corner])
                aruco.drawAxis(rgb_image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], self.marker_size / 2)
        else:
            self.markers = []
 
    def draw(self, image, color, thickness):
        for marker in self.markers:
            # Aggiungi il disegno dei marker sull'immagine
            cv2.drawMarker(image, (int(marker['position'][0]), int(marker['position'][1])), color, thickness)
 

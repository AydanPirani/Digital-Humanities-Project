import cv2
import mediapipe as mp

class FaceDetectionTools:
    def __init__(self):
        self.MP_mesh = mp.solutions.face_mesh
        self.mesh = self.MP_mesh.FaceMesh()

        self.MP_detection = mp.solutions.face_detection
        self.detection = self.MP_detection.FaceDetection()

        self.MP_draw = mp.solutions.drawing_utils
        pass

    def detect_face(self):
        pass

    def generate_mesh(self):
        pass

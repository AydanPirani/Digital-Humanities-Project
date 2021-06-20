import cv2
import mediapipe as mp


class FaceDetectionTools:
    # Initialize tools that will be used to either detect face or generate the mesh
    def __init__(self):
        self.MP_facemesh = mp.solutions.face_mesh
        self.MP_detection = mp.solutions.face_detection

        self.facemesh = self.MP_facemesh.FaceMesh(max_num_faces=2)
        self.detection = self.MP_detection.FaceDetection()

        self.faces = []
        self.landmarks = []

        self.MP_draw = mp.solutions.drawing_utils
        pass

    # Use MP face detection to find and display faces
    def detect_face(self, img):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detection.process(new_img)
        faces = results.detections
        self.faces = []
        if faces:
            for face in faces:
                self.faces.append(face)
        return

    # Use MP mesh function to get landmark points
    def generate_mesh(self, img):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.facemesh.process(new_img)
        landmarks = results.multi_face_landmarks
        self.landmarks = []
        if landmarks:
            for landmark_list in landmarks:
                self.landmarks.append(landmark_list)
        return

    # Draw all face detections, landmarks included
    def draw(self, img):
        for f in self.faces:
            self.MP_draw.draw_detection(img, f)
        for l in self.landmarks:
            self.MP_draw.draw_landmarks(img, l)
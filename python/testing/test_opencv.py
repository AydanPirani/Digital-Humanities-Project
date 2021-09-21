import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=3, circle_radius=2)

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
import cv2
import mediapipe as mp
import numpy
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from matplotlib.path import Path
from shapely.geometry import Polygon
from sklearn.decomposition import NMF, KernelPCA


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

FOREHEAD_POINTS = [251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 139, 70, 63, 105, 66, 107,
                   9, 336, 296, 334, 293, 300, 383, 368, 389]
LCHEEK_POINTS = [31, 35, 143, 116, 123, 147, 213, 192, 214, 212, 216, 206, 203, 36, 101, 119, 229, 228]
RCHEEK_POINTS = [261, 265, 372, 345, 352, 376, 433, 434, 432, 436, 426, 423, 266, 330, 348, 449, 448]


# Given an image and points to take measurements from, return "valid" points in the image
def get_points(n_img, landmarks):
    i_h, i_w, i_c = n_img.shape
    for faceLms in landmarks[:1]:
        # List of all the landmark coordinates from the generated face
        forehead_landmarks = [(i_w * f.x, i_h * f.y) for f in [faceLms.landmark[i] for i in FOREHEAD_POINTS]]
        lcheek_landmarks = [(i_w * f.x, i_h * f.y) for f in [faceLms.landmark[i] for i in LCHEEK_POINTS]]
        rcheek_landmarks = [(i_w * f.x, i_h * f.y) for f in [faceLms.landmark[i] for i in RCHEEK_POINTS]]

        # Generating MPL paths for each body part - used to iterate pixels
        forehead_path = Path(forehead_landmarks)
        lcheek_path = Path(lcheek_landmarks)
        rcheek_path = Path(rcheek_landmarks)

        lcheek_area, rcheek_area = Polygon(lcheek_landmarks).area, Polygon(rcheek_landmarks).area

        forehead_pts = set() # Set of all pixels to change
        rcheek_pts, lcheek_pts = set(), set()

        # Iterate through all pixels in image, check if pixel in path, then add
        for i in range(i_h):
            for j in range(i_w):
                if forehead_path.contains_point((j, i)):
                    forehead_pts.add((i, j))

                # Check if the left cheek is at least 1/2 the size of the right cheek - if so, add to points
                if lcheek_area/rcheek_area > 0.5 and lcheek_path.contains_point((j, i)):
                    lcheek_pts.add((i, j))

                # Same process ass mentioned above, but with right cheek
                if rcheek_area/lcheek_area > 0.5 and rcheek_path.contains_point((j, i)):
                    rcheek_pts.add((i, j))

    # Return all the points that are within 2 standard deviations of YUB values

    means, stds = get_stats(n_img, [forehead_pts, lcheek_pts, rcheek_pts])

    return [clean_data(n_img, forehead_pts, means, stds),
            clean_data(n_img, lcheek_pts, means, stds),
            clean_data(n_img, rcheek_pts, means, stds)]


# Return the mean and stdevs for YUB values within the image, for select points
def get_stats(n_img, arr):
    y, u, b = [], [], []
    for points in arr:
        for p0, p1 in points:
            temp = n_img[p0, p1]
            y.append(temp[0])
            u.append(temp[1])
            b.append(temp[2])

    means = (np.mean(y), np.mean(u), np.mean(b))
    stds = (np.std(y), np.std(u), np.std(b))

    return means, stds


# Given an image, means/stdevs, and points, return only the points within 2 stds
def clean_data(n_img, points, means, stds):
    print("pre-cleaning", len(points))

    y_mean, u_mean, b_mean = means
    y_std, u_std, b_std = stds

    y_range, u_range, b_range = (y_mean - 2*y_std, y_mean + 2*y_std), (u_mean - 2*u_std, u_mean + 2*u_std), (b_mean - 2*b_std, b_mean + 2*b_std)

    new_points = []
    for p0, p1 in points:
        temp = n_img[p0, p1]
        # Check if point is within given ranges -> if so, add to new set
        if (y_range[0] <= temp[0] <= y_range[1]) and (u_range[0] <= temp[1] <= u_range[1]) and (b_range[0] <= temp[2] <= b_range[1]):
            new_points.append((p0, p1))

    print("post-cleaning", len(new_points))
    return new_points


def display_points(n_img, points, n_name):
    invert = n_img.copy()
    i_h, i_w, i_c = n_img.shape
    for i in range(i_h):
        for j in range(i_w):
            if (i, j) in points:
                invert[i, j] = [0, 0, 0]
            else:
                n_img[i, j] = [0, 0, 0]

    cv2.imwrite(f"../results/{n_name}_IMAGE.jpg", n_img)
    cv2.imwrite(f"../results/{n_name}_INVERT.jpg", invert)


def calculate_color(n_imgYUB, arr):
    values = numpy.array([n_imgYUB[x, y] for x, y in arr])
    model = NMF(n_components=2, init='random', random_state=0, max_iter=500)
    W = model.fit_transform(values)
    H = model.components_
    print("values", values.shape)
    print("w", W.shape)
    print("h", H.shape)

    specular = sum(H[1]) > sum(H[0])
    diffuse = 1 - specular

    transformer = KernelPCA(kernel='poly')

    X_transformed = transformer.fit_transform([H[diffuse]])
    print("x_transformed", X_transformed)

    return


def process_image(n_img, n_name):
    imgRGB = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
    imgYUB = cv2.cvtColor(n_img, cv2.COLOR_BGR2YUV)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        points = get_points(imgYUB, results.multi_face_landmarks)
        for p in points[0:1]:
            calculate_color(imgYUB, p)
        # display_points(n_img, points, n_name)

    return imgYUB

name = "will_smith"
img = cv2.imread(f"../images/{name}.jpg")
img = process_image(img, name)

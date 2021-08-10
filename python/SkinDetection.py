import cv2
import mediapipe as mp
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon
from sklearn.decomposition import NMF, KernelPCA, PCA


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

                # Same process as mentioned above, but with right cheek
                if rcheek_area/lcheek_area > 0.5 and rcheek_path.contains_point((j, i)):
                    rcheek_pts.add((i, j))

    # Return all the points that are within 2 standard deviations of RGB values

    means, stds = get_stats(n_img, [forehead_pts, lcheek_pts, rcheek_pts])

    return [clean_data(n_img, forehead_pts, means, stds),
            clean_data(n_img, lcheek_pts, means, stds),
            clean_data(n_img, rcheek_pts, means, stds)]


# Return the mean and stdevs for RGB values within the image, for select points
def get_stats(n_img, arr):
    r, g, b = [], [], []

    for points in arr:
        for p0, p1 in points:
            temp = n_img[p0, p1]
            r.append(temp[0])
            g.append(temp[1])
            b.append(temp[2])

    means = (np.mean(r), np.mean(g), np.mean(b))
    stds = (np.std(r), np.std(g), np.std(b))

    return means, stds


# Given an image, means/stdevs, and points, return only the points within 2 stds
def clean_data(n_img, points, means, stds):
    print("pre-cleaning", len(points))

    r_mean, g_mean, b_mean = means
    r_std, g_std, b_std = stds

    r_range, g_range, b_range = (r_mean - 2*r_std, r_mean + 2*r_std), (g_mean - 2*g_std, g_mean + 2*g_std), (b_mean - 2*b_std, b_mean + 2*b_std)

    new_points = []
    for p0, p1 in points:
        temp = n_img[p0, p1]
        # Check if point is within given ranges -> if so, add to new set
        if (r_range[0] <= temp[0] <= r_range[1]) and (g_range[0] <= temp[1] <= g_range[1]) and (b_range[0] <= temp[2] <= b_range[1]):
            new_points.append((p0, p1))

    print("post-cleaning", len(new_points))
    return new_points


def display_points(n_img, points, n_name):
    s = set()
    for arr in points:
        for p in arr:
            s.add(p)

    image = n_img.copy()
    invert = n_img.copy()
    i_h, i_w, i_c = n_img.shape
    for i in range(i_h):
        for j in range(i_w):
            if (i, j) in s:
                invert[i, j] = [0, 0, 0]
            else:
                image[i, j] = [0, 0, 0]

    cv2.imwrite(f"../results/{n_name}_IMAGE.jpg", image)
    cv2.imwrite(f"../results/{n_name}_INVERT.jpg", invert)


def calculate_color(n_img, arr):

    values = np.array([n_img[x, y] for x, y in arr]) / 256
    model = NMF(n_components=2, init='nndsvd', random_state=0, max_iter=500)

    W = model.fit_transform(values)
    H = model.components_

    print("W", W.shape)
    print("H", H.shape, H)

    # values = np.rot90(values)
    # print(np.array([256 * sum(v)/len(v) for v in values]))


    # TODO: test zeroth element of each row of H vs. the whole rowsum
    specular = int(sum(H[1]) < sum(H[0]))
    diffuse = 1 - specular
    index = diffuse

    # H[specular] = [0] * 3
    values = np.dot(W, H)
    # values = W * H
    # print("NEW MULTIPLIED VALUES", values)

    # print("INDEX", index)
    # print("H0", sum(H[0]), H[0].shape)
    # print("H1", sum(H[1]), H[1].shape)
    X = np.rot90(np.array([H[index]]))
    values = np.rot90(values)
    # print("X", X.shape)
    print("values", values.shape, values)
    values = np.array([256 * sum(v)/len(v) for v in values])
    # transformer = KernelPCA(n_components=1, kernel="poly")
    # [x * 3] -> [n_c * 3]
    # transformer = transformer.fit(values)
    # values = transformer.fit_transform(values)
    # print("values", values.shape, values)
    print(values)
    print("---------------")

    return values


def process_image(n_img, n_name):
    imgRGB = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        patches = get_points(imgRGB, results.multi_face_landmarks)
        r_s, g_s, b_s = 0, 0, 0
        for p in patches:
            r, g, b = calculate_color(imgRGB, p)
            r_s += r
            g_s += g
            b_s += b
        print(r_s/3, g_s/3, b_s/3)
        display_points(n_img, patches, n_name)

    return imgRGB


name = "will_smith2"
img = cv2.imread(f"../images/{name}.jpg", flags=cv2.IMREAD_COLOR)
img = process_image(img, name)

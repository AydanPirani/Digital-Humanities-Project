import cv2
import mediapipe as mp
import numpy as np
from matplotlib.path import Path
from sklearn.decomposition import NMF, KernelPCA

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

# Landmarks for each point, obtained via https://github.com/google/mediapipe/issues/1615
FOREHEAD_POINTS = [251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 139, 70, 63, 105, 66, 107,
                   9, 336, 296, 334, 293, 300, 383, 368, 389]
LCHEEK_POINTS = [31, 35, 143, 116, 123, 147, 213, 192, 214, 212, 216, 206, 203, 36, 101, 119, 229, 228]
RCHEEK_POINTS = [261, 265, 372, 345, 352, 376, 433, 434, 432, 436, 426, 423, 266, 330, 348, 449, 448]

# USER INPUT VALUES, DEPENDENT ON TEST RUNS
POINTS_THRESHOLD = 20  # Minimum points needed in a skin patch for it to "count"
DISPLAY_POINTS = False  # If true, visualizes the points selected
USE_STDEVS = False # If true, only includes points that are within 2 RGB stdevs of all selected points

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

        # Array of all pixels in the given area
        forehead_pts, rcheek_pts, lcheek_pts = [], [], []

        # Iterate through all pixels in image, check if pixel in path, then add
        for i in range(i_h):
            for j in range(i_w):
                # Check if point in the given shape - if so, add to array
                if forehead_path.contains_point((j, i)):
                    forehead_pts.append((i, j))

                # Same process as mentioned above, but with left cheek
                if lcheek_path.contains_point((j, i)):
                    lcheek_pts.append((i, j))

                # Same process as mentioned above, but with right cheek
                if rcheek_path.contains_point((j, i)):
                    rcheek_pts.append((i, j))

    # Check if cheeks don't have enough points - if so, then array becomes nullified
    if len(forehead_pts) < POINTS_THRESHOLD:
        lcheek_pts = []

    if len(lcheek_pts) < POINTS_THRESHOLD:
        lcheek_pts = []

    if len(rcheek_pts) < POINTS_THRESHOLD:
        rcheek_pts = []

    if not USE_STDEVS:
        return [forehead_pts, lcheek_pts, rcheek_pts]

    # Return all the points that are within 2 standard deviations of RGB values
    means, stds = get_stats(n_img, [forehead_pts, lcheek_pts, rcheek_pts])

    return [clean_data(n_img, forehead_pts, means, stds),
            clean_data(n_img, lcheek_pts, means, stds),
            clean_data(n_img, rcheek_pts, means, stds)]


# Return the mean and stdevs for RGB values within the image, for select points
def get_stats(n_img, arr):
    r_vals, g_vals, b_vals = [], [], []

    for points in arr:
        for p0, p1 in points:
            temp = n_img[p0, p1]
            r_vals.append(temp[0])
            g_vals.append(temp[1])
            b_vals.append(temp[2])

    means = (np.mean(r_vals), np.mean(g_vals), np.mean(b_vals))
    stds = (np.std(r_vals), np.std(g_vals), np.std(b_vals))

    return means, stds


# Given an image, means/stdevs, and points, return only the points within 2 stds
def clean_data(n_img, points, means, stds):
    if len(points) == 0:
        print("no elements in face, returning []")
        return points

    r_mean, g_mean, b_mean = means
    r_std, g_std, b_std = stds

    r_range, g_range, b_range = (r_mean - 2 * r_std, r_mean + 2 * r_std), (g_mean - 2 * g_std, g_mean + 2 * g_std), (
        b_mean - 2 * b_std, b_mean + 2 * b_std)

    new_points = []
    for p0, p1 in points:
        temp = n_img[p0, p1]
        # Check if point is within given ranges -> if so, add to new set
        if (r_range[0] <= temp[0] <= r_range[1]) and (g_range[0] <= temp[1] <= g_range[1]) and (
                b_range[0] <= temp[2] <= b_range[1]):
            new_points.append((p0, p1))

    return new_points


# Given an array of points, draw the points on the provided image and create a copy
def display_points(n_img, points, n_name):
    s = set()
    for arr in points:
        s.update(arr)

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


# Calculate the perceived brightness of a single pixel, given RGB values, sourced from link below (ITU BT.709)
# https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color#596243
def calculate_luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# SREDS-sourced method for perceived brightness, assuming that a higher RGB leads to a brighter color
def approximate_luminance(r, g, b):
    return r + g + b


# Calculate average color of a skin patch, ASSUMING that the array has AT LEAST ONE valid point in it
def calculate_color(n_img, arr):
    values = np.array([n_img[x, y] for x, y in arr])

    NMF_model = NMF(n_components=2, init='nndsvda', random_state=0, max_iter=2000, tol=5e-3, l1_ratio=0.2)
    KPCA_model = KernelPCA(n_components=1, kernel="poly")

    # Results of NMF operation
    W = NMF_model.fit_transform(values)  # size N x 2
    H = NMF_model.components_  # size 2 x 3

    # Specular = row with a higher luminance (bool -> int casting)
    H0_lum, H1_lum = calculate_luminance(*H[0]), calculate_luminance(*H[1])
    specular = int(H1_lum > H0_lum)
    diffuse = 1 - specular

    # Eliminates all impacts of specular component
    specular_comp = np.copy(H[specular])
    H[specular] = [0, 0, 0]

    # Converts W, from N rows of length 2, to 2 rows of length N
    X = np.rot90(W, axes=(1, 0))

    transformed = KPCA_model.fit_transform(X)
    skin_color = np.multiply(transformed, H)  # Array of size 2 x 3, row[specular] = 0, 0, 0

    diffuse_comp = [abs(i) for i in skin_color[diffuse]]

    return diffuse_comp, specular_comp


# Wrapper function to work on the image
def process_image(n_img, n_name, display=False):
    imgRGB = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        # Pull skin patches given the image (using Google MP)
        patches = get_points(imgRGB, results.multi_face_landmarks)
        diff_r, diff_g, diff_b = 0, 0, 0
        spec_r, spec_g, spec_b = 0, 0, 0
        lum_diff = 0

        # Calculate average color and luminance difference of each skin patch
        for p in patches:
            diff, spec = calculate_color(imgRGB, p)
            d_r, d_g, d_b = diff
            s_r, s_g, s_b = spec

            diff_r += d_r
            diff_g += d_g
            diff_b += d_b

            spec_r += s_r
            spec_b += s_b
            spec_g += s_g

        spec_comp = np.array([spec_r, spec_g, spec_b]) / len(patches)
        diff_comp = np.array([diff_r, diff_g, diff_b]) / len(patches)

        print("specular component", spec_comp)
        print("diffuse component", diff_comp)

        if display:
            display_points(n_img, patches, n_name)

    return imgRGB


name = "morgan_freeman"
img = cv2.imread(f"../images/{name}.jpg", flags=cv2.IMREAD_COLOR)
process_image(img, name, DISPLAY_POINTS)

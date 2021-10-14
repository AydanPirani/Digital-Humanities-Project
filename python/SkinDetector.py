import cv2
import mediapipe as mp
import numpy as np
from matplotlib.path import Path
from sklearn.decomposition import NMF, KernelPCA


# Return the mean and stdevs for RGB values within the image, for select points
def get_stats(img, arr):
    r_vals, g_vals, b_vals = [], [], []

    for points in arr:
        for p0, p1 in points:
            temp = img[p0, p1]
            r_vals.append(temp[0])
            g_vals.append(temp[1])
            b_vals.append(temp[2])

    means = (np.mean(r_vals), np.mean(g_vals), np.mean(b_vals))
    stds = (np.std(r_vals), np.std(g_vals), np.std(b_vals))

    return means, stds


# Given an image, means/stdevs, and points, return only the points within 2 stds
def clean_data(img, points, means, stds):
    if len(points) == 0:
        print("no elements in face, returning []")
        return points

    r_mean, g_mean, b_mean = means
    r_std, g_std, b_std = stds

    # Ranges for which points should be included
    r_range = (r_mean - 2 * r_std, r_mean + 2 * r_std)
    g_range = (g_mean - 2 * g_std, g_mean + 2 * g_std)
    b_range = (b_mean - 2 * b_std, b_mean + 2 * b_std)

    new_points = []
    for p0, p1 in points:
        temp = img[p0, p1]
        # Check if point is within given ranges -> if so, add to new set
        if (r_range[0] <= temp[0] <= r_range[1]) and (g_range[0] <= temp[1] <= g_range[1]) and (
                b_range[0] <= temp[2] <= b_range[1]):
            new_points.append((p0, p1))

    return new_points


# Calculate the perceived brightness of a single pixel, given RGB values, sourced from link below (ITU BT.709)
# https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color#596243
def calculate_luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# SREDS-sourced method for perceived brightness, assuming that a higher RGB leads to a brighter color
def estimate_luminance(r, g, b):
    return r + g + b


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


class SkinDetector:
    # Landmarks for each point, obtained via https://github.com/google/mediapipe/issues/1615
    FOREHEAD_POINTS = [251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 139, 70, 63, 105, 66, 107,
                       9, 336, 296, 334, 293, 300, 383, 368, 389]
    LCHEEK_POINTS = [31, 35, 143, 116, 123, 147, 213, 192, 214, 212, 216, 206, 203, 36, 101, 119, 229, 228]
    RCHEEK_POINTS = [261, 265, 372, 345, 352, 376, 433, 434, 432, 436, 426, 423, 266, 330, 348, 449, 448]

    # Required values from MP library
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh

    def __init__(self):
        self.USE_STDEVS = None
        self.DISPLAY_POINTS = None
        self.POINTS_THRESHOLD = None

        self.faceMesh = None

        self.NMF_model = NMF(n_components=2, init='nndsvda', random_state=0, max_iter=2000, tol=5e-3, l1_ratio=0.2)
        self.KPCA_model = KernelPCA(n_components=1, kernel="poly")

    def generate_json(self, img_id, img_path):
        data = {"threshold": 20,
                "true": {"spec": {}, "diff": {}},
                "false": {"spec": {}, "diff": {}}
                }
        vals = SkinDetector.process(self,img_id, img_path, {"use_stdevs": False})

        spec, diff = vals["spec"], vals["diff"]

        data["true"]["spec"]["r"], data["true"]["spec"]["g"], data["true"]["spec"]["b"] = spec
        data["true"]["diff"]["r"], data["true"]["diff"]["g"], data["true"]["diff"]["b"] = diff

        data["true"]["spec"]["act_lum"] = calculate_luminance(*spec)
        data["true"]["spec"]["esp_lum"] = estimate_luminance(*spec)
        data["true"]["diff"]["act_lum"] = calculate_luminance(*diff)
        data["true"]["diff"]["esp_lum"] = estimate_luminance(*diff)


        vals = SkinDetector.process(self, img_id, img_path, {"use_stdevs": True})

        spec, diff = vals["spec"], vals["diff"]

        data["false"]["spec"]["r"], data["false"]["spec"]["g"], data["false"]["spec"]["b"] = spec
        data["false"]["diff"]["r"], data["false"]["diff"]["g"], data["false"]["diff"]["b"] = diff

        data["false"]["spec"]["act_lum"] = calculate_luminance(*spec)
        data["false"]["spec"]["esp_lum"] = estimate_luminance(*spec)
        data["false"]["diff"]["act_lum"] = calculate_luminance(*diff)
        data["false"]["diff"]["esp_lum"] = estimate_luminance(*diff)

        return data

    def process(self, img_id, img_path, params={}):
        self.POINTS_THRESHOLD = 20 if "points_threshold" not in params else params["points_threshold"]
        self.DISPLAY_POINTS = False if "display_points" not in params else params["display_points"]
        self.USE_STDEVS = False if "use_stdevs" not in params else params["use_stdevs"]
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1 if "max_faces" not in params else params["max_faces"])

        img = cv2.imread(img_path)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            # Pull skin patches given the image (using Google MP)
            patches = self.get_points(imgRGB, results.multi_face_landmarks)
            diff_r, diff_g, diff_b = 0, 0, 0
            spec_r, spec_g, spec_b = 0, 0, 0

            # Calculate average color and luminance difference of each skin patch
            for p in patches:
                diff, spec = self.calculate_color(imgRGB, p)
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

            for i in range(3):
                spec_comp[i] = round(spec_comp[i], 2)
                diff_comp[i] = round(diff_comp[i], 2)

            print("specular component", spec_comp)
            print("diffuse component", diff_comp)

            if self.DISPLAY_POINTS:
                display_points(img, patches, img_id)

            return {"spec": spec_comp, "diff": diff_comp}


    # Given an image and points to take measurements from, return "valid" points in the image
    def get_points(self, img, landmarks):
        i_h, i_w, i_c = img.shape
        for faceLms in landmarks[:1]:
            # List of all the landmark coordinates from the generated face
            forehead_landmarks = [(i_w * f.x, i_h * f.y) for f in [faceLms.landmark[i] for i in self.FOREHEAD_POINTS]]
            lcheek_landmarks = [(i_w * f.x, i_h * f.y) for f in [faceLms.landmark[i] for i in self.LCHEEK_POINTS]]
            rcheek_landmarks = [(i_w * f.x, i_h * f.y) for f in [faceLms.landmark[i] for i in self.RCHEEK_POINTS]]

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
        if len(forehead_pts) < self.POINTS_THRESHOLD:
            lcheek_pts = []

        if len(lcheek_pts) < self.POINTS_THRESHOLD:
            lcheek_pts = []

        if len(rcheek_pts) < self.POINTS_THRESHOLD:
            rcheek_pts = []

        if self.USE_STDEVS:
            # Return all the points that are within 2 standard deviations of RGB values
            means, stds = get_stats(img, [forehead_pts, lcheek_pts, rcheek_pts])
            return [clean_data(img, forehead_pts, means, stds),
                    clean_data(img, lcheek_pts, means, stds),
                    clean_data(img, rcheek_pts, means, stds)]
        else:
            return [forehead_pts, lcheek_pts, rcheek_pts]

    # Calculate average color of a skin patch, ASSUMING that the array has AT LEAST ONE valid point in it
    def calculate_color(self, img, arr):
        values = np.array([img[x, y] for x, y in arr])


        # Results of NMF operation
        W = self.NMF_model.fit_transform(values)  # size N x 2
        H = self.NMF_model.components_  # size 2 x 3

        # Specular = row with a higher luminance (bool -> int casting)
        H0_lum, H1_lum = calculate_luminance(*H[0]), calculate_luminance(*H[1])
        specular = int(H1_lum > H0_lum)
        diffuse = 1 - specular

        # Eliminates all impacts of specular component
        specular_comp = np.copy(H[specular])
        H[specular] = [0, 0, 0]

        # Converts W, from N rows of length 2, to 2 rows of length N
        X = np.rot90(W, axes=(1, 0))

        transformed = self.KPCA_model.fit_transform(X)
        skin_color = np.multiply(transformed, H)  # Array of size 2 x 3, row[specular] = 0, 0, 0

        diffuse_comp = [abs(i) for i in skin_color[diffuse]]

        return diffuse_comp, specular_comp

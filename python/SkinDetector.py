import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, KernelPCA

import python.utilities as utilities


class SkinDetector:
    # Required values from MP library
    mpDraw = mp.solutions.drawing_utils

    def __init__(self):

        self.diff = []
        self.patches = []

    def get_data(self, img_id, img_path, params):

        data = {}
        vals = SkinDetector.process(self, img_id, img_path, {"use_stdevs": False})

        spec, diff = vals["spec"], vals["diff"]

        data["spec"]["r"], data["true"]["spec"]["g"], data["true"]["spec"]["b"] = spec
        data["diff"]["r"], data["true"]["diff"]["g"], data["true"]["diff"]["b"] = diff

        data["spec"]["act_lum"] = utilities.calculate_luminance(*spec)
        data["spec"]["esp_lum"] = utilities.estimate_luminance(*spec)
        data["diff"]["act_lum"] = utilities.calculate_luminance(*diff)
        data["diff"]["esp_lum"] = utilities.estimate_luminance(*diff)

        return data

    def generate_json(self, img_id, img_path):
        data = {"threshold": 20,
                "true": self.get_data(img_id, img_path, {"use_stdevs": True}),
                "false": self.get_data(img_id, img_path, {"use_stdevs": False})
                }
        return data


    # TODO: convert JSON to CSV using get_data() method
    def generate_csv(self, img_id, img_path):
        cols = ["id", "true/spec/r", "true/spec/g", "true/spec/b", "true/spec/act_lum",
                "true/spec/esp_lum", "true/diff/r", "true/diff/g", "true/diff/b",
                "true/diff/act_lum", "true/diff/esp_lum", "false/spec/r", "false/spec/g",
                "false/spec/b", "false/spec/act_lum", "false/spec/esp_lum", "false/diff/r",
                "false/diff/g", "false/diff/b", "false/diff/act_lum", "false/diff/esp_lum"]
        data = [img_id]

        for i in [True, False]:

            vals = SkinDetector.process(self, img_id, img_path, {"use_stdevs": i})

            spec, diff = vals["spec"], vals["diff"]

            data.extend(spec)
            data.append(utilities.calculate_luminance(*spec))
            data.append(utilities.estimate_luminance(*spec))
            data.extend(diff)
            data.append(utilities.calculate_luminance(*diff))
            data.append(utilities.estimate_luminance(*diff))

        df = pd.DataFrame([data], columns=cols)
        df.to_csv(f"./results/data/{img_id}.csv", index=False)

    def process(self, img_id, img_path, params={}):
        points_threshold = params.get("points_threshold", 20)
        display_points = params.get("display_points", False)
        use_stdevs = params.get("use_stdevs", False)
        max_faces = params.get("max_faces", 1)

        faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_faces)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        results = faceMesh.process(img)

        if results.multi_face_landmarks:
            # Pull skin patches given the image (using Google MP)
            patches = self.get_points(img, results.multi_face_landmarks, points_threshold, use_stdevs)

            diff_comp = np.array([0, 0, 0], dtype=float)
            spec_comp = np.array([0, 0, 0], dtype=float)

            # Calculate average color and luminance difference of each skin patch
            for p in patches:
                patch_diff, patch_spec = self.calculate_color(img, p)

                diff_comp += patch_diff
                spec_comp += patch_spec

            diff_comp = np.around(diff_comp / len(patches), 3)
            spec_comp = np.around(spec_comp / len(patches), 3)

            self.diff = diff_comp

            if display_points:
                self.display_points(img, patches, img_id)

            return {"spec": spec_comp, "diff": diff_comp}

    # Given an image and points to take measurements from, return "valid" points in the image
    def get_points(self, img, landmarks, threshold, use_stdevs):

        # Should only happen once - generate patches
        if len(self.patches) == 0:
            self.patches = list(utilities.get_patches(img, landmarks))

        forehead_pts, lcheek_pts, rcheek_pts = self.patches

        # Check if cheeks don't have enough points - if so, then array becomes nullified
        if len(forehead_pts) < threshold:
            forehead_pts = []

        if len(lcheek_pts) < threshold:
            lcheek_pts = []

        if len(rcheek_pts) < threshold:
            rcheek_pts = []

        if use_stdevs:
            # Return all the points that are within 2 standard deviations of RGB values
            means, stds = utilities.get_stats(img, [forehead_pts, lcheek_pts, rcheek_pts])
            return [utilities.clean_data(img, forehead_pts, means, stds),
                    utilities.clean_data(img, lcheek_pts, means, stds),
                    utilities.clean_data(img, rcheek_pts, means, stds)]
        else:
            return [forehead_pts, lcheek_pts, rcheek_pts]

    # Calculate average color of a skin patch, ASSUMING that the array has AT LEAST ONE valid point in it
    def calculate_color(self, img, arr):
        values = np.array([img[x, y] for x, y in arr])

        NMF_model = NMF(n_components=2, init='nndsvda', random_state=0, max_iter=2000, tol=5e-3, l1_ratio=0.2)

        # Results of NMF operation
        W = NMF_model.fit_transform(values)  # size N x 2
        H = NMF_model.components_  # size 2 x 3

        # Specular = row with a higher luminance (bool -> int casting)
        H0_lum, H1_lum = utilities.calculate_luminance(*H[0]), utilities.calculate_luminance(*H[1])
        specular = int(H1_lum > H0_lum)
        diffuse = 1 - specular

        # Eliminates all impacts of specular component
        specular_comp = np.copy(H[specular])
        H[specular] = [0, 0, 0]

        # Converts W, from N rows of length 2, to 2 rows of length N
        X = np.rot90(W, axes=(1, 0))

        KPCA_model = KernelPCA(n_components=1, kernel="poly")

        transformed = KPCA_model.fit_transform(X)
        skin_color = np.multiply(transformed, H)  # Array of size 2 x 3, row[specular] = 0, 0, 0

        diffuse_comp = [abs(i) for i in skin_color[diffuse]]

        return diffuse_comp, specular_comp

    def display_points(self, n_img, points, n_name):
        # Given an array of points, draw the points on the provided image and create a copy
        s = set()
        for arr in points:
            s.update(arr)

        image = n_img.copy()
        invert = n_img.copy()
        diffuse = n_img.copy()
        i_h, i_w, i_c = n_img.shape

        for i in range(i_h):
            for j in range(i_w):
                if (i, j) in s:
                    invert[i, j] = [0, 0, 0]
                    diffuse[i, j] = self.diff.copy()
                else:
                    image[i, j] = [0, 0, 0]
                    diffuse[i, j] = [0, 0, 0]

        diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"./results/imgs/{n_name}_IMAGE.jpg", image)
        cv2.imwrite(f"./results/imgs/{n_name}_INVERT.jpg", invert)
        cv2.imwrite(f"./results/imgs/{n_name}_DIFFUSE.jpg", diffuse)

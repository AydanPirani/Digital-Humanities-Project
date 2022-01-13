import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, KernelPCA

import python.utilities as utilities


class SkinDetector:
    def __init__(self):
        self.diff = []
        self.patches = []

    def get_data(self, img_id, img_path, params):
        data = {"spec": {},
                "diff": {}
                }
        vals = SkinDetector.process(self, img_id, img_path, params)

        spec, diff = vals["spec"], vals["diff"]

        data["spec"]["r"], data["spec"]["g"], data["spec"]["b"] = spec
        data["diff"]["r"], data["diff"]["g"], data["diff"]["b"] = diff

        data["spec"]["act_lum"] = utilities.calculate_luminance(*spec)
        data["spec"]["est_lum"] = utilities.estimate_luminance(*spec)
        data["diff"]["act_lum"] = utilities.calculate_luminance(*diff)
        data["diff"]["est_lum"] = utilities.estimate_luminance(*diff)

        return data

    def generate_json(self, img_id, img_path):
        data = {"threshold": 20,
                "true": self.get_data(img_id, img_path, {"use_stdevs": True}),
                "false": self.get_data(img_id, img_path, {"use_stdevs": False})
                }
        return data


    def generate_csv(self, img_id, img_path):
        data = {"id": img_id,
                "true": self.get_data(img_id, img_path, {"use_stdevs": True}),
                "false": self.get_data(img_id, img_path, {"use_stdevs": False})
                }
        df = pd.json_normalize(data)
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
            print("78:", diff_comp, spec_comp)
            self.diff = diff_comp

            if display_points:
                utilities.display_points(img, patches, img_id, self.diff)

            return {"spec": spec_comp, "diff": diff_comp}

    # Given an image and points to take measurements from, return "valid" points in the image
    def get_points(self, img, landmarks, threshold, use_stdevs):
        # Should only happen once - generate patches
        if len(self.patches) == 0:
            self.patches = utilities.get_patches(img, landmarks)

        forehead_pts, lcheek_pts, rcheek_pts = utilities.clean_patches(img, self.patches, use_stdevs, threshold)
        return np.array([forehead_pts, lcheek_pts, rcheek_pts])


    # Calculate average color of a skin patch, ASSUMING that the array has AT LEAST ONE valid point in it
    def calculate_color(self, img, arr):
        values = np.array([img[y, x] for y, x in arr])

        NMF_model = NMF(n_components=2, init='nndsvda', random_state=0, max_iter=2000, tol=5e-3, l1_ratio=0.2)

        # Results of NMF operation
        W = NMF_model.fit_transform(values)  # size N x 2
        H = NMF_model.components_  # size 2 x 3

        # Specular = row with a higher luminance (bool -> int casting)
        H0_lum, H1_lum = utilities.calculate_luminance(*H[0]), utilities.calculate_luminance(*H[1])

        specular, diffuse = -1, -1
        if H0_lum > H1_lum:
            specular, diffuse = 0, 1
        else:
            specular, diffuse = 1, 0

        # Eliminates all impacts of specular component
        specular_comp = np.copy(H[specular])
        H[specular] = [0, 0, 0]

        # print(H)

        # Converts W, from N rows of length 2, to 2 rows of length N
        X = np.rot90(W, axes=(1, 0))

        KPCA_model = KernelPCA(n_components=1, kernel="poly")

        transformed = KPCA_model.fit_transform(X)
        skin_color = np.multiply(transformed, H)  # Array of size 2 x 3, row[specular] = 0, 0, 0
        print(skin_color)
        diffuse_comp = [abs(i) for i in skin_color[diffuse]]

        print("124:", diffuse_comp, specular_comp)
        return diffuse_comp, specular_comp



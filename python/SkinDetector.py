import os
import shutil

import cv2
import mediapipe as mp
import numpy as np
import pandas
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


    def process(self, img_id, img_path, output, params={}):
        points_threshold = params.get("points_threshold", 20)
        display_points = params.get("display_points", False)
        max_faces = params.get("max_faces", 7)

        faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_faces)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        results = faceMesh.process(img)

        image_diff_img = np.zeros(img.shape, dtype=np.uint8)
        image_spec_img = np.zeros(img.shape, dtype=np.uint8)
        image_patch_img = img.copy()
        image_invert_img = np.zeros(img.shape, dtype=np.uint8)

        num_faces = 0

        if results.multi_face_landmarks:
            os.chdir(f"{output}/{img_id}")
        else:
            return

        cols = ["id", "true.spec.r", "true.spec.g", "true.spec.b", "true.spec.act_lum",
                "true.spec.est_lum", "true.diff.r", "true.diff.g", "true.diff.b",
                "true.diff.act_lum", "true.diff.est_lum", "false.spec.r", "false.spec.g", "false.spec.b",
                "false.spec.act_lum", "false.spec.est_lum", "false.diff.r", "false.diff.g", "false.diff.b",
                "false.diff.act_lum", "false.diff.est_lum"]

        df = pandas.DataFrame(columns = cols)

        for face_landmarks in results.multi_face_landmarks:
            num_faces += 1
            data = [f"face{num_faces}"]
            shutil.rmtree(f"face{num_faces}", ignore_errors=True)
            os.mkdir(f"face{num_faces}")

            # pull skin patches given the image (using google mp)
            face = np.array([{"x": res.x, "y": res.y} for res in face_landmarks.landmark])

            for use_stdevs in [True, False]:
                patches = self.get_points(img, face, points_threshold, use_stdevs)

                diff_comp = np.array([0, 0, 0], dtype=float)
                spec_comp = np.array([0, 0, 0], dtype=float)

                # calculate average color and luminance difference of each skin patch
                for p in patches:
                    patch_diff, patch_spec = self.calculate_color(img, p)

                    diff_comp += patch_diff
                    spec_comp += patch_spec

                diff_comp = np.clip(np.around(diff_comp / len(patches), 3), 0, 255)
                spec_comp = np.clip(np.around(spec_comp / len(patches), 3), 0, 255)

                data.extend(spec_comp)
                data.append(utilities.calculate_luminance(*spec_comp))
                data.append(utilities.estimate_luminance(*spec_comp))

                data.extend(diff_comp)
                data.append(utilities.calculate_luminance(*diff_comp))
                data.append(utilities.estimate_luminance(*diff_comp))

                self.diff = diff_comp
                self.patches = []


                if display_points:
                    face_diff_img = np.zeros(img.shape, dtype=np.uint8)
                    face_spec_img = np.zeros(img.shape, dtype=np.uint8)
                    face_patch_img = img.copy()
                    face_invert_img = np.zeros(img.shape, dtype=np.uint8)
                    for p in patches:
                        for x, y in p:
                            pt = img[x, y].copy()
                            face_diff_img[x, y] = diff_comp
                            face_spec_img[x, y] = spec_comp
                            face_patch_img[x, y] = [0, 0, 0]
                            face_invert_img[x, y] = pt

                            image_diff_img[x, y] = diff_comp
                            image_spec_img[x, y] = spec_comp
                            image_patch_img[x, y] = [0, 0, 0]
                            image_invert_img[x, y] = pt

                    face_diff_img = cv2.cvtColor(face_diff_img, cv2.COLOR_RGB2BGR)
                    face_spec_img = cv2.cvtColor(face_spec_img, cv2.COLOR_RGB2BGR)
                    face_patch_img = cv2.cvtColor(face_patch_img, cv2.COLOR_RGB2BGR)
                    face_invert_img = cv2.cvtColor(face_invert_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"face{num_faces}/{img_id}_FACE{num_faces}_{use_stdevs}_DIFF.jpg", face_diff_img)
                    cv2.imwrite(f"face{num_faces}/{img_id}_FACE{num_faces}_{use_stdevs}_SPEC.jpg", face_spec_img)
                    cv2.imwrite(f"face{num_faces}/{img_id}_FACE{num_faces}_{use_stdevs}_PATCH.jpg", face_patch_img)
                    cv2.imwrite(f"face{num_faces}/{img_id}_FACE{num_faces}_{use_stdevs}_INVERT.jpg", face_invert_img)

            df.loc[len(df.index)] = data

        if display_points:
            image_diff_img = cv2.cvtColor(image_diff_img, cv2.COLOR_BGR2RGB)
            image_spec_img = cv2.cvtColor(image_spec_img, cv2.COLOR_BGR2RGB)
            image_patch_img = cv2.cvtColor(image_patch_img, cv2.COLOR_BGR2RGB)
            image_invert_img = cv2.cvtColor(image_invert_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{img_id}_{use_stdevs}_DIFF.jpg", image_diff_img)
            cv2.imwrite(f"{img_id}_{use_stdevs}_SPEC.jpg", image_spec_img)
            cv2.imwrite(f"{img_id}_{use_stdevs}_PATCH.jpg", image_patch_img)
            cv2.imwrite(f"{img_id}_{use_stdevs}_INVERT.jpg", image_invert_img)

        df.to_csv(f"{img_id}_results.csv", index=False)
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
        diffuse_comp = [abs(i) for i in skin_color[diffuse]]

        return diffuse_comp, specular_comp



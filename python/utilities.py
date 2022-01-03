import numpy as np
from matplotlib.path import Path

# Landmarks for each point, obtained via https://github.com/google/mediapipe/issues/1615
FOREHEAD_POINTS = set([251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 139, 70, 63, 105, 66, 107,
                       9, 336, 296, 334, 293, 300, 383, 368, 389])
LCHEEK_POINTS = set([31, 35, 143, 116, 123, 147, 213, 192, 214, 212, 216, 206, 203, 36, 101, 119, 229, 228])
RCHEEK_POINTS = set([261, 265, 372, 345, 352, 376, 433, 434, 432, 436, 426, 423, 266, 330, 348, 449, 448])

# Calculate the perceived brightness of a single pixel, given RGB values, sourced from link below (ITU BT.709)
# https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color#596243
def calculate_luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# SREDS-sourced method for perceived brightness, assuming that a higher RGB leads to a brighter color
def estimate_luminance(r, g, b):
    return (r + g + b) / 3

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


# inner function to get patches
def get_patches(img, landmarks):
    print("calling get patches!")
    i_h, i_w, i_c = img.shape
    for faceLms in landmarks[:1]:
        # List of all the landmark coordinates from the generated face

        x_left, x_right = float("inf"), -float("inf")
        y_up, y_down = float("inf"), -float("inf")

        forehead_landmarks, lcheek_landmarks, rcheek_landmarks = [], [], []

        for i in range(0, len(faceLms.landmark)):
            x, y = int(i_w * faceLms.landmark[i].x), int(i_h * faceLms.landmark[i].y)

            if i in FOREHEAD_POINTS:
                forehead_landmarks.append((x, y))

            if i in LCHEEK_POINTS:
                lcheek_landmarks.append((x, y))

            if i in RCHEEK_POINTS:
                rcheek_landmarks.append((x, y))

            if i in FOREHEAD_POINTS or i in LCHEEK_POINTS or i in RCHEEK_POINTS:
                x_left, x_right = min(x_left, x), max(x_right, x)
                y_up, y_down = min(y_up, y), max(y_down, y)

        # Generating MPL paths for each body part - used to iterate pixels
        forehead_path = Path(forehead_landmarks)
        lcheek_path = Path(lcheek_landmarks)
        rcheek_path = Path(rcheek_landmarks)

        # # Array of all pixels in the given area
        f_pts, r_pts, l_pts = [], [], []

        # Iterate through all pixels in image, check if pixel in path, then add
        for i in range(y_up, y_down + 1):
            for j in range(x_left, x_right + 1):
                # Check if point in the given shape - if so, add to array
                if forehead_path.contains_point((j, i)):
                    f_pts.append((i, j))

                # Same process as mentioned above, but with left cheek
                if lcheek_path.contains_point((j, i)):
                    l_pts.append((i, j))

                # Same process as mentioned above, but with right cheek
                if rcheek_path.contains_point((j, i)):
                    r_pts.append((i, j))

    return f_pts, l_pts, r_pts

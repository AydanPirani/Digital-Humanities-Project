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


def get_range(pt, std):
    return tuple(pt - 2 * std, pt + 2 * std)


# Check if a point falls within a particular range
def point_in_range(point, ranges):
    for i in range(len(point)):
        if point[i] < ranges[i][0] or point[i] > ranges[i][1]:
            # point lies outside the range
            return False
    return True


# Given an image, means/stdevs, and points, return only the points within 2 stds
def clean_data(img, points, means, stds):
    r_mean, g_mean, b_mean = means
    r_std, g_std, b_std = stds

    # Ranges for which points should be included
    r_range = get_range(r_mean, r_std)
    g_range = get_range(g_mean, g_std)
    b_range = get_range(b_mean, b_std)

    ranges = [r_range, g_range, b_range]

    # list of all points that fall within valid rgb ranges
    new_points = [(y, x) for y, x in points if point_in_range(img[y, x], ranges)]
    return new_points


# inner function to get patches
def get_patches(img, landmarks):
    i_h, i_w, i_c = img.shape
    for faceLms in landmarks[:1]:
        # List of all the landmark coordinates from the generated face

        x_left, x_right = float("inf"), -float("inf")
        y_up, y_down = float("inf"), -float("inf")

        forehead_landmarks, lcheek_landmarks, rcheek_landmarks = [], [], []

        for i in range(0, len(faceLms.landmark)):
            point = faceLms.landmark[i]
            img_width, img_height =  i_w, i_h

            x_coord = int(img_width * point.x)
            y_coord = int(img_height * point.y)

            if i in FOREHEAD_POINTS:
                forehead_landmarks.append((x_coord, y_coord))

            if i in LCHEEK_POINTS:
                lcheek_landmarks.append((x_coord, y_coord))

            if i in RCHEEK_POINTS:
                rcheek_landmarks.append((x_coord, y_coord))

            if i in FOREHEAD_POINTS or i in LCHEEK_POINTS or i in RCHEEK_POINTS:
                x_left, x_right = min(x_left, x_coord), max(x_right, x_coord)
                y_up, y_down = min(y_up, y_coord), max(y_down, y_coord)

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
                elif lcheek_path.contains_point((j, i)):
                    l_pts.append((i, j))
                elif rcheek_path.contains_point((j, i)):
                    r_pts.append((i, j))

    return f_pts, l_pts, r_pts

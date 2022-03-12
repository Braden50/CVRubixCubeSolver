import cv2
import numpy as np
from collections import defaultdict
from util import show_image
import time


def main():
    # file_path = 'C:/Users/jason/OneDrive/Pictures/Project/'
    file_path = "C:/Users/brade/OneDrive/Desktop/EE428/FinalPro/"
    file_name = "rube_test.jpg"

    img = cv2.imread(file_name, cv2.IMREAD_COLOR)


    gray_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.GaussianBlur(gray_image,(5,5),cv2.BORDER_DEFAULT)
    edges = cv2.Canny(gray_image, 10, 50, apertureSize=3)
    print('Size is: ', img.shape)

    show_image(edges)


    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * a)
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * a)

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)
    print(intersections)

    separated_intersections = separate_intersections(intersections)

    for point in separated_intersections:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img, (pt[0], pt[1] - length), (pt[0], pt[1] + length), (255, 0, 255),
                 1)  # vertical line
        cv2.line(img, (pt[0] - length, pt[1]), (pt[0] + length, pt[1]), (255, 0, 255), 1)

    show_image(img, 'Segmented Lines', wait=False)
    show_image(edges, 'Edges')


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections

def separate_intersections(intersections):
    """This function takes the average of intersected points in the region
    and makes sure there's only one point in the region
    input: intersections
    output: unique_intersections"""
    unique_intersections = [intersections[0]]
    radius = 80

    for intersection in intersections:
        unique_flag = True
        for point in unique_intersections:
            if intersection[0][0] > (point[0][0] - radius)  and intersection[0][0] < (point[0][0] + radius) \
            and intersection[0][1] > (point[0][1] - radius) and intersection[0][1] < (point[0][1] + radius):
                unique_flag = False
                # dont change false flag
        if unique_flag == True:
            #add intersection to unique intersections
            unique_intersections.append([[intersection[0][0], intersection[0][1]]])
    print("Unique Intersections are:", unique_intersections)
    return unique_intersections


if __name__ == "__main__":
    main()
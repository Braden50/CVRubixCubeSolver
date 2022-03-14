import cv2
import numpy as np
from collections import defaultdict
from util import show_image
import time
import math


def main():
    file_path = "C:/Users/brade/OneDrive/Desktop/EE428/FinalPro/"
    file_name = "rube_test.jpg"
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    gray_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    findIntercepts(img, gray_img, show=True)
    


def findIntercepts(img, gray_img, canny_min=5, canny_max=30, hough_thresh=150,
                   scale=20, show=False, separate=True):
    # file_path = 'C:/Users/jason/OneDrive/Pictures/Project/'

    height = img.shape[0]
    width = img.shape[1]

    needed_lines = 4  # how many lines are necessary

    # Hard coded parameter: how far from 90 and 0 degrees can the lines be
    rad_tolerance = 0.01
    gray_img = cv2.GaussianBlur(gray_img,(5,5),cv2.BORDER_DEFAULT)  # watch out for the mutation of original gray_img var
    
    # NOTE: The number of lines heavily affects efficiency of program
    edges = cv2.Canny(gray_img, canny_min, canny_max, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    try:
        if len(lines) < needed_lines:
            return None
    except:  # no lines
        return None
    line_set = []  # makes it so repeated lines are avoided
    new_lines = []
    for line in lines:
        for rho, theta in line:
            add = True
            if theta > math.pi or theta < 0:
                print("Weird theta val:", theta)
            # Assumption: theta is between 0 and pi
            # Assumes face lines are either vertical or horizontal
            if (math.pi / 2) + rad_tolerance < theta < math.pi - rad_tolerance or rad_tolerance < theta < (math.pi / 2) - rad_tolerance: 
                # print("Skipped:", theta)
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * a)
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * a)
            
            # PARAMETER: * how close the rubix cube needs to be *
            scale = 20   # how far apart squares can be, ex if scale is 8, edges must be img_len / 8 pixels apart
            x_tolerance = height / scale      
            y_tolerance = width / scale
            #print("TOLERANCES:", (x_tolerance, y_tolerance))
            if theta < rad_tolerance or math.pi - rad_tolerance < theta < math.pi + rad_tolerance:  # must be close to 0 slope
                #print("0:", (theta, x0, y0))
                for seen_line in line_set:    # seen_line = (rad, x, y)
                    #print("test: ", seen_line, (0, x0, y0))
                    if seen_line[0] == 0 and seen_line[1] - x_tolerance < x0 < seen_line[1] + x_tolerance:
                        add = False
                       # print("SEEN")
                        break
                line_set.append((0, x0, y0))
            else: # must be close to 90 degree slope
                # if its point has a similar y value to others, we don't need it. Can replace seen lines if needed, this decides on the first
               # print("90:", (theta, x0, y0)) 
                for seen_line in line_set:    # seen_line = (rad, x, y)
                    #print("test: ", seen_line, (90, x0, y0))
                    if seen_line[0] == 90 and seen_line[2] - y_tolerance < y0 < seen_line[2] + y_tolerance: 
                        add = False
                        #print("SEEN")
                        break
                line_set.append((90, x0, y0))

            if add:
                new_lines.append(line)
            if show:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("test", img)
    if len(new_lines) < needed_lines:
        return None
    segmented = segment_by_angle_kmeans(new_lines)
    intersections = segmented_intersections(segmented)
    if not intersections or len(intersections) < 4:
        return None

    separated_intersections = separate_intersections(intersections)

    for point in separated_intersections:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img, (pt[0], pt[1] - length), (pt[0], pt[1] + length), (255, 0, 255),
                 1)  # vertical line
        cv2.line(img, (pt[0] - length, pt[1]), (pt[0] + length, pt[1]), (255, 0, 255), 1)

    if show:
        show_image(img, 'Segmented Lines', wait=False)
        show_image(edges, 'Edges')
    if separate:
        return separated_intersections
    else:
        return intersections

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
    try:
        x0, y0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersection_point = intersection(line1, line2)
                    if intersection_point:
                        intersections.append(intersection_point)

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
    # print("Unique Intersections are:", unique_intersections)
    return unique_intersections


if __name__ == "__main__":
    main()

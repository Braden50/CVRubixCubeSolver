import cv2
import numpy as np
from findIntercepts import findIntercepts
from util import show_image
import numpy as np
import math

'''
Diction:
- Face = Entire rubix cube side
- Square = one of the (usually) 9 subsections of the face

Heuristics given many intersection points:
- 16 possible intersections can be detected, one for each corner of a square
    - Can't promise all of them will be detected, but we need some way to deduce the parallelogram
      for one square, from that we can assume the same projection and locate and extract the other 
      squares
- Rubix cube face size can be assumed to be [min_x_val:max_x_val, min_y_val:max_y_val]
    - From this, square size can be estimated to be [(min_x_val:max_x_val)/3, (min_y_val:max_y_val)/3]
    - This assumes an intersection can be found on all edges of the face (1 out of the 4 options, so not a bad assumption)
- All intersection points will be clustered into one of four x location clusters and y location clusters

'''

def main():
    radian_tolerance = 0.1
    file_name = "rube_test.jpg"
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    gray_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    boxes = findSquares(img, gray_img, show=True)
    print(boxes)
    


def findSquares(img, gray_img, canny_min=5, canny_max=30, hough_thresh=150,
                scale=20, show=False):
    original_img = img.copy()
    if img is None:
        raise Exception("No Image Uploaded")
    intercepts = findIntercepts(img, gray_img, canny_min,
                                canny_max, hough_thresh)
    if not intercepts:
        return None
    intercepts = fixIntercepts(intercepts)
    
    if len(intercepts) <= 3:
        print("Not enough intersections")
        return
    x_vals = [intercept[0] for intercept in intercepts]
    y_vals = [intercept[1] for intercept in intercepts]
    x_vals.sort()
    y_vals.sort()
    max_x = max(x_vals)
    min_x = min(x_vals)
    max_y = max(y_vals)
    min_y = min(y_vals)

    top_left = [min_x, min_y]
    top_right = [max_x, min_y]
    bottom_left = [min_x, max_y]
    bottom_right = [max_x, max_y]
    try:
        # face_vertical_slope_cartsian = (max_y - y_vals[-2]) / (max_x - min_x) # estimate, could also be min
        face_horizontal_slope_cartsian = (max_y - min_y), (min_x - x_vals[1])   # could also be max
        face_vertical_slope_rad = math.atan2((max_y - y_vals[-2]), (max_x - min_x)) # estimate, could also be min
        face_horizontal_slope_rad = math.atan2((max_y - min_y), (min_x - x_vals[1]))   # could also be max
    except ZeroDivisionError:
        print('up')
    
    # print(face_vertical_slope_cartsian, face_vertical_slope_cartsian)
    # print('\n')
    # print(face_vertical_slope_rad, face_vertical_slope_rad)

    if show:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", cv2.rectangle(original_img, top_left, bottom_right,(0,255,0),3))
        cv2.waitKey(0)
    print("HERE:", top_left, top_right)
    return (top_left, bottom_right)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def fixIntercepts(intersections):
    ''' [[[x, y]],]  -> [[x, y],] '''
    new_intersections = []

    for super_intersection in intersections:
        if len(super_intersection) != 1:
            raise Exception(super_intersection)
        for intersection in super_intersection:
            new_intersections.append(intersection)
    
    return new_intersections


if __name__=="__main__":
    main()


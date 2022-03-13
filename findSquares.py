import cv2
import numpy as np
from FindIntercepts import findIntercepts
from util import show_image
import numpy as np

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
    findSquares("rube_test2.jpg", show=True)


def findSquares(img_loc, show=False):
    img = cv2.imread(img_loc, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("No Image Uploaded")
    intercepts = fixIntercepts(findIntercepts(img_loc, show=True))
    x_vals = [intercept[0] for intercept in intercepts]
    y_vals = [intercept[1] for intercept in intercepts]

    max_x = max(x_vals)
    min_x = min(x_vals)
    max_y = max(y_vals)
    min_y = min(y_vals)

    top_left = [min_x, min_y]
    top_right = [max_x, min_y]
    bottom_left = [min_x, max_y]
    bottom_right = [max_x, max_y]

    # face_vertical_slope = 
    # face_horizontal_slope = 

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", cv2.rectangle(img, top_left, bottom_right,(0,255,0),3)) 
    # show_image(img, wait=False)
    
    cv2.waitKey(0)



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


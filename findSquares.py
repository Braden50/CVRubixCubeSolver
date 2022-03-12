import cv2
import numpy as np
from FindIntercepts import main as findIntercepts

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


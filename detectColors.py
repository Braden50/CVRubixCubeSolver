import cv2
from findSquares import findSquares

def main():
    file_name = "rube_test.jpg"
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    gray_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    boxes = findSquares(img, gray_img)
    detectColors(img, boxes)


def detectColors(img, boxes):
    ''' 
    Inputs:
        img: original image
        boxes: boxes for all 9 squares of the face, in order: left to right then top to bottom.
    Outputs:
        String defining each square color in the same order (ex: ['red', 'orange', ...])
    '''
    pass
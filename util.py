import cv2
import time


def show_image(img, window_name="window", normal=True, wait=True):
    ''' Simply pauses code to show image in window '''
    if normal:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
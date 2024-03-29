import cv2
import numpy as np
import kociemba as Cube
import time
import colorama
from findSquares import findSquares

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
RED = colorama.Fore.RED
MAGENTA=colorama.Fore.MAGENTA
colorama.init()
print(rf"{RED}          ______    __     __    _________    _________                                   ___ ___ ___                 ")
print(rf"{GREEN}         |  ____|  |  |   |  |  |  _____  |  | ________|                                 |   |   |   |            ")
print(rf"{GREEN}         | |       |  |   |  |  | |_____| |  | |____                                     |___|___|___|               ")
print(rf"{GREEN}         | |       |  |   |  |  |  _____  |  |  ____|                                    |   |   |   |            ")
print(rf"{GREEN}         | |____   |  |___|  |  | |_____| |  | |_______                                  |_N_|_I_|_C_|            ")
print(rf"{GREEN}         |______|  |_________|  |_________|  |_________|                                 |   |   |   |            ")
print(rf"{RED}                                                                                         |___|___|___|                     ")    
print(rf"{RED}                            _______    _________    __      ___      ___   ________    ________         ",end='\n')
print(rf"{GREEN}                           |  _____|  |   ___   |  |  |     \  \    /  /  |  ______|  |  _____ |    ")
print(rf"{GREEN}                           | |_____   |  |   |  |  |  |      \  \  /  /   | |____     |  ______|    ")
print(rf"{GREEN}                           |_____  |  |  |   |  |  |  |       \  \/  /    |  ____|    |   \  \        ")
print(rf"{GREEN}                            _____| |  |  |___|  |  |  |____    \    /     | |______   |  | \  \       ")
print(rf"{RED}                           |_______|  |_________|  |_______|    \__/      |________|  |__|  \__\      ")

time.sleep(2)
print("")
print("")
print(f"{MAGENTA}Please refer preview window for which side you have scanned and which color should be in centre on each side. ")

state=  {
            'up':['white','white','white','white','white','white','white','white','white',],
            'right':['white','white','white','white','white','white','white','white','white',],
            'front':['white','white','white','white','white','white','white','white','white',],
            'down':['white','white','white','white','white','white','white','white','white',],
            'left':['white','white','white','white','white','white','white','white','white',],
            'back':['white','white','white','white','white','white','white','white','white',]
        }

sign_conv={
            'green'  : 'F',
            'white'  : 'U',
            'blue'   : 'B',
            'red'    : 'R',
            'orange' : 'L',
            'yellow' : 'D'
          }

color = {
        'red'    : (0,0,255),
        'orange' : (0,165,255),
        'blue'   : (255,0,0),
        'green'  : (0,255,0),
        'white'  : (255,255,255),
        'yellow' : (0,255,255)
        }

stickers = {
        'main': [
            [200, 120], [300, 120], [400, 120],
            [200, 220], [300, 220], [400, 220],
            [200, 320], [300, 320], [400, 320]
        ],
        'current': [
            [20, 20], [54, 20], [88, 20],
            [20, 54], [54, 54], [88, 54],
            [20, 88], [54, 88], [88, 88]
        ],
        'preview': [
            [20, 130], [54, 130], [88, 130],
            [20, 164], [54, 164], [88, 164],
            [20, 198], [54, 198], [88, 198]
        ],
        'left': [
            [50, 280], [94, 280], [138, 280],
            [50, 324], [94, 324], [138, 324],
            [50, 368], [94, 368], [138, 368]
        ],
        'front': [
            [188, 280], [232, 280], [276, 280],
            [188, 324], [232, 324], [276, 324],
            [188, 368], [232, 368], [276, 368]
        ],
        'right': [
            [326, 280], [370, 280], [414, 280],
            [326, 324], [370, 324], [414, 324],
            [326, 368], [370, 368], [414, 368]
        ],
        'up': [
            [188, 128], [232, 128], [276, 128],
            [188, 172], [232, 172], [276, 172],
            [188, 216], [232, 216], [276, 216]
        ],
        'down': [
            [188, 434], [232, 434], [276, 434],
            [188, 478], [232, 478], [276, 478],
            [188, 522], [232, 522], [276, 522]
        ], 
        'back': [
            [464, 280], [508, 280], [552, 280],
            [464, 324], [508, 324], [552, 324],
            [464, 368], [508, 368], [552, 368]
        ],
           }

font = cv2.FONT_HERSHEY_SIMPLEX  
textPoints=  {
            'up':[['U',242, 202],['W',(255,255,255),260,208]],
            'right':[['R',380, 354],['R',(0,0,255),398,360]],
            'front':[['F',242, 354],['G',(0,255,0),260,360]],
            'down':[['D',242, 508],['Y',(0,255,255),260,514]],
            'left':[['L',104,354],['O',(0,165,255),122,360]],
            'back':[['B',518, 354],['B',(255,0,0),536,360]],
        }

check_state=[]
solution=[]
solved=False

cap=cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)


''' Parameters '''
# Default
DEFAULT_CANNY_MIN = 5
DEFAULT_CANNY_MAX = 30
DEFAULT_HOUGH_THRESH = 150
DEFAULT_SCALE = 20

parameters = {
    # name : [default, min, max]
    "Canny Min": [DEFAULT_CANNY_MIN, 0, 30],
    "Canny Max": [DEFAULT_CANNY_MAX, 1, 50],
    "Hough Threshold": [DEFAULT_HOUGH_THRESH, 30, 400],
    "Scale": [DEFAULT_SCALE, 3, 30]
}


# https://www.geeksforgeeks.org/python-opencv-gettrackbarpos-function/
def createTrackbars(window_name):
    for parameter in parameters:
        values = parameters[parameter]
        # Trackbars have to have a left value of 0, so all values are shifted for UI
        # print(1, cv2.getTrackbarPos(parameter, window_name))
        cv2.createTrackbar(parameter, window_name, values[0] - values[1],
                           values[2] - values[1], 
                           lambda new_val : new_val) 
        # print(2, cv2.getTrackbarPos(parameter, window_name))


def rotate(side):
    main=state[side]
    front=state['front']
    left=state['left']
    right=state['right']
    up=state['up']
    down=state['down']
    back=state['back']
    
    if side=='front':
        left[2],left[5],left[8],up[6],up[7],up[8],right[0],right[3],right[6],down[0],down[1],down[2]=down[0],down[1],down[2],left[8],left[5],left[2],up[6],up[7],up[8],right[6],right[3],right[0] 
    elif side=='up':
        left[0],left[1],left[2],back[0],back[1],back[2],right[0],right[1],right[2],front[0],front[1],front[2]=front[0],front[1],front[2],left[0],left[1],left[2],back[0],back[1],back[2],right[0],right[1],right[2]
    elif side=='down':
        left[6],left[7],left[8],back[6],back[7],back[8],right[6],right[7],right[8],front[6],front[7],front[8]=back[6],back[7],back[8],right[6],right[7],right[8],front[6],front[7],front[8],left[6],left[7],left[8]
    elif side=='back':
        left[0],left[3],left[6],up[0],up[1],up[2],right[2],right[5],right[8],down[6],down[7],down[8]=up[2],up[1],up[0],right[2],right[5],right[8],down[8],down[7],down[6],left[0],left[3],left[6] 
    elif side=='left':
        front[0],front[3],front[6],down[0],down[3],down[6],back[2],back[5],back[8],up[0],up[3],up[6]=up[0],up[3],up[6],front[0],front[3],front[6],down[6],down[3],down[0],back[8],back[5],back[2]
    elif side=='right':
        front[2],front[5],front[8],down[2],down[5],down[8],back[0],back[3],back[6],up[2],up[5],up[8]=down[2],down[5],down[8],back[6],back[3],back[0],up[8],up[5],up[2],front[2],front[5],front[8]

    main[0],main[1],main[2],main[3],main[4],main[5],main[6],main[7],main[8]=main[6],main[3],main[0],main[7],main[4],main[1],main[8],main[5],main[2]

def revrotate(side):
    main=state[side]
    front=state['front']
    left=state['left']
    right=state['right']
    up=state['up']
    down=state['down']
    back=state['back']
    
    if side=='front':
        left[2],left[5],left[8],up[6],up[7],up[8],right[0],right[3],right[6],down[0],down[1],down[2]=up[8],up[7],up[6],right[0],right[3],right[6],down[2],down[1],down[0],left[2],left[5],left[8]
    elif side=='up':
        left[0],left[1],left[2],back[0],back[1],back[2],right[0],right[1],right[2],front[0],front[1],front[2]=back[0],back[1],back[2],right[0],right[1],right[2],front[0],front[1],front[2],left[0],left[1],left[2]
    elif side=='down':
        left[6],left[7],left[8],back[6],back[7],back[8],right[6],right[7],right[8],front[6],front[7],front[8]=front[6],front[7],front[8],left[6],left[7],left[8],back[6],back[7],back[8],right[6],right[7],right[8]
    elif side=='back':
        left[0],left[3],left[6],up[0],up[1],up[2],right[2],right[5],right[8],down[6],down[7],down[8]=down[6],down[7],down[8],left[6],left[3],left[0],up[0],up[1],up[2],right[8],right[5],right[2] 
    elif side=='left':
        front[0],front[3],front[6],down[0],down[3],down[6],back[2],back[5],back[8],up[0],up[3],up[6]=down[0],down[3],down[6],back[8],back[5],back[2],up[0],up[3],up[6],front[0],front[3],front[6]
    elif side=='right':
        front[2],front[5],front[8],down[2],down[5],down[8],back[0],back[3],back[6],up[2],up[5],up[8]=up[2],up[5],up[8],front[2],front[5],front[8],down[8],down[5],down[2],back[6],back[3],back[0]

    main[0],main[1],main[2],main[3],main[4],main[5],main[6],main[7],main[8]=main[2],main[5],main[8],main[1],main[4],main[7],main[0],main[3],main[6]


def solve(state):
    raw=''
    for i in state:
        for j in state[i]:
            raw+=sign_conv[j]
    print("answer:",Cube.solve(raw))
    return Cube.solve(raw)

def color_detect(h,s,v):
    # print(h,s,v)
    if h < 5 and s>5:
        return 'red'
    elif h <10 and h>=3:
        return 'orange'
    elif h <= 25 and h>10:
        return 'yellow'
    elif h>=70 and h<= 85 and s>100 and v<180:
        return 'green'
    elif h <= 130 and s>70:
        return 'blue'
    elif h <= 100 and s<10 and v<200:
        return 'white'

    return 'white'

def draw_stickers(frame,stickers,name):
        for x,y in stickers[name]:
            cv2.rectangle(frame, (x,y), (x+30, y+30), (255,255,255), 2)

def draw_preview_stickers(frame,stickers):
        stick=['front','back','left','right','up','down']
        for name in stick:
            for x,y in stickers[name]:
                cv2.rectangle(frame, (x,y), (x+40, y+40), (255,255,255), 2)

def texton_preview_stickers(frame,stickers):
        stick=['front','back','left','right','up','down']
        for name in stick:
            for x,y in stickers[name]:
                sym,x1,y1=textPoints[name][0][0],textPoints[name][0][1],textPoints[name][0][2]
                cv2.putText(preview, sym, (x1,y1), font,1,(0, 0, 0), 1, cv2.LINE_AA)  
                sym,col,x1,y1=textPoints[name][1][0],textPoints[name][1][1],textPoints[name][1][2],textPoints[name][1][3]             
                cv2.putText(preview, sym, (x1,y1), font,0.5,col, 1, cv2.LINE_AA)  

def fill_stickers(frame,stickers,sides):    
    for side,colors in sides.items():
        num=0
        for x,y in stickers[side]:
            cv2.rectangle(frame,(x,y),(x+40,y+40),color[colors[num]],-1)
            num+=1

def process(operation):
    replace={
                "F":[rotate,'front'],
                "F2":[rotate,'front','front'],
                "F'":[revrotate,'front'],
                "U":[rotate,'up'],
                "U2":[rotate,'up','up'],
                "U'":[revrotate,'up'],
                "L":[rotate,'left'],
                "L2":[rotate,'left','left'],
                "L'":[revrotate,'left'],
                "R":[rotate,'right'],
                "R2":[rotate,'right','right'],
                "R'":[revrotate,'right'],
                "D":[rotate,'down'],
                "D2":[rotate,'down','down'],
                "D'":[revrotate,'down'],
                "B":[rotate,'back'],
                "B2":[rotate,'back','back'],
                "B'":[revrotate,'back']           
    }    
    a=0
    for i in operation:
        for j in range(len(replace[i])-1):
            replace[i][0](replace[i][j+1])
        cv2.putText(preview, i, (700,a+50), font,1,(0,255,0), 1, cv2.LINE_AA)  
        fill_stickers(preview,stickers,state)
        solution.append(preview)
        cv2.imshow('solution',preview)
        cv2.waitKey()
        cv2.putText(preview, i, (700,50), font,1,(0,0,0), 1, cv2.LINE_AA)  

def getMedianPixelValue(im):
    # gets median pixel value of three channel image. Is agnostic to image type (ex HSV vs RGB)
    try:
        x, y, z = cv2.split(im) # Split channels
    except:
        return None, None, None
    # Remove zeros
    x = x[x != 0]
    y = y[y != 0]
    z = z[z != 0]
    # median values
    x_median = np.median(x)
    y_median = np.median(y)
    z_median = np.median(z)
    return x_median,y_median,z_median


if __name__=='__main__':

    preview = np.zeros((700,800,3), np.uint8)
    upload = False
    capture = False
    
    counter = 0
    raw_img = np.zeros((512,512,3), np.uint8) # intiialize black image
    
    current_state = ['orange'] * 9
    while True:   
        hsv=[]
        if upload:
            file_name = "rube_test2.jpg"
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            # dim = (500, 500)
            # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            if not capture:
                ret,img=cap.read()
                raw_img = img.copy()
            else:
                img = raw_img.copy()

        # img=cv2.flip(img,1)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(frame.shape, dtype=np.uint8)
        

        # draw_stickers(img,stickers,'main')
        # draw_stickers(img,stickers,'current')
        # draw_preview_stickers(preview,stickers)

        fill_stickers(preview,stickers,state)
        texton_preview_stickers(preview,stickers)
        try:
            canny_min = cv2.getTrackbarPos('Canny Min', 'frame')
        except:  # trackbar not initialized
            createTrackbars('frame')
            cv2.createTrackbar("Capture", 'frame', 0, 1, lambda new_val : new_val) 

        # cv2.createButton("Capture",trigger_capture,None,cv2.QT_PUSH_BUTTON,1)


        # for i in range(9):
        #     hsv.append(frame[stickers['main'][i][1]+10][stickers['main'][i][0]+10])
        
        
        # Current state defines what colors populate the model for a face
        # if current_state == []:
        #     current_state = ['orange'] * 9
        # update view
        
            # current_state.append(color_name)
        # print(current_state)

        # TODO: get current state
        
        # parameters are trackbar + min since trackbar starts at 0
        canny_min = cv2.getTrackbarPos('Canny Min', 'frame') + parameters["Canny Min"][1]
        canny_max = cv2.getTrackbarPos('Canny Max', 'frame') + parameters["Canny Max"][1]
        hough_thresh = cv2.getTrackbarPos('Hough Threshold', 'frame') + parameters["Hough Threshold"][1]   
        scale = cv2.getTrackbarPos('Scale', 'frame') + parameters["Scale"][1]
        capture_val = cv2.getTrackbarPos('Capture', 'frame')
        capture = False if capture_val == 0 else True
        
        if counter % 100 == 0:
            print("Adjusted input params:", canny_min, canny_max, hough_thresh, scale, capture)

        squares = [(0, 0)]   # initalize square: (top_left, bottom_right) 
        squares = findSquares(img, gray_img, canny_min, canny_max, hough_thresh, scale)
        
        

        cv2.imshow('preview',preview)
        cv2.imshow('frame',img) # [0:500,0:500]
        
        if squares is not None:
            if len(squares) != 9:
                raise Exception("Not enough square subsections")
            for i, square in enumerate(squares):
                cv2.rectangle(img, square[0], square[1],(0,255,0),3) 
                h, s, v = getMedianPixelValue(frame[square[0][0]:square[1][0], square[0][1]:square[1][1]])  # extracts median pixel value from image
                if h is None:
                    detected_color = "orange"
                else:
                    detected_color = color_detect(h, s, v)
                current_state[i] = detected_color
            
        a=0
        for x,y in stickers['current']:
            color_name=current_state[a]  #color_detect(hsv[a][0],hsv[a][1],hsv[a][2])
            cv2.rectangle(img,(x,y),(x+30,y+30),color[color_name],-1)
            a+=1
        cv2.imshow('frame', img)


        counter += 1           


        

        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        elif k ==ord('u'):
            state['up']=current_state
            check_state.append('u')
        elif k ==ord('r'):
            check_state.append('r')
            state['right']=current_state
        elif k ==ord('l'):
            check_state.append('l')
            state['left']=current_state
        elif k ==ord('d'):
            check_state.append('d')
            state['down']=current_state       
        elif k ==ord('f'):
            check_state.append('f')
            state['front']=current_state       
        elif k ==ord('b'):
            check_state.append('b')
            state['back']=current_state       
        elif k == ord('\r'):
            # process(["R","R'"])
            if len(set(check_state))==6:    
                try:
                    solved=solve(state)
                    if solved:
                        operation=solved.split(' ')
                        process(operation)
                except:
                    print("error in side detection ,you may do not follow sequence or some color not detected well.Try again")
            else:
                print("all side are not scanned check other window for finding which left to be scanned?")
                print("left to scan:",6-len(set(check_state)))
        

    cv2.destroyAllWindows()
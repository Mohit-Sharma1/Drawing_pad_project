import cv2
import numpy as np
import math
import sounddevice
import threading
import mousecontrol

# camera setup
camera = cv2.VideoCapture(0) # 0 for built-in webcam, 1 for external webcam
camera.set(10,200)

# variables
frame_size_l = False
points = []
uncovered_point = True
points_logged = False
TL = (0, 0)
TR = (0, 0)
BR = (0, 0)
BL = (0, 0)
quad_logged = False
last_thumb_point = (9999, 9999)
thumb_pos_locked = False
mouseLock = True
top_line = 0
bottom_line = 0
volume = 0
smoothing_factor = 2


# draw circle
def draw_circle(event, x, y, flags, param):
    global points, points_logged, quad_logged

    if event == cv2.EVENT_LBUTTONDBLCLK:
        uncovered_point = True
        for point in points:
            if len(points) > 0:
                if abs(point[0]-x) <= 7 and abs(point[1]-y) <= 7:
                    uncovered_point = False
                    del points[points.index(point)]
                    points_logged = False
                    if len(points) <= 4:
                        quad_logged = False
                    break
        if uncovered_point:
            points.append((x, y))
            if len(points) <= 4:
                quad_logged = False
            points_logged = False


# round up to even num for mouse movement smoothing
def round_smoothing(num):
    global smoothing_factor
    return math.ceil(num / float(smoothing_factor)) * smoothing_factor


# print sound bars - dev testing function
def get_mic_input(indata, frames, time, status):
    global volume
    
    sensitivity = 40 # higher = more sensitive
    volume_norm = int(np.linalg.norm(indata)*sensitivity)
    volume = volume_norm
    print('|' * volume_norm)


# main thread
def mainthread():
    global camera, frame_size_l, points, uncovered_point, points_logged, TL, TR, BR, BL, quad_logged, last_thumb_point, thumb_pos_locked, mouseLock, smoothing_factor

    while camera.isOpened():
        # initialize camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.bilateralFilter(frame, 5, 50, 100) # smoothing filter

        # log frame size
        if not frame_size_l:
            print("frame size:", frame.shape[1], frame.shape[0])
            frame_size_l = True

        # draw paper points
        cv2.setMouseCallback('onal', draw_circle)
        for point in points:
            cv2.circle(frame, point, 5, (255,0,0), -1)
        
        if len(points) >= 1:
            print(frame[points[0][1]-10][points[0][1]-10])

        # log points
        if not points_logged:
            print("points:", points)
            points_logged = True

        cv2.imshow('original', frame)
        
        if len(points) >= 4:
            TL = points[0]
            TR = points[1]
            BR = points[2]
            BL = points[3]
        elif len(points) == 3:
            TL = points[0]
            TR = points[1]
            BR = points[2]
            BL = (0, 0)
        elif len(points) == 2:
            TL = points[0]
            TR = points[1]
            BR = (0, 0)
            BL = (0, 0)
        elif len(points) == 1:
            TL = points[0]
            TR = (0, 0)
            BR = (0, 0)
            BL = (0, 0)
        else:
            TL = (0, 0)
            TR = (0, 0)
            BR = (0, 0)
            BL = (0, 0)
        
        # log quad
        if not quad_logged:
            if len(points) >= 4:
                top_line = (TL[1] + TR[1])//2
                bottom_line = (BL[1] + BR[1])//2
            print("quad:", TL, TR, BR, BL)
            quad_logged = True

        # detect hand and pen/pencil tip
        try:
            kernel = np.ones((3, 3), np.uint8)

            # define frame input area
            roi = frame[100:500, 100:500]
            # cv2.rectangle(frame,(100,100),(500,500),(0,255,0),0)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # define skin colour range
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # extract skin colour image
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # dilate the hand to fill dark spots within
            mask = cv2.dilate(mask, kernel, iterations=4)

            # blur image
            mask = cv2.GaussianBlur(mask, (5, 5), 100)

            # find contours
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # find contour of max area (hand)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))

            # approximate contour
            epsilon = 0.0005*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # make convex hull around hand
            hull = cv2.convexHull(cnt)

            # define hull and hand area
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)

            # find area percentage not covered by hand in convex hull
            arearatio = ((areahull-areacnt)/areacnt)*100

            # find convex hull defects with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
            num_defects = 0
            thumb_point = (9999, 9999)

            # find num of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100,180)

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

                # calculate distance between point and convex hull
                d = (2*ar)/a

                # apply cosine law
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                # ignore angles and points that generally come due to noise
                if angle <= 90 and d > 30:
                    num_defects += 1
                    cv2.circle(roi, far, 3, [255,0,0], -1)

                # draw lines around hand
                cv2.line(roi, start, end, [0,255,0], 2)

                # identify thumb point
                x_trans = 73
                y_trans = 103
                if start[0] < thumb_point[0]:
                    if thumb_pos_locked:
                        if math.sqrt(abs(start[0]+x_trans-thumb_point[0])**2 + abs(start[1]+y_trans-thumb_point[1])**2) < 20:
                            thumb_point = (round_smoothing(start[0]+x_trans), round_smoothing(start[1]+y_trans))
                        else:
                            thumb_point = last_thumb_point
                    else:
                        thumb_point = (round_smoothing(start[0]+x_trans), round_smoothing(start[1]+y_trans))
                if end[0] < thumb_point[0]:
                    if thumb_pos_locked:
                        if math.sqrt(abs(end[0]+x_trans-thumb_point[0])**2 + abs(end[1]+y_trans-thumb_point[1])**2) < 20:
                            thumb_point = (round_smoothing(end[0]+x_trans), round_smoothing(end[1]+y_trans))
                        else:
                            thumb_point = last_thumb_point
                    else:
                        thumb_point = (round_smoothing(end[0]+x_trans), round_smoothing(end[1]+y_trans))
            
            last_thumb_point = thumb_point
            cv2.circle(frame, thumb_point, 5, (0,0,255), -1)

            num_defects += 1

            cv2.imshow('frame', frame)

        except Exception as e:
            if "max() arg" in str(e):
                print("...")
            else:
                print("Exception:", e)

        # move and click mouse
        if not mouseLock:
            mousecontrol.mouse_move(int((1920/int(frame.shape[1]))*int(thumb_point[0])), int((1080/int(frame.shape[0]))*(frame.shape[0]-int(thumb_point[1]))))
            if volume > 0:
                mousecontrol.mouse_down()
            else:
                mousecontrol.mouse_up()

        # keyboard press functions
        k = cv2.waitKey(20) & 0xFF
        if k == 27: # press ESC to exit
            camera.release()
            cv2.destroyAllWindows()
            break
        elif k == ord('a'): # press a to dynamically lock pen/pencil tip to thumb position area
            if not thumb_pos_locked:
                thumb_pos_locked = True
            else:
                thumb_pos_locked = False
        elif k == ord('b'):
            if not mouseLock: # press b to turn off/on mouse control
                mouseLock = True
            else:
                mouseLock = False
        elif k == ord('c'):
            if smoothing_factor > 0:
                smoothing_factor -= 1
                print("smoothing_factor", smoothing_factor)
        elif k == ord('d'):
            if smoothing_factor < 10:
                smoothing_factor += 1
                print("smoothing_factor", smoothing_factor)


# sound thread
def soundthread():
    with sounddevice.InputStream(callback=get_mic_input):
        sounddevice.sleep(60*60*1000)


# create threads
t1 = threading.Thread(target=mainthread, name='t1')
t2 = threading.Thread(target=soundthread, name='t2')

# start threads
t1.start()
t2.start()

# wait for all threads finish
t1.join()
t2.join()
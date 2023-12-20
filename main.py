import cv2
from viola_jones import viola_jones

def main():
    # KLT parameters
    # params for corner detection
    feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))
    
    camera = cv2.VideoCapture(0)
    while True:
        _,frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = viola_jones(gray)
        
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            frame[y:y+height, x:x+width, 0] = gray[y:y+height, x:x+width]
            frame[y:y+height, x:x+width, 1] = gray[y:y+height, x:x+width]
            frame[y:y+height, x:x+width, 2] = gray[y:y+height, x:x+width]
        
        cv2.imshow("faces", frame)

        if (cv2.waitKey(1) == ord("q")):
            break
        
    camera.release()
    cv2.destroyAllWindows()


main()
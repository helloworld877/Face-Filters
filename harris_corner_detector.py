import cv2
import numpy as np
# sobel kernels

Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sy = Sx.T

# Gaussian kernel

G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


# Corner Resposne Function


def corner_response(image, k, Sx=Sx, Sy=Sy):
    # computing first derivatives
    dx = cv2.filter2D(image, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(image, ddepth=-1, kernel=Sy)

    # Gaussian Filter
    A = cv2.filter2D(dx * dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy * dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx * dy, ddepth=-1, kernel=G)

    # Compute Corner response at all pixels
    return (A * B - (C * C)) - k * (A + B) * (A + B)


# harris corner Function
k = 0.05


def get_harris_corners(image, k=k):
    # compute corner response
    R = corner_response(image, k)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(R > 1e-2))
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    return cv2.cornerSubPix(image, np.float32(centroids), (9, 9), (-1, -1), criteria)


# get camera footage
camera = cv2.VideoCapture(0)
while True:
    _, frame = camera.read()
    cv2.imshow("faces", frame)

    # Get corners
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = get_harris_corners(gray_image, 0.00001)
    image_out = np.copy(frame)
    for x, y in corners:
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        cv2.circle(image_out, (x, y), radius=3,
                   color=(0, 255, 0, 255), thickness=-1)
    cv2.imshow("output", image_out)

    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

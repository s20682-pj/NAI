"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Python to detect faces, when finds red color it will put scope in the middle of the face

System requirements:
- Python 3.10
- Numpy
- OpenCV
"""

import cv2
import numpy as np

"""
Define a video capture object
"""
vid = cv2.VideoCapture(0)

face_filter = cv2.CascadeClassifier("face_detector.xml")

"""
Check if there is a face in the video
"""
if face_filter.empty():
    raise IOError("Where is my filter?!?!!")

while True:
    """
    Capture the video frame by frame
    """
    ret, frame = vid.read()

    """
    Change video for another color scale
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    face_rects = face_filter.detectMultiScale(gray_frame, 1.3, 5)

    """
    Define color range to find red
    """
    lower = np.array([155, 25, 0])
    upper = np.array([190, 255, 255])

    """
    Create mask, put on video to show only red things
    """
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    """
    If there are any white pixels on mask, sum will be > 0
    """
    hasRed = np.sum(mask)

    if hasRed > 0:
        for (x, y, w, h) in face_rects:
            center_coordinates = x + w // 2, y + h // 2
            cv2.circle(result, center_coordinates, 10, (0, 0, 255), 20)
    else:
        print("There is no red color")

    """
    Show original video and a one with mask
    """
    cv2.imshow("me", frame)
    cv2.imshow('result', result)

    """
    Close the video if ESC button will be clicked
    """
    if cv2.waitKey(1) == 27:  # Esc button
        break

"""
After the loop release the cap object
"""
vid.release()
"""
Destroy all the windows
"""
cv2.destroyAllWindows()
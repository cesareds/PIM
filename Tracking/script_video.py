import cv2
import numpy as np

# Load the video and template image
video_path = 'Video.mp4'
template_path = 'Bola.png'
cap = cv2.VideoCapture(video_path)
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

if template is None:
    print("Error: Could not load template image.")
    exit()

# Get the dimensions of the template
w, h = template.shape[::-1]

# List of methods to use for template matching
methods = [
    'cv2.TM_CCOEFF',
    'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR',
    'cv2.TM_CCORR_NORMED',
    'cv2.TM_SQDIFF',
    'cv2.TM_SQDIFF_NORMED'
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for method_name in methods:
        method = eval(method_name)

        # Apply template matching
        res = cv2.matchTemplate(gray_frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # For SQDIFF and SQDIFF_NORMED, the best match is the minimum value
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw a rectangle around the matched region
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # Display the method name above the rectangle
        text_position = (top_left[0], top_left[1] - 10)
        cv2.putText(frame, method_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Show the result
        cv2.imshow('Template Matching', frame)

        # Wait for a short duration or until 'q' is pressed to move to the next method
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Display each method result for 1 second
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Initialize a dictionary to store the min and max values for each method
results = {method: {'min_vals': [], 'max_vals': []} for method in methods}

frame_index = 0
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

        # Save min and max values
        results[method_name]['min_vals'].append(min_val)
        results[method_name]['max_vals'].append(max_val)

    frame_index += 1

cap.release()

# Create a DataFrame to store the results
df = pd.DataFrame()
for method_name in methods:
    df[method_name + '_min'] = results[method_name]['min_vals']
    df[method_name + '_max'] = results[method_name]['max_vals']

# Plot the results using matplotlib
plt.figure(figsize=(15, 10))
for i, method_name in enumerate(methods):
    plt.subplot(3, 2, i + 1)
    plt.plot(results[method_name]['min_vals'], label='Min Values', color='b')
    plt.plot(results[method_name]['max_vals'], label='Max Values', color='r')
    plt.title(f'{method_name} Results')
    plt.xlabel('Frame Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig('all_methods_results.png')
plt.show()

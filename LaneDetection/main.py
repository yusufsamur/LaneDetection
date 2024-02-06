import math
import cv2
import numpy as np
import matplotlib.pylab as plt
from helpfunctions import *

cap = cv2.VideoCapture("./test_video_lane.mp4")
width = cap.get(3)  # width
height = cap.get(4)  # height
fps = int(cap.get(cv2.CAP_PROP_FPS))

milisec_for_frame = 1000 // fps

canny_low = 200
canny_high = 300

region_ratios = [
    (0.1, 1),
    (0.45, 0.75),
    (0.55, 0.75),
    (0.7, 1),
]  # roi için width height değerlerine oranlı köşe koordinatları (videoya göre ayarlanmalıdır)
region_points = [
    (int(ratio[0] * width), int(ratio[1] * height)) for ratio in region_ratios
]


hough_rho = 2
hough_theta = math.pi / 180
hough_threshold = 20
hough_min_line_len = 15
hough_max_line_gap = 40

theta_threshold = 15 * math.pi / 180
left_theta = 140 * math.pi / 180
right_theta = -140 * math.pi / 180
x_threshold = 0.3


region_lines = []
line_coords = []
line_coords.extend(region_points[-1])
line_coords.extend(region_points[0])
region_lines.append([line_coords])

for i in range(1, len(region_points)):
    line_coords = []
    line_coords.extend(region_points[i - 1])
    line_coords.extend(region_points[i])
    region_lines.append([line_coords])

vertices = np.array([region_points], dtype=np.int32)


while cap.isOpened():
    ret, frame = cap.read()

    gray_frame = grayscale(frame)  # kolay işlem yapılması için gray scale'e çevirdik
    blur_frame = gaussian_blur(gray_frame, 5)  # blur uyguladık
    canny_frame = canny(blur_frame, canny_low, canny_high)  # canny edge detection
    masked_edge_frame = region_of_interest(canny_frame, vertices)  # roi
    lines = hough_lines(
        masked_edge_frame,
        hough_rho,
        hough_theta,
        hough_threshold,
        hough_min_line_len,
        hough_max_line_gap,
    )

    averaged_lines = average_lines(
        lines,
        theta_threshold,
        left_theta,
        right_theta,
        x_threshold * width,
        x_threshold * width,
        (1 - x_threshold) * width,
        region_points[1][1],
        region_points[0][1],
    )
    line_frame = draw_lines(masked_edge_frame, averaged_lines, [255, 0, 0], 10)
    line2_frame = draw_lines(masked_edge_frame, lines, [0, 0, 255])
    result = weighted_img(line2_frame, weighted_img(line_frame, frame))
    # result = weighted_img(line_frame, frame)
    # result = weighted_img(line2_frame,frame)

    cv2.imshow("Lane Detection", result)
    if cv2.waitKey(milisec_for_frame) == 27:
        break


cap.release()
cv2.destroyAllWindows()

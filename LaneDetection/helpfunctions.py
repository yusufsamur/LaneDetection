import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def average_lines(
    lines,
    theta_threshold,
    left_theta_exp,
    right_theta_exp,
    x_threshold,
    left_x_exp,
    right_x_exp,
    region_top,
    region_bottom,
):
    """
    Bu fonksiyon Hough Dönüşümünden gelen çizgileri alır ve sağ ve solda birer çizgi olmak üzere iki çizgi haline getirir.
    Bu işlemi eğimlerin ve noktaların ortalamasını alarak yapmaktadır.
    """

    left_theta_sum = 0
    right_theta_sum = 0
    left_theta_count = 0
    right_theta_count = 0
    left_x_sum = 0
    right_x_sum = 0
    left_x_count = 0
    right_x_count = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            theta = math.atan2(y1 - y2, x1 - x2)
            bottom_x = line_x(theta, x1, y1, region_bottom)
            length = line_length(x1, y1, x2, y2)
            if (
                (not y1 - y2 == 0)
                and (
                    (
                        math.fabs(bottom_x - left_x_exp) < x_threshold
                        or math.fabs(bottom_x - right_x_exp) < x_threshold
                    )
                )
                and (
                    math.fabs(theta - left_theta_exp) < theta_threshold
                    or math.fabs(theta - right_theta_exp) < theta_threshold
                )
            ):
                if theta > 0:
                    left_theta_sum += theta * length
                    left_theta_count += length
                    left_x_sum += bottom_x * length
                    left_x_count += length
                else:
                    right_theta_sum += theta * length
                    right_theta_count += length
                    right_x_sum += bottom_x * length
                    right_x_count += length
    left_theta = (
        left_theta_exp if left_theta_count == 0 else left_theta_sum / left_theta_count
    )
    right_theta = (
        right_theta_exp
        if right_theta_count == 0
        else right_theta_sum / right_theta_count
    )
    left_x = left_x_exp if left_x_count == 0 else left_x_sum / left_x_count
    right_x = right_x_exp if right_x_count == 0 else right_x_sum / right_x_count

    # Sol ortalama theta, sağ ortalama theta ve iki tarafın geçtiği
    # noktaları bulduk. Artık çizgileri hesaplayabiliriz.

    left_x2 = line_x(left_theta, left_x, region_bottom, region_top)
    right_x2 = line_x(right_theta, right_x, region_bottom, region_top)
    left_line = [int(left_x), region_bottom, int(left_x2), region_top]
    right_line = [int(right_x), region_bottom, int(right_x2), region_top]
    return [[left_line, right_line]]

def line_length(x1, y1, x2, y2):
    # çizginin uzunluğunu ölçer

    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def line_x(theta, x, y, y1):
    # y = y1 için (theta ve (x, y)) bilgilerini bildiğimiz çizginin x değerini veriyor

    tangent = math.tan(theta)
    x1 = x - (y - y1) / tangent
    return x1

"""
hough_lines fonksiyonundan gelen çizgi bilgilerinin üzerinde değişiklik yaptıktan sonra çizgileri
çizdirmek için hough_lines ve draw_lines fonksiyonlarını düzenlemek gerekiyor.  
"""

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    return lines

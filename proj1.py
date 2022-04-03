import cv2
import numpy as np
import argparse
#import sys
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from math import sqrt, degrees, atan, atan2, pow
import math
import requests
import os


#creating a mask for yellow color
def createYellowMask(frame):
    lower_range = np.uint8([10, 50, 50])
    upper_range = np.uint8([30, 255, 255])
    mask = cv2.inRange(frame, lower_range, upper_range)
    return mask

#creating a mask for white color
def createWhiteMask(frame):
    lower_range = np.uint8([0, 190, 0])
    upper_range = np.uint8([180, 255, 255])
    mask = cv2.inRange(frame, lower_range, upper_range)
    return mask


#creating a combined mask of white and yellow
def get_filtered_colors(frame):
    img_hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    yellow = createYellowMask(img_hls)
    white = createWhiteMask(img_hls)
    combined = cv2.bitwise_or(white, yellow)
    masked_image = cv2.bitwise_and(frame, frame, mask=combined)
    return masked_image

#getting the contour
def get_contour(frame, max_area):
    contour, hierarchy = cv2.findContours(np.copy(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_filtered = []
    for c in contour:
        if 100 <= cv2.contourArea(c) <= max_area:
            contour_filtered.append(c)
    image = cv2.fillPoly(frame, pts=contour_filtered, color=(255, 255, 255))
    return image

#dialation technique on mask
def maskDilation(image, kernel_size=3, iterations=1):
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8), iterations)

#reducing the noise
def noiseReduction(frame):
    img = cv2.morphologyEx(np.copy(frame), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return img

#extracting gray color
def grayExtraction(frame):
    frame_copy = np.copy(frame)
    blur = cv2.GaussianBlur(frame_copy, (15, 15), cv2.BORDER_CONSTANT)
    hls = cv2.cvtColor(blur, cv2.COLOR_RGB2HLS)

    # Gray color mask1
    lower_m1 = np.uint8([0, 25, 0])
    upper_m1 = np.uint8([35, 180, 25])
    mask1 = cv2.inRange(hls, lower_m1, upper_m1)

    # Gray mask2 threshold
    lower_m2 = np.uint8([102, 25, 0])
    upper_m2 = np.uint8([180, 180, 25])
    mask2 = cv2.inRange(hls, lower_m2, upper_m2)
    gray_mask = cv2.bitwise_or(mask1, mask2)
    return gray_mask


def grayFilter(frame):
    gray_mask = grayExtraction(frame)
    blur_gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), cv2.BORDER_CONSTANT)
    contour_gray_mask = get_contour(blur_gray_mask, 100000)
    intense_gray_mask = cv2.GaussianBlur(contour_gray_mask, (5, 5), cv2.BORDER_CONSTANT)
    dilated_gray_mask = maskDilation(intense_gray_mask, 5)
    masked_image = cv2.bitwise_and(frame, frame, mask=dilated_gray_mask)
    return masked_image


# canny edge detection
def edgeDetection(frame):
    img = cv2.Canny(frame, 230, 255, apertureSize=7)
    return img


#detecting stop sign for bonus marks
def detect_stop_sign(frame, formatted_image):
    if not os.path.exists('cascade_stop_sign.xml'):
        url = 'https://raw.githubusercontent.com/agrawalvivek2017/ExtraFiles/main/cascade_stop_sign.xml'
        r = requests.get(url, allow_redirects=True)
        open('cascade_stop_sign.xml', 'wb').write(r.content)
    stop_sign_classifier = cv2.CascadeClassifier('cascade_stop_sign.xml')
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = stop_sign_classifier.detectMultiScale(gray_img, 1.3, 5)
    stop_count = 0
    for (x, y, w, h) in stop_sign_scaled:
        x1=(x+x+w)//2
        y1=(y+y+h)//2
        radius=int(sqrt((x-x1)**2 + (y-y1)**2))
        stop_sign_circle=cv2.circle(frame, (x1,y1), radius, (0, 255, 255), 3)
        stop_count = stop_count + 1
        cv2.putText(img=stop_sign_circle, text="Stop Sign", org=(x, y + h + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_4)
    if stop_count > 0:
        showImage(formatted_image)
    print('Total stop signs detected in image is:', stop_count)



def slope(x1, y1, x2, y2):
    if x2 == x1:
        return float('inf')
    return (y2 - y1) / (x2 - x1)


def intercept(slope, x1, y1):
    return y1 - (slope * x1)

#counting hough lines
def count_lines(frame, hough_lines):
    updated_hough_lines = process_lines(hough_lines)
    sq_hough_lines = np.squeeze(updated_hough_lines, axis=1)
    slopes = []
    intercepts = []
    merged_lines = []
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        line_length = sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
        line_slope = slope(x1, y1, x2, y2)
        line_intercept = intercept(line_slope, x1, y1)
        angle = abs(degrees(atan(line_slope)))
        if all((30 <= angle <= 89, line_length > 50)):
            merged_lines.append((x1, y1, x2, y2))
            slopes.append(line_slope)
            intercepts.append(line_intercept)
    for line in merged_lines:
        (x1, y1, x2, y2) = line
        cv2.line(frame, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=8, lineType=cv2.FILLED)
    return frame, len(merged_lines)

#defining a region of interest
def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygon=np.array([
      [(0, height),(width,height),(width,0.2*height),(0,0.2*height)]
    ], dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    #computed bitwise & of both images so it masks the canny image on area of interest
    masked_image=cv2.bitwise_and(image, mask)
    return masked_image

#hough transformation
def hough_transform(frame, masked_image):
    hough_lines = cv2.HoughLinesP(masked_image, rho=1, theta=np.pi / 180, threshold=110, minLineLength=150, maxLineGap=30)
    if hough_lines is None:
        return frame, 0
    hough_lines_processed = process_lines(hough_lines)
    hough_lines_squeezed = np.squeeze(hough_lines_processed, axis=1)
    slopes = []
    intercepts = []
    merged_lines = []
    for line in hough_lines_squeezed:
        x1, y1, x2, y2 = line
        line_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_slope = slope(x1, y1, x2, y2)
        line_intercept = intercept(line_slope, x1, y1)
        angle = abs(degrees(atan(line_slope)))
        if all((15 <= angle <= 89, line_length > 50)):
            merged_lines.append((x1, y1, x2, y2))
            slopes.append(line_slope)
            intercepts.append(line_intercept)
    for line in merged_lines:
        (x1, y1, x2, y2) = line
        cv2.line(frame, (x1, y1), (x2, y2), color=[255, 0, 100], thickness=6, lineType=cv2.FILLED)
    return frame, len(merged_lines)


#resizing all the images to one size
def resize_image(frame):
    return cv2.resize(np.copy(frame), (600, 480), interpolation=cv2.INTER_AREA)

#making a function to show image using matplot
def showImage(frame):
    plt.imshow(frame)
    plt.show()

#detecting the lane function
def detect_lane(frame):
    resized_image = resize_image(frame)
    gray_image = grayFilter(resized_image)
    filtered_image = get_filtered_colors(gray_image)
    canny_img = edgeDetection(filtered_image)
    noise_reduced_img = noiseReduction(canny_img)
    contour_img = get_contour(noise_reduced_img, 10000000000)
    masked_canny_img = edgeDetection(contour_img)
    dilated_image = maskDilation(masked_canny_img, kernel_size=2, iterations=1)
    dilated_image=region_of_interest(dilated_image)
    new_image, line_count = hough_transform(resized_image, dilated_image)
    plt.imshow(new_image)
    plt.show()
    detect_stop_sign(resized_image, new_image)
    return line_count

##########
#ALL THE FUNCTIONS HERE AFTER TILL MAIN IS TO MERGE THE HOUGH LINES PROPERLY
#get orientation of a line, using its length
def get_orientation(line):
    orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
    return math.degrees(orientation)

#Check if line have enough distance and angle to be count as similar
def checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
    for group in groups:
        for line_old in group:
            if get_distance(line_old, line_new) < min_distance_to_merge:
                orientation_new = get_orientation(line_new)
                orientation_old = get_orientation(line_old)
                if abs(orientation_new - orientation_old) < min_angle_to_merge:
                    group.append(line_new)
                    return False
    return True

#Get distance between point and line
def DistancePointLine(point, line):
    px, py = point
    x1, y1, x2, y2 = line

    def lineMagnitude(x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        return lineMagnitude

    LineMag = lineMagnitude(x1, y1, x2, y2)
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine

#Get all possible distances between each dot of two lines and second line and return the shortest
def get_distance(a_line, b_line):
    dist1 = DistancePointLine(a_line[:2], b_line)
    dist2 = DistancePointLine(a_line[2:], b_line)
    dist3 = DistancePointLine(b_line[:2], a_line)
    dist4 = DistancePointLine(b_line[2:], a_line)
    return min(dist1, dist2, dist3, dist4)

#Clusterize (group) lines
def merge_lines_pipeline_2(lines):
    groups = [] 
    min_distance_to_merge = 6
    min_angle_to_merge = 5
    groups.append([lines[0]])
    for line_new in lines[1:]:
        if checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
            groups.append([line_new])
    return groups

#Sort lines cluster and return first and last coordinates
def merge_lines_segments1(lines):
    orientation = get_orientation(lines[0])

    # special case
    if (len(lines) == 1):
        return [lines[0][:2], lines[0][2:]]

    # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
    points = []
    for line in lines:
        points.append(line[:2])
        points.append(line[2:])
    # if vertical
    if 45 < orientation < 135:
        # sort by y
        points = sorted(points, key=lambda point: point[1])
    else:
        # sort by x
        points = sorted(points, key=lambda point: point[0])

    # return first and last point in sorted group
    # [[x,y],[x,y]]
    return [points[0], points[-1]]

#Main function for lines from cv.HoughLinesP() output merging
def process_lines(lines):
    lines_x = []
    lines_y = []
    for line_i in [l[0] for l in lines]:
        orientation = get_orientation(line_i)
        if 45 < orientation < 135:
            lines_y.append(line_i)
        else:
            lines_x.append(line_i)
    lines_y = sorted(lines_y, key=lambda line: line[1])
    lines_x = sorted(lines_x, key=lambda line: line[0])
    merged_lines_all = []
    for i in [lines_x, lines_y]:
        if len(i) > 0:
            groups = merge_lines_pipeline_2(i)
            merged_lines = []
            for group in groups:
                merged_lines.append(merge_lines_segments1(group))
            merged_lines_all.extend(merged_lines)
    final_lines = []
    for line in merged_lines_all:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        final_lines.append([[x1, y1, x2, y2]])
    return final_lines
#############
###########################################################################


def runon_image(path):
    frame = cv2.imread(path)
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections_in_frame = detect_lane(frame1)
    print('Total lanes detected in image: ', path, ' is:', detections_in_frame)
    return detections_in_frame


def runon_folder(path):
    files = None
    if (path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    for f in files:
        print()
        print(f)
        if f != 'lane/.DS_Store':
            f_detections = runon_image(f)
            all_detections += f_detections
    return all_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None:
        folder = 'lane'
    # print("Folder path must be given \n Example: python proj1.py -folder images")
    # sys.exit()

    if folder is not None:
        all_detections = runon_folder(folder)
        print()
        print("Total of ", all_detections, " detections")
        print()
    cv2.destroyAllWindows()
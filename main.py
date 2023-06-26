import torch
import cv2
import numpy as np
import statistics
import random
import time
import argparse


# randomize bgr values (colors)
# for drawing of bounding boxes, labels ...
def randomize_bgr():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


# grab the region of interest
def region_of_interest(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    mask = np.zeros_like(frame)

    # define the region of interest
    roi = np.array([[
        (0.1*width, 0.8*height),
        (0.4*width, 0.5*height),
        (0.6*width, 0.5*height),
        (0.9*width, 0.8*height)]],
        np.int32
    )

    cv2.fillPoly(mask, roi, 255)
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image


# canny edge -> grab ROI -> HoughLinesP
def line_detection(frame):
    # canny edge parameters
    ratio = 3
    kernel_size = 3
    threshold = 50

    # applying the canny edge filter
    # resize the frame to decrease resolution
    # reduce computational cost
    edge = cv2.Canny(
        cv2.resize(frame, (320, 320)),
        threshold,
        threshold*ratio,
        kernel_size
    )

    # grab the region of interest
    roi = region_of_interest(edge)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=40,
        minLineLength=3,
        maxLineGap=10
    )

    return lines


# return the best line (selecting the median results)
def get_best_line(lines, frame, left_lane, right_lane):
    lines_details_1 = []
    lines_details_2 = []

    # Iterate over points
    for points in lines:
        # extracting points nested in the list
        x1, y1, x2, y2 = points[0]

        # getting the angle of the line
        angle = np.arctan2(
            y2 - y1, x2 - x1) * 180 / np.pi

        # normalizing angle
        # add 180 if angle is less than 0
        if angle < 0:
            angle = angle + 180

        # getting relevant line details
        # filtering out irrelevant line details such as
        # horizontal lines and vertical lines
        # only accepting diagonal lines at a certain angle

        # get lines that are between 40 degrees and 80 degrees
        if angle >= 40 and angle < 80:
            lines_details_1.append(
                (angle, int(x1*2), int(y1*2), int(x2*2), int(y2*2))
            )

        # get lines that are between 100 degrees and 120 degrees
        if angle > 100 and angle <= 120:
            lines_details_2.append(
                (angle, int(x1*2), int(y1*2), int(x2*2), int(y2*2))
            )

    # usually, we will get multiple results
    # we will select the optimal result
    # by grabbing the median
    # getting the mean is not ideal as there might be outliers
    # which will shift the line detection result

    # get the median line details (right)
    if len(lines_details_1) != 0:
        # if the length of the list is even,
        # remove the last element
        if len(lines_details_1) % 2 == 0:
            lines_details_1.pop(-1)

        # get the median
        median_1 = statistics.median(lines_details_1)

        # using basic trigonometry to shift the origin point
        # to x1, frame_height
        # finding the x-coordinate (x1) when y = frame_height
        # tan(theta) = opp / hyp
        opp = int(frame.shape[0] - median_1[2])
        x1 = int(median_1[1] + opp / np.tan(
            median_1[0] * np.pi / 180))

        # finding the x-coordinate (x2)
        # when y2 = frame_height * 0.5
        # half the height of the frame
        # tan(theta) = opp / hyp
        y2 = int(0.5 * frame.shape[0])
        x2 = int(x1 - y2 / np.tan(
            median_1[0] * np.pi / 180))

        right_lane = (x1, frame.shape[0], x2, y2, median_1[0])

    # get the median line details (left)
    if len(lines_details_2) != 0:
        # if the length of the list is even,
        # remove the last element
        if len(lines_details_2) % 2 == 0:
            lines_details_2.pop(-1)

        # get the median
        median_2 = statistics.median(lines_details_2)

        # using basic trigonometry to shift the origin point
        # to x1, frame_height
        # finding the x-coordinate (x1) when y = frame_height
        # tan(theta) = opp / hyp
        opp = int(frame.shape[0] - median_2[2])
        x1 = int(median_2[1] + opp / np.tan(
            median_2[0] * np.pi / 180))

        # finding the x-coordinate (x2)
        # when y2 = frame_height * 0.5
        # half the height of the frame
        # tan(theta) = opp / hyp
        y2 = int(0.5 * frame.shape[0])
        x2 = int(x1 - y2 / np.tan(
            median_2[0] * np.pi / 180))

        left_lane = (x1, frame.shape[0], x2, y2, median_2[0])

    return left_lane, right_lane


# finding which lane number the vehicle is in
def lane_segregation(left_lane, right_lane, veh_x, veh_y):
    left_lane_x = None
    right_lane_x = None

    if left_lane is not None:
        # tan(theta) = opp / hyp
        left_lane_x = int(left_lane[0] +
                          (veh_y - left_lane[1]) /
                          np.tan(left_lane[4] *
                                 np.pi / 180))

    if right_lane is not None:
        # tan(theta) = opp / hyp
        right_lane_x = int(right_lane[0] +
                           (veh_y - right_lane[1]) /
                           np.tan(right_lane[4] *
                                  np.pi / 180))

    lane_number = None

    # checking which lane number the vehicle is in
    if left_lane_x is not None:
        if right_lane_x is not None:
            if veh_x <= left_lane_x:
                lane_number = 1
            elif veh_x <= right_lane_x:
                lane_number = 2
            else:
                lane_number = 3

        else:
            if veh_x <= left_lane_x:
                lane_number = 1
            else:
                lane_number = 2

    else:
        if left_lane_x is not None:
            if veh_x <= left_lane_x:
                lane_number = 1
            elif veh_x <= right_lane_x:
                lane_number = 2
            else:
                lane_number = 3

    return left_lane_x, right_lane_x, lane_number


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, required=False)

    args = parser.parse_args()

    if args.video_path is not None:
        video_path = args.video_path
    else:
        video_path = './sample_video.mp4'

    # class decoder (index to class name)
    label_details = {
        0: ['biker', randomize_bgr()],
        1: ['car', randomize_bgr()],
        2: ['pedestrian', randomize_bgr()],
        3: ['trafficLight', randomize_bgr()],
        4: ['trafficLight-Green', randomize_bgr()],
        5: ['trafficLight-GreenLeft', randomize_bgr()],
        6: ['trafficLight-Red', randomize_bgr()],
        7: ['trafficLight-RedLeft', randomize_bgr()],
        8: ['trafficLight-Yellow', randomize_bgr()],
        9: ['trafficLight-YellowLeft', randomize_bgr()],
        10: ['truck', randomize_bgr()]
    }

    # path to the yolov7 model
    path = './neural_networks/yolov7.pt'

    # load model
    model = torch.hub.load(
        'WongKinYiu/yolov7',
        'custom',
        f'{path}',
        trust_repo=True,
        force_reload=True
    )

    model.eval()

    cap = cv2.VideoCapture(video_path)

    left_lane = None
    right_lane = None

    while (cap.isOpened()):
        start_time = time.perf_counter()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            # reduce resolution
            # save computational cost
            frame_copy = cv2.resize(frame.copy(), (640, 640))

            # lane detection (using canny edge --> hough line transform)
            lines = line_detection(frame)

            # get left and right lane results
            if lines is not None:
                left_lane, right_lane = get_best_line(
                    lines, frame_copy, left_lane, right_lane
                )

            # draw lanes (right)
            if right_lane is not None:
                cv2.line(
                    frame_copy,
                    (right_lane[0], right_lane[1]),
                    (right_lane[2], right_lane[3]),
                    (0, 0, 255),
                    5,
                )

            # draw lanes (left)
            if left_lane is not None:
                cv2.line(
                    frame_copy,
                    (left_lane[0], left_lane[1]),
                    (left_lane[2], left_lane[3]),
                    (0, 0, 255),
                    5,
                )

            # vehicle detection (using yolov7)

            # get the detection
            results = model(frame)
            det = results.xyxy[0]

            for d in det:
                # scale to the new resized frame
                xmin = int(d[0] / frame.shape[1] * 640)
                ymin = int(d[1] / frame.shape[0] * 640)
                xmax = int(d[2] / frame.shape[1] * 640)
                ymax = int(d[3] / frame.shape[0] * 640)
                conf = d[4]
                label = int(d[5])

                # confidence threshold at more than 0.5
                # filter out detections that are at less than
                # 0.5 confidence rate
                if label == 1 or label == 10:
                    if conf >= 0.5:
                        cv2.rectangle(
                            frame_copy,
                            (xmin, ymin),
                            (xmax, ymax),
                            label_details[label][1],
                            (2)
                        )

                        cv2.putText(
                            img=frame_copy,
                            text=f'{label_details[label][0]} {conf*100:.2f}%',
                            org=(xmin, ymin-5),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=0.5,
                            color=label_details[label][1],
                            thickness=1
                        )

                        # lane segregation

                        # taking the reference point of the vehicle
                        # reference point somewhere in the middle
                        veh_x = int(xmin + (xmax - xmin) / 2)
                        veh_y = int(ymin + (ymax - ymin) * (3 / 4))

                        # draw vehicle reference point
                        cv2.circle(
                            img=frame_copy,
                            center=(veh_x, veh_y),
                            radius=1,
                            color=(0, 255, 0),
                            thickness=5
                        )

                        # getting the lane number
                        (left_lane_x,
                         right_lane_x,
                         lane_number) = lane_segregation(
                            left_lane, right_lane, veh_x, veh_y
                        )

                        # drawing lanes
                        if left_lane_x is not None:
                            if right_lane_x is not None:
                                if lane_number == 1:
                                    cv2.circle(
                                        img=frame_copy,
                                        center=(left_lane_x, veh_y),
                                        radius=1,
                                        color=(0, 255, 0),
                                        thickness=5
                                    )

                                    cv2.line(
                                        frame_copy,
                                        (veh_x, veh_y),
                                        (left_lane_x, veh_y),
                                        (255, 255, 255),
                                        2,
                                    )

                                elif lane_number == 2:
                                    cv2.circle(
                                        img=frame_copy,
                                        center=(left_lane_x, veh_y),
                                        radius=1,
                                        color=(0, 255, 0),
                                        thickness=5
                                    )

                                    cv2.line(
                                        frame_copy,
                                        (veh_x, veh_y),
                                        (left_lane_x, veh_y),
                                        (255, 255, 255),
                                        2,
                                    )

                                    cv2.circle(
                                        img=frame_copy,
                                        center=(right_lane_x, veh_y),
                                        radius=1,
                                        color=(0, 255, 0),
                                        thickness=5
                                    )

                                    cv2.line(
                                        frame_copy,
                                        (veh_x, veh_y),
                                        (right_lane_x, veh_y),
                                        (255, 255, 255),
                                        2,
                                    )

                                else:
                                    cv2.circle(
                                        img=frame_copy,
                                        center=(right_lane_x, veh_y),
                                        radius=1,
                                        color=(0, 255, 0),
                                        thickness=5
                                    )

                                    cv2.line(
                                        frame_copy,
                                        (veh_x, veh_y),
                                        (right_lane_x, veh_y),
                                        (255, 255, 255),
                                        2,
                                    )

                            else:
                                cv2.circle(
                                    img=frame_copy,
                                    center=(left_lane_x, veh_y),
                                    radius=1,
                                    color=(0, 255, 0),
                                    thickness=5
                                )

                                cv2.line(
                                    frame_copy,
                                    (veh_x, veh_y),
                                    (left_lane_x, veh_y),
                                    (255, 255, 255),
                                    2,
                                )

                        else:
                            if right_lane_x is not None:
                                cv2.circle(
                                    img=frame_copy,
                                    center=(right_lane_x, veh_y),
                                    radius=1,
                                    color=(0, 255, 0),
                                    thickness=5
                                )

                                cv2.line(
                                    frame_copy,
                                    (veh_x, veh_y),
                                    (right_lane_x, veh_y),
                                    (255, 255, 255),
                                    2,
                                )

                        if lane_number is not None:
                            cv2.putText(
                                img=frame_copy,
                                text=f'lane: {lane_number}',
                                org=(xmin, ymax+15),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=0.5,
                                color=(255, 0, 0),
                                thickness=1
                            )

            # calculate the time taken to process for 1 frame
            end_time = time.perf_counter() - start_time

            cv2.putText(
                img=frame_copy,
                text=f'{1/end_time:0.2f} fps',
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=1
            )

            cv2.imshow('lane_det', cv2.resize(frame_copy, (640, 640)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

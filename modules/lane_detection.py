import cv2
import numpy as np
import configparser
import statistics


class LaneDetection:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # store lane details
        self.lines = {'left': None, 'right': None}

        # number of frames of not detecting any line
        # before resetting
        self.__count = 10
        self.__left_count = self.__count
        self.__right_count = self.__count

    # canny edge
    def __canny(self, frame):
        t_lower = int(self.config['canny']['t_lower'])
        t_upper = int(self.config['canny']['t_upper'])
        aperture_size = int(self.config['canny']['aperture_size'])
        L2Gradient = self.config['canny']['L2Gradient']
        L2Gradient = False if L2Gradient == 'False' else True

        # applying the canny edge filter
        edge = cv2.Canny(
            image=frame,
            threshold1=t_lower,
            threshold2=t_upper,
            apertureSize=aperture_size,
            L2gradient=L2Gradient,
        )

        return edge

    # grab the region of interest
    def __region_of_interest(self, frame):
        coords = {}
        for coord in self.config['region_of_interest']:
            value = float(self.config['region_of_interest'][coord])
            coords.update({coord: value})

        height = frame.shape[0]
        width = frame.shape[1]
        mask = np.zeros_like(frame)

        # define the region of interest
        roi = np.array([[
            (coords['x1']*width, coords['y1']*height),
            (coords['x2']*width, coords['y2']*height),
            (coords['x3']*width, coords['y3']*height),
            (coords['x4']*width, coords['y4']*height)]],
            np.int32
        )

        cv2.fillPoly(mask, roi, 255)
        masked_image = cv2.bitwise_and(frame, mask)

        return masked_image

    # perform hough line transform
    def __hough_line_transform(self, frame):
        rho = int(self.config['hough_line_transform']['rho'])
        theta = float(self.config['hough_line_transform']['theta'])
        threshold = int(self.config['hough_line_transform']['threshold'])
        min_line_length = int(
            self.config['hough_line_transform']['min_line_length']
        )
        max_line_gap = int(self.config['hough_line_transform']['max_line_gap'])

        # Apply HoughLinesP method to
        # to directly obtain line end points
        lines = cv2.HoughLinesP(
            frame,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        return lines

    # get the angle of the line
    def __get_angle(self, x1, y1, x2, y2):
        opp = y2 - y1
        adj = x2 - x1
        theta = np.arctan(opp / adj) * 180 / np.pi

        return theta

    # filter lines based on angle threshold
    def __filter_lines(self, lines, thresh):
        filtered_lines = []

        if lines is not None:
            for line in lines:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]
                theta = self.__get_angle(x1, y1, x2, y2)

                for i, v in enumerate(thresh.values()):
                    if i % 2 == 0:
                        t_lower = v
                    else:
                        if theta >= t_lower and theta <= v:
                            filtered_lines.append((theta, x1, y1, x2, y2))

        return filtered_lines

    # split lines into 2 lists (left and right)
    def __split_lines(self, lines):
        left = []
        right = []

        if lines is not None:
            for line in lines:
                theta = line[0]

                if theta < 0:
                    left.append(line)
                else:
                    right.append(line)

        return left, right

    # further filter the lines output by getting the median
    def __get_best_line(self, lines):
        if len(lines) == 0:
            return None

        if len(lines) > 1:
            if len(lines) % 2 == 0:
                lines.pop(-1)

        if len(lines) > 1:
            return statistics.median(lines)

        return lines[0]

    # get new x-coord given new y-coord
    def __get_x_coord(self, line, new_y):
        x1 = line[1]
        y1 = line[2]
        theta = line[0]
        opp = new_y - y1

        adj = opp / np.tan(theta * np.pi / 180)
        new_x = int(adj + x1)

        return new_x

    # store lane details (retain memory)
    def __lane_details(self, left, right):
        if left is not None:
            self.lines.update({'left': left})

            if self.__left_count != self.__count:
                self.__left_count = self.__count
        else:
            if self.__left_count > 0:
                self.__left_count -= 1
            else:
                self.lines.update({'left': None})
                self.__left_count = self.__count

        if right is not None:
            self.lines.update({'right': right})

            if self.__right_count != self.__count:
                self.__right_count = self.__count
        else:
            if self.__right_count > 0:
                self.__right_count -= 1
            else:
                self.lines.update({'right': None})
                self.__right_count = self.__count

    # get the lanes
    def get_lanes(self, frame):
        edge = self.__canny(frame)
        region_of_interest = self.__region_of_interest(edge)
        lines = self.__hough_line_transform(region_of_interest)

        thresh = {}
        for param in self.config['angle_threshold']:
            value = float(self.config['angle_threshold'][param])
            thresh.update({param: value})

        filtered_lines = self.__filter_lines(lines, thresh)
        left, right = self.__split_lines(filtered_lines)
        left = self.__get_best_line(left)
        right = self.__get_best_line(right)
        self.__lane_details(left, right)

    # draw lanes
    def draw_lanes(self, frame):
        for v in self.lines.values():
            if v is not None:
                new_y1 = frame.shape[0]
                new_y2 = int(frame.shape[0]*0.5)

                new_x1 = self.__get_x_coord(v, new_y1)
                new_x2 = self.__get_x_coord(v, new_y2)

                cv2.line(
                    img=frame,
                    pt1=(new_x1, new_y1),
                    pt2=(new_x2, new_y2),
                    color=(255, 0, 0),
                    thickness=5,
                )

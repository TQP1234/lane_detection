import cv2
import numpy as np


class Vehicles:
    def __init__(self, veh_det, lines):
        self.veh_det = veh_det
        self.lines = lines

    # get the vehicle's reference point
    # set it to slightly below the center of the bbox
    def __get_veh_ref_point(self):
        ref = []
        for veh in self.veh_det:
            xmin = veh['xmin']
            ymin = veh['ymin']
            xmax = veh['xmax']
            ymax = veh['ymax']

            x_ref = xmin + (xmax - xmin) // 2
            y_ref = int(ymin + (ymax - ymin) * 0.8)

            ref.append(
                {
                    'x_ref': x_ref,
                    'y_ref': y_ref,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
            )

        return ref

    # get new x-coord given new y-coord
    def __get_x_coord(self, line, new_y):
        x1 = line[1]
        y1 = line[2]
        theta = line[0]
        opp = new_y - y1

        adj = opp / np.tan(theta * np.pi / 180)
        new_x = int(adj + x1)

        return new_x

    # if only 1 lane is being detected
    def __one_lane_sort(self, lines):
        veh_ref = self.__get_veh_ref_point()

        veh = []
        for ref in veh_ref:
            for line in lines:
                lane_x = self.__get_x_coord(line, ref['y_ref'])

                if ref['x_ref'] < lane_x:
                    veh.append(
                        {
                            'ref': ref,
                            'lane_x': [lane_x],
                            'lane_num': 1,
                        }
                    )

                else:
                    veh.append(
                        {
                            'ref': ref,
                            'lane_x': [lane_x],
                            'lane_num': 2,
                        }
                    )

        return veh

    # if 2 lanes are being detected
    def __two_lanes_sort(self):
        veh_ref = self.__get_veh_ref_point()

        veh = []
        for ref in veh_ref:
            left_x = self.__get_x_coord(self.lines['left'], ref['y_ref'])
            right_x = self.__get_x_coord(self.lines['right'], ref['y_ref'])

            if ref['x_ref'] < left_x:
                veh.append(
                    {
                        'ref': ref,
                        'lane_x': [left_x],
                        'lane_num': 1,
                    }
                )

            else:
                if ref['x_ref'] < right_x:
                    veh.append(
                        {
                            'ref': ref,
                            'lane_x': [left_x, right_x],
                            'lane_num': 2,
                        }
                    )

                else:
                    veh.append(
                        {
                            'ref': ref,
                            'lane_x': [right_x],
                            'lane_num': 3,
                        }
                    )

        return veh

    # check vehicles' current lane position
    def sort(self):
        # check if there is/are any non-None values for lines
        check = [k for k, v in self.lines.items() if v is not None]
        len_check = len(check)

        lines = []
        for k in check:
            lines.append(self.lines[k])

        if len_check <= 1:
            veh = self.__one_lane_sort(lines)
        else:
            veh = self.__two_lanes_sort()

        return veh

    # insert text
    def __insert_text(self, frame, text, x, y):
        cv2.putText(
            img=frame,
            text=text,
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
        )

    # draw circle
    def __draw_cirle(self, frame, x, y):
        cv2.circle(
            img=frame,
            center=(x, y),
            radius=1,
            color=(255, 255, 255),
            thickness=5,
        )

    # draw line
    def __draw_line(self, frame, x1, y1, x2, y2):
        cv2.line(
            img=frame,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=(0, 0, 255),
            thickness=3
        )

    # draw lines (veh ref point to lane)
    def draw_lane_details(self, frame, veh_details):
        for veh in veh_details:
            xmin = veh['ref']['xmin']
            ymax = veh['ref']['ymax']
            veh_ref_x = veh['ref']['x_ref']
            veh_ref_y = veh['ref']['y_ref']
            lane_x = veh['lane_x']

            for x in lane_x:
                self.__draw_line(
                    frame, veh_ref_x, veh_ref_y, x, veh_ref_y
                )

                self.__draw_cirle(frame, x, veh_ref_y)

            self.__insert_text(frame, f'lane {veh["lane_num"]}', xmin, ymax+20)
            self.__draw_cirle(frame, veh_ref_x, veh_ref_y)

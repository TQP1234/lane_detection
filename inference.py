from modules.vehicles_detection import VehiclesDetection
from modules.lane_detection import LaneDetection
from modules.vehicles_sort import Vehicles
import cv2
import configparser


def main():
    weights_path = './neural_networks/yolov8n.pt'
    label_path = './yolov8_classes.json'
    config_path = './config.ini'

    veh_det = VehiclesDetection(weights_path, label_path, config_path)
    lane_det = LaneDetection(config_path)

    video = configparser.ConfigParser()
    video.read(config_path)
    vid_path = video['frame']['video_path']
    width = int(video['frame']['width'])
    height = int(video['frame']['height'])

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(vid_path)

    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            frame = cv2.resize(frame, (width, height))

            # vehicles detection
            results = veh_det.inference(frame)

            # lane detection
            lane_det.get_lanes(frame)
            lines = lane_det.lines

            # check vehicles' current lane position
            vehicles = Vehicles(results, lines)
            veh_details = vehicles.sort()

            # draw boxes, lanes, etc...
            veh_det.draw_bounding_boxes(frame, results)
            lane_det.draw_lanes(frame)
            vehicles.draw_lane_details(frame, veh_details)

            cv2.imshow('video', frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    main()

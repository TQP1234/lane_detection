from ultralytics import YOLO
import json
import cv2
import random
import configparser


class VehiclesDetection:
    def __init__(self, weights_path, label_path, config_path):
        self.__weights_path = weights_path
        self.__label_path = label_path

        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.__conf_thresh = float(
            self.config['vehicles_detection']['conf_thresh']
        )

        # load model
        self.model = YOLO(self.__weights_path)

        # retrieve label mapping
        with open(self.__label_path, 'r') as f:
            self.__labels = json.load(f)

        self.__color = self.__color_map()

    # perform inference and return the results
    def inference(self, image):
        results = self.model.predict(image, verbose=False)

        # store results (bbox, label, confidence) in a list
        det = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                label = int(box.cls.tolist()[0])

                if label == 2 or label == 3 or label == 5 or label == 7:
                    # get box coordinates in (top, left, bottom, right) format
                    bbox = box.xyxy[0].tolist()
                    conf = box.conf.tolist()[0]

                    if conf >= self.__conf_thresh:
                        det.append(
                            {
                                'xmin': int(bbox[0]),
                                'ymin': int(bbox[1]),
                                'xmax': int(bbox[2]),
                                'ymax': int(bbox[3]),
                                'class': self.__labels[str(label)],
                                'confidence': conf,
                            }
                        )

        return det

    # randomize rgb
    def __randomize_rgb(self):
        # Generating a random number in between 0 and 255
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # opencv format (bgr)
        return (b, g, r)

    # color mapping
    def __color_map(self):
        color_dict = {}
        for _, v in self.__labels.items():
            color_dict.update({v: self.__randomize_rgb()})

        return color_dict

    # draw bounding boxes
    def draw_bounding_boxes(self, frame, detections):
        for det in detections:
            xmin = det['xmin']
            ymin = det['ymin']
            xmax = det['xmax']
            ymax = det['ymax']
            label = det['class']
            conf = f'{det["confidence"] * 100:.2f}'

            cv2.rectangle(
                img=frame,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=self.__color[label],
                thickness=5,
            )

            cv2.putText(
                img=frame,
                text=f'{label} - {conf}%',
                org=(xmin, ymin-10),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.5,
                color=self.__color[label],
                thickness=1,
            )

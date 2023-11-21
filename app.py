import streamlit as st
from modules.vehicles_detection import VehiclesDetection
from modules.lane_detection import LaneDetection
from modules.vehicles_sort import Vehicles
import cv2
import tempfile
import configparser


def main():
    st.set_page_config(page_title='Object Detection')

    st.markdown('''
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    ''', unsafe_allow_html=True)

    st.title('Lane Detection')

    if 'veh_det' not in st.session_state:
        weights_path = './neural_networks/yolov8n.pt'
        label_path = './yolov8_classes.json'
        config_path = './config.ini'

        veh_det = VehiclesDetection(weights_path, label_path, config_path)
        st.session_state['veh_det'] = veh_det

    if 'lane_det' not in st.session_state:
        config_path = './config.ini'

        lane_det = LaneDetection(config_path)
        st.session_state['lane_det'] = lane_det

    uploaded_file = st.file_uploader(
        'Choose a file', type=['png', 'jpg', 'mp4']
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        config_path = './config.ini'
        video = configparser.ConfigParser()
        video.read(config_path)
        width = int(video['frame']['width'])
        height = int(video['frame']['height'])

        veh_det = st.session_state['veh_det']
        lane_det = st.session_state['lane_det']

        # Read until video is completed
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret is True:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

                stframe.image(frame)


if __name__ == '__main__':
    main()

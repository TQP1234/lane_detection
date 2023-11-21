# Lane Detection

This is an experimental project showcasing vehicle and lane detection. A combination of deep neural network and traditional computer vision techiques are being employed.

A deep convolutional neural network (YOLOv8n - https://github.com/ultralytics/ultralytics) is used for vehicle detection.

For lane detection, traditional computer vision technique (OpenCV library) is used instead. A canny edge detector is first used to extract the edges. Secondly, define the Region of Interest (ROI) and apply Hough Line Transform (to get the lines) on the ROI. This is the basis for the lane detection.

## 1) Algorithm - Lane Detection

### Parameters

Parameters are stored in config.ini.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/config.ini#L1-L37

### Step 1: Canny Edge Detection

First, we will perform canny edge detection to get the edges.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/modules/lane_detection.py#L21-L38

### Step 2: Grab the ROI

Next, grab the ROI where we will perform Hough Line Transform with.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/modules/lane_detection.py#L40-L63

### Step 3: Hough Line Transform

Use Hough Line Transform the get the lines.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/modules/lane_detection.py#L65-L86

### Step 4: Filter out redundant lines

Using trigonometry formula, we will get the angle of the lines. And set a condition to only include lines that has an angle between 40-80 degrees.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/modules/lane_detection.py#L88-L115

### Step 5: Get the best line

Usually, we will get multiple line detection results. But we only need one line. To choose the best line, I have decided to select the median results.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/modules/lane_detection.py#L133-L145

### Step 6: Check which lane the vehicle is in

First, we will get the reference point of the vehicle (slightly below the center of the bounding box). Then, we will calculate both the lane point X where Y = the y-position of the vehicle. Lastly, we will compare the x-position of the vehicle against the x-postion of the left and right lane. And hence we shall get the lane position of the vehicle.

https://github.com/TQP1234/lane_detection/blob/ce49f8c62e9ade63a6e2cd10d516510ae63ed093/modules/vehicles_sort.py#L10-L131

## 2) Usage

Install the dependencies.

``` shell
pip install -r requirements.txt
```

<b>There are 2 methods to run this.</b>

### Method 1 - Executing the Python script directly

Change the video path that you want to inference on through config.ini (line 2). Then run the following command.

``` shell
python inference.py
```

### Method 2 - Running through web app

Please take note using this method will result in increased latency.

``` shell
streamlit run app.py
```

## 3) Sample Test Video

Video file is too large to upload in the README.md. Full video analysis can be found in the repository named as <b>lane_detection_analysis.mp4</b>.</br>

Snippet of the video analysis is shown below.

<img width="476" alt="lane_detection_screenshot" src="https://github.com/TQP1234/lane_detection/assets/75831732/ccae7433-c140-4aa4-96f3-cae6feb65c68">

## 4) Conclusion

From the test video, it could be seen that the lane detection accuracy is not the best. There are certain frames that show the lines being off. And this approach only allows for straight line detection. And if we are on a curved road, it's probably not going to work. For increased accuracy, a more modern approach such as a deep neural network could be used for lane detection.

Additionally, trucks are not being detected very accurately as well. To improve the model accuracy, we could finetune the model by adding more trucks to the dataset.

For traditional approach, it still can be used in certain cases such as in a fixed location where the environment is non-changing.

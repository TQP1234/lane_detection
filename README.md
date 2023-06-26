# Lane Detection

This is an experimental project showcasing vehicle and lane detection. A combination of deep neural network and traditional computer vision techiques are being employed.

A deep convolutional neural network (YOLOv7 - https://github.com/WongKinYiu/yolov7) is used for vehicle detection. Model (640 x 640) is trained using the dataset from https://universe.roboflow.com/roboflow-gw7yv/self-driving-car. There are class imbalance with the car class being over-represented. So, expect some biases.

For lane detection, traditional computer vision technique (OpenCV library) is used instead. A canny edge detector is first used to extract the edges. Secondly, define the Region of Interest (ROI) and apply Hough Line Transform (to get the lines) on the ROI. This is the basis for the lane detection.

## Algorithm - Lane Detection

### Step 1: Canny Edge Detection

First, we will perform canny edge detection to get the edges.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L152-L160

### Step 2: Grab the ROI

Next, grab the ROI where we will perform Hough Line Transform with.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L20-L37

### Step 3: Hough Line Transform

Use Hough Line Transform the get the lines.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L165-L174

### Step 4: Filter out redundant lines

Using trigonometry, we will get the angle of the lines. Set a condition to only include lines that has an angle between 40-80 degrees OR 100-120 degrees.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L181-L209

### Step 5: Get the optimal line

Usually, we will get multiple line detection results. But we only need one line. To choose the best line, I have decided to select the median results.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L224-L225

### Step 6: Shift the origin point

Using trigonometry, find the new origin point X where Y = frame_height.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L227-L233

### Step 7: Check which lane the vehicle is in

First, we will get the reference point of the vehicle (slightly below the center of the bounding box). Then, we will calculate both the lane point X where Y = the y-position of the vehicle. Lastly, we will compare the x-position of the vehicle against the x-postion of the left and right lane. And hence we shall get the lane position of the vehicle.

https://github.com/TQP1234/lane_detection/blob/4ea9e7d8858555a81c8526402c1e18fcf0a43cb7/main.py#L40-L86

## Conclusion

From the test video, it could be seen that the lane detection accuracy is not the best. There are certain frames that show the lines being off. And this approach only allows for line detection. And if we are on a curved road, it's probably not going to work. For increased accuracy, a more modern approach such as a deep neural network could be used for lane detection.

For traditional approach, it still can be used in certain cases such as in a fixed location where the environment is non-changing.

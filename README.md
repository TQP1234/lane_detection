# Lane Detection

This is an experimental project showcasing vehicle and lane detection. A combination of deep neural network and traditional computer vision techiques are being employed.

A deep convolutional neural network (YOLOv7 - https://github.com/WongKinYiu/yolov7) is used for vehicle detection. Model (640 x 640) is trained using the dataset from https://universe.roboflow.com/roboflow-gw7yv/self-driving-car. There are class imbalance with the car class being over-represented. So, expect some biases.

For lane detection, traditional computer vision technique (OpenCV library) is used instead. A canny edge detector is first used to extract the edges. Secondly, define the Region of Interest (ROI) and apply Hough Line Transform (to get the lines) on the ROI. This is the basis for the lane detection.

## Algorithm - Lane Detection

### Step 1: Canny Edge Detection

First, we will perform canny edge detection to get the edges.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L42-L55

### Step 2: Grab the ROI

Next, grab the ROI where we will perform Hough Line Transform with.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L20-L37

### Step 3: Hough Line Transform

Use Hough Line Transform the get the lines.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L60-L69

### Step 4: Filter out redundant lines

Using basic trigonometry, we will get the angle of the lines. Set a condition to only include lines that has an angle between 40-80 degrees OR 100-120 degrees.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L76-L108

### Step 5: Get the best line

Usually, we will get multiple line detection results. But we only need one line. To choose the best line, I have decided to select the median results.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L123-L124

### Step 6: Shift the origin point

Using trigonometry, find the new origin point X where Y = frame_height.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L126-L132

### Step 7: Check which lane the vehicle is in

First, we will get the reference point of the vehicle (slightly below the center of the bounding box). Then, we will calculate both the lane point X where Y = the y-position of the vehicle. Lastly, we will compare the x-position of the vehicle against the x-postion of the left and right lane. And hence we shall get the lane position of the vehicle.

https://github.com/TQP1234/lane_detection/blob/8607f089f622fe0c19af736ab2203fb76f4edf89/main.py#L175-L221

## Usage

Install the dependencies if you have not.

``` shell
pip install -r requirements.txt
```

Use the following command to run the lane detection.

``` shell
python main.py --video_path sample_video.mp4
```

Table of parameters:

| Parameter | Function | Required? | Example input | Default Value |
| :-- | :-: | :-: | :-: | :-: |
| video_path | Path to the video | No | sample_video.mp4 | sample_video.mp4 |

## Sample Test Video

Video file is too large to upload in the README.md. Full video analysis can be found in the repository named as <b>lane_detection_analysis.mp4</b>.</br>

Snippet of the video analysis is shown below.

<img width="476" alt="lane_detection_screenshot" src="https://github.com/TQP1234/lane_detection/assets/75831732/ccae7433-c140-4aa4-96f3-cae6feb65c68">

## Conclusion

From the test video, it could be seen that the lane detection accuracy is not the best. There are certain frames that show the lines being off. And this approach only allows for line detection. And if we are on a curved road, it's probably not going to work. For increased accuracy, a more modern approach such as a deep neural network could be used for lane detection.

Additionally, trucks are not being detected very accurately as well. This is to be expected, as the truck class is under-represented. To improve the model accuracy, we could increase the truck class by image augmentation or collecting more data (balancing the classes in the dataset).

For traditional approach, it still can be used in certain cases such as in a fixed location where the environment is non-changing.

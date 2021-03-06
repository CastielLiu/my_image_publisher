# ros读取摄像头并发布图像

## overview
    这是一个摄像头/视频图像捕获并发布驱动包，支持发布原始图像以及畸变矫正后的图像（需要内参矩阵文件），发布相机参数
    若内参文件不存在或载入失败，节点发布原始图像，否则发布校正后的图像
    
## modify log
	2020.12.17 添加故障检测机制，当连续多帧获取的图片为空时，退出节点程序，launch文件中设置节点重生(respawn="true"),可在摄像头插头松动恢复时继续工作。
	2021.2.11  添加视频读取并发布功能，可通过传递视频文件路径参数，实现视频读取并发布，当视频文件路径不为空时，自动读取视频而不再开启摄像头

## Nodes

### Node my_image_publisher_node
    摄像头图像获取和发布节点，
#### Params

* **`camera_id`** (int)
    摄像头id,默认为0

* **`calibration_file_path`** (string)
    摄像头内参文件路径

* **`frame_rate`**(bool)
    摄像头捕获帧率,如果摄像头不支持该帧率，将自动配置为默认帧率

* **`image_resolution`**(vector(2))
    摄像头分辨率设置，若摄像头不支持该分辨率，将自动配置为默认分辨率

* **`image_scale`**(float)
    图片缩放比例，默认值为1.0 不缩放

* **`frame_id`** (string)
    摄像头坐标系

* **`is_show_image`**(bool)
    是否可视化显示发布的图片

* **`video_file`**(string)
    视频文件路径(此参数默认为空，当不为空时，自动读取视频而不再开启摄像头)

#### Published Topics

* **`/image_raw`** ([sensor_msgs/Image])
    摄像头内参文件不存在或载入失败时发布/image_raw

* **`/image_rectified`** ([sensor_msgs/Image])
    摄像头内参文件正常载入时发布/image_rectified



### Node my_image_publisher_node
    摄像头图像获取和发布节点，
#### Params

* **`calibration_file_path`** (string)
    摄像头内参文件路径

#### Published Topics

* **`/camera_info`** ([sensor_msgs/CameraInfo])
    

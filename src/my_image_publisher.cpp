#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include "opencv2/imgproc/detail/distortion_model.hpp"

#define _NODE_NAME_ "image_publisher"
#define _RECTIFIED_WINDOW_ "rectified_image"
#define _RAW_WINDOW_ "raw_image"
using namespace cv;


class ImageTalker
{
public:
	ImageTalker()
	{
	}
	bool init()
	{
		ros::NodeHandle nh;
		ros::NodeHandle nh_private("~");
		
		std::string calibration_file_path;
		nh_private.param<std::string>("calibration_file_path",calibration_file_path,"");
		nh_private.param<std::string>("frame_id", frame_id_, "camera");
		
		nh_private.param<bool>("is_show_image",is_show_image_,false);
		nh_private.param<int>("frame_rate",frame_rate_,30);
		nh_private.param<float>("image_scale", imageScale_, 1.0);
		nh_private.param<int>("camera_id", cameraId_,0);
		ROS_INFO("[%s] camera id: %d", _NODE_NAME_, cameraId_);
		
		if(!ros::param::get("~image_resolution",imageResolution_))
		{
			imageResolution_[0] = 640;
			imageResolution_[1] = 480;
		}
		ROS_INFO("image size: %d*%d",imageResolution_[0],imageResolution_[1]);
	
		ROS_INFO("%s",calibration_file_path.c_str());
		
		image_transport::ImageTransport it(nh);
		if(!Loadintrinsics(calibration_file_path))
		{
			pub_ = it.advertise("/image_raw", 1);
			is_rectify_ = false;
		}
		else
		{
			pub_ = it.advertise("/image_rectified", 1);
			new_camera_instrinsics_ = getOptimalNewCameraMatrix(camera_instrinsics_,distortion_coefficients_,imgSize_,0.0);
			is_rectify_ = true;
		}
		return true;
	}
	
	void run()
	{
		cv::VideoCapture cap(cameraId_);
		
		if(!cap.isOpened())
		{
			ROS_ERROR("can not open video device\n");
			return;
		}
		
		cap.set(CV_CAP_PROP_FPS, frame_rate_);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, imageResolution_[0]);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, imageResolution_[1]);
		
		int w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
		int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		
		ROS_INFO("cv set camera size is: %d*%d",w,h);
		
		if(imgSize_.width != w || imgSize_.height != h)
		{
			ROS_ERROR("[%s] The camera resolution dont match the size in calibration file!", _NODE_NAME_);
			return;
		}
		
		int frame_rate = cap.get(CV_CAP_PROP_FPS);
		ROS_INFO("cv set camera frame rate is: %d",frame_rate);
		
		cv::Size target_size(int(w*imageScale_), int(h*imageScale_));
		
		if(is_show_image_)
		{
			if(is_rectify_) cv::namedWindow(_RECTIFIED_WINDOW_, cv::WINDOW_NORMAL);
			else cv::namedWindow(_RAW_WINDOW_, cv::WINDOW_NORMAL);
		}
		
		cv::Mat frame,src;
		sensor_msgs::ImagePtr msg;
		ros::Rate loop_rate(frame_rate_);
		
		while (ros::ok())
		{
			cap >> frame;
			//cv::flip(src,frame,-1); //rotate 90deg
			
			//ROS_INFO("image size:%d*%d",frame.size[0],frame.size[1]);

			if(!frame.empty())
			{
				if(is_rectify_)
				{
					cv::undistort(frame, src, camera_instrinsics_, distortion_coefficients_,new_camera_instrinsics_);

					if(imageScale_!=1.0)  cv::resize(src, src, target_size);
						
					msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", src).toImageMsg();
					if(is_show_image_) 
						cv::imshow(_RECTIFIED_WINDOW_, src); cv::waitKey(1);
				}
				else
				{
					if(imageScale_!=1.0)  cv::resize(frame, frame, target_size);
					
					msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
					
					if(is_show_image_) 
						cv::imshow(_RAW_WINDOW_, frame); cv::waitKey(1);
				}
				msg->header.frame_id = frame_id_;
				pub_.publish(msg);
			}
			
			loop_rate.sleep();
			ros::spinOnce();
		}
	}
	
	bool Loadintrinsics(const std::string &calibration_file_path)
	{
		if (calibration_file_path.empty())
		{
			ROS_INFO("[%s] missing calibration file path", _NODE_NAME_);
			return false;
		}

		cv::FileStorage fs(calibration_file_path, cv::FileStorage::READ);

		if (!fs.isOpened())
		{
			ROS_INFO("[%s] cannot open calibration file %s", _NODE_NAME_, calibration_file_path.c_str());
	 		return false;
		}

		camera_instrinsics_ = cv::Mat(3, 3, CV_64F);
		distortion_coefficients_ = cv::Mat(1, 5, CV_64F);

		cv::Mat dis_tmp;
		
		fs["CameraMat"] >> camera_instrinsics_;
		fs["DistCoeff"] >> dis_tmp;
		fs["ImageSize"] >> imgSize_;
		fs["DistModel"] >> distModel_;
		
		for (int col = 0; col < 5; col++)
			distortion_coefficients_.at<double>(col) = dis_tmp.at<double>(col);

		fs.release();
		
		return true;
	}

private:
	image_transport::Publisher pub_ ;
	ros::Publisher camera_info_pub_;
	
	cv::Mat new_camera_instrinsics_;
	cv::Mat camera_instrinsics_;
	cv::Mat distortion_coefficients_;
	std::string distModel_;
	
	int cameraId_;
	bool is_rectify_;
	cv::Size imgSize_;
	float imageScale_;
	std::vector<int> imageResolution_;
	int frame_rate_;
	std::string frame_id_;
	bool is_show_image_;
};


int main(int argc, char** argv)
{
	ros::init(argc, argv, _NODE_NAME_);
	
	ImageTalker image_talker;
	image_talker.init();
	image_talker.run();
	return 0;
}

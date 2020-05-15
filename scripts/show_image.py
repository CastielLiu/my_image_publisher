#!/usr/bin/python  
import roslib
import sys
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage 
from cv_bridge import CvBridge, CvBridgeError

class ShowImage:
	def __init__(self):
		self.bridge = CvBridge()
		self.sub_image = rospy.Subscriber("/image_raw/compressed",CompressedImage,self.image_callback, queue_size=1)
		self.is_save_video = False
		self.videoOut = None 
		
	
	def image_callback(self,image_data):
		try:
			np_arr = np.fromstring(image_data.data, np.uint8)
			image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			cv2.line(image,(19,238),(92,123),(0,255,0),3)
			cv2.line(image,(316,238),(230,123),(0,255,0),3)
#			print(image.shape)
			if self.videoOut is None:
				fourcc = cv2.VideoWriter_fourcc(*'XVID')
				self.videoOut = cv2.VideoWriter("result.avi",fourcc, 20.0, image.shape[0:2])
				cv2.namedWindow("robot",0)
				
			if(self.is_save_video):
				self.videoOut.write(image)
			cv2.imshow("robot",image)
			cv2.waitKey(10)
			
		except CvBridgeError as e:
			print(e)
			
def main(args):
	rospy.init_node('show_image_node')
	show_image = ShowImage()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

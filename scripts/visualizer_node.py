#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge
import cv2
import numpy as np

class NanoTrackVisualizer(Node):
    def __init__(self):
        super().__init__('nanotrack_visualizer')
        
        # Params
        self.declare_parameter('visualization_topic', '/nanotrack/visualization')
        self.declare_parameter('target_roi_topic', '/target_roi')
        
        self.vis_topic = self.get_parameter('visualization_topic').get_parameter_value().string_value
        self.roi_topic = self.get_parameter('target_roi_topic').get_parameter_value().string_value
        
        # Sub/Pub
        self.sub_img = self.create_subscription(Image, self.vis_topic, self.img_callback, 10)
        self.pub_roi = self.create_publisher(RegionOfInterest, self.roi_topic, 10)
        
        self.bridge = CvBridge()
        self.window_name = "NanoTrack C++ Visualizer"
        
        self.selection_start = None
        self.drag_rect = None
        
        # Mouse logic
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.has_display = True
            self.get_logger().info("Visualizer Node Started. Waiting for images...")
        except Exception as e:
            self.get_logger().error(f"Could not initialize window (Headless?): {e}")
            self.has_display = False
            self.get_logger().warn("Visualizer is running without GUI. Only ROI publication via parameters/requests will work.")

    def mouse_callback(self, event, x, y, flags, param):
        if not self.has_display:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.drag_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selection_start:
                x_start, y_start = self.selection_start
                self.drag_rect = (min(x_start, x), min(y_start, y), abs(x - x_start), abs(y - y_start))
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selection_start:
                x_start, y_start = self.selection_start
                w = abs(x - x_start)
                h = abs(y - y_start)
                x_min = min(x_start, x)
                y_min = min(y_start, y)
                
                if w > 10 and h > 10:
                    roi = RegionOfInterest()
                    roi.x_offset = int(x_min)
                    roi.y_offset = int(y_min)
                    roi.width = int(w)
                    roi.height = int(h)
                    roi.do_rectify = False 
                    
                    self.pub_roi.publish(roi)
                    self.get_logger().info(f"Published ROI: {roi.x_offset}, {roi.y_offset}, {roi.width}x{roi.height}")
                
            self.selection_start = None
            self.drag_rect = None

    def img_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_frame = cv_img.copy()
            
            # Draw drag rect
            display_img = self.current_frame.copy()
            if self.drag_rect:
                x, y, w, h = self.drag_rect
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            if self.has_display:
                cv2.imshow(self.window_name, display_img)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Image Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = NanoTrackVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from hb_interfaces.msg import Pose2D, Poses2D

class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')
        self.bridge = CvBridge()

        # Arena Constants (mm)
        self.size = 2438.4
        self.cam_height_m = 2.4384    
        self.floor_z_m = 0.0010    
        self.crate_top_z_m = 0.0600    

        # ID Configuration
        self.corner_ids_ordered = [1, 3, 7, 5]   # TL, TR, BR, BL
        self.bot_ids = {0, 2, 4}                 # Crystal, Frostbite, Glacio
        self.crate_min, self.crate_max = 10, 49

        # Topics
        self.image_sub       = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.crate_poses_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.bot_poses_pub   = self.create_publisher(Poses2D, '/bot_pose', 10)
        self.debug_img_pub   = self.create_publisher(Image, '/debug_image', 10)

        # World homography target (mm): Perfectly Relative Square
        self.world_matrix = np.array([
            [0.0, 0.0],
            [self.size, 0.0],
            [self.size, self.size],
            [0.0, self.size]], dtype=np.float32)

        # ArUco Setup (Compatible with OpenCV 4.6.0)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.get_logger().info(f'Strict Relative PoseDetector Online: 0 to {self.size}mm')

    def _to_world_pts(self, pts_px, H):
        """Converts pixels to relative arena coordinates with clipping."""
        pts_in = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        try:
            res = cv2.perspectiveTransform(pts_in, H).reshape(-1, 2)
            # Clip results to ensure they stay inside the square (0 to 2438.4)
            x = np.clip(res[0][0], 0.0, self.size)
            y = np.clip(res[0][1], 0.0, self.size)
            return float(x), float(y)
        except: return None

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = self.clahe.apply(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0))
            overlay = frame.copy()

            # 1. Detect Markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

            if ids is not None:
                detected_ids = ids.flatten().tolist()
                
                # --- STRICT GATEKEEPER: ALL 4 CORNERS MUST BE VISIBLE ---
                if all(cid in detected_ids for cid in self.corner_ids_ordered):
                    
                    # Map outermost corner of each marker to the arena ends
                    px_pts = []
                    # ID 1 (TL) -> Top-Left corner [0]
                    px_pts.append(corners[detected_ids.index(1)][0][0])
                    # ID 3 (TR) -> Top-Right corner [1]
                    px_pts.append(corners[detected_ids.index(3)][0][1])
                    # ID 7 (BR) -> Bottom-Right corner [2]
                    px_pts.append(corners[detected_ids.index(7)][0][2])
                    # ID 5 (BL) -> Bottom-Left corner [3]
                    px_pts.append(corners[detected_ids.index(5)][0][3])

                    # Calculate H for THIS FRAME only (No memory/counters)
                    H, _ = cv2.findHomography(np.array(px_pts, dtype=np.float32), self.world_matrix)

                    if H is not None:
                        crates_msg, bots_msg = Poses2D(), Poses2D()
                        s_crate = (self.cam_height_m - self.crate_top_z_m) / self.cam_height_m

                        for i, mid in enumerate(detected_ids):
                            if mid in self.corner_ids_ordered: continue
                            
                            is_bot = (mid in self.bot_ids)
                            is_crate = (self.crate_min <= mid <= self.crate_max)
                            if not (is_bot or is_crate): continue

                            # Calculate Center and World Position
                            c = corners[i][0]
                            pixel_center = np.mean(c, axis=0)
                            
                            # Parallax for crates
                            if is_crate:
                                img_h, img_w = gray.shape
                                img_center = np.array([img_w/2, img_h/2])
                                pixel_center = img_center + (pixel_center - img_center) * s_crate

                            world_res = self._to_world(pixel_center, H)
                            if world_res is None: continue
                            wx, wy = world_res

                            # Calculate Relative Yaw
                            p1_w = self._to_world(c[0], H)
                            p2_w = self._to_world(c[1], H)
                            yaw = math.degrees(math.atan2(p2_w[1] - p1_w[1], p2_w[0] - p1_w[0]))

                            p = Pose2D(id=int(mid), x=wx, y=wy, w=yaw)
                            if is_bot: bots_msg.poses.append(p)
                            else: crates_msg.poses.append(p)

                            # --- VISUAL DASHBOARD ---
                            cv2.aruco.drawDetectedMarkers(overlay, [corners[i]], np.array([[mid]]))
                            label = f"X:{p.x:.1f} Y:{p.y:.1f} W:{p.w:.1f}"
                            cv2.putText(overlay, label, (int(c[0][0]), int(c[0][1]) - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        self.bot_poses_pub.publish(bots_msg)
                        self.crate_poses_pub.publish(crates_msg)

            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(overlay, "bgr8"))

        except Exception as e:
            self.get_logger().error(f'Perception Error: {str(e)}')

    # Internal helper for world conversion inside the loop
    def _to_world(self, pt_px, H):
        pts_in = np.array([pt_px], dtype=np.float32).reshape(-1, 1, 2)
        try:
            res = cv2.perspectiveTransform(pts_in, H).reshape(-1, 2)
            x = np.clip(res[0][0], 0.0, self.size)
            y = np.clip(res[0][1], 0.0, self.size)
            return float(x), float(y)
        except: return None

def main(args=None):
    rclpy.init(args=args)
    node = PoseDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from hb_interfaces.msg import BotCmdArray
import paho.mqtt.client as mqtt
import json

class UnifiedBridge(Node):
    def __init__(self):
        super().__init__('unified_hardware_bridge')
        
        # --- MQTT Setup (Updated for Paho MQTT 2.0 compatibility) ---
        try:
            # For Paho-MQTT v2.0+
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Global_Bridge_4B")
        except AttributeError:
            # For Paho-MQTT v1.x
            self.mqtt_client = mqtt.Client("Global_Bridge_4B")
            
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.loop_start() 

        # --- ROS Subscription ---
        self.subscription = self.create_subscription(
            BotCmdArray, 
            '/bot_cmd', 
            self.cmd_callback, 
            10)
        
        self.get_logger().info("!!! UNIFIED BRIDGE 4B ONLINE !!!")

    def cmd_callback(self, msg):
        for cmd in msg.cmds:
            topic = f"hb_bot_{cmd.id}/hardware_cmds"
            payload = {
                "m1": round(float(cmd.m1), 2),
                "m2": round(float(cmd.m2), 2),
                "m3": round(float(cmd.m3), 2),
                "base": int(cmd.base),
                "elbow": int(cmd.elbow),
                "grip": int(cmd.grip)
            }
            self.mqtt_client.publish(topic, json.dumps(payload))

def main():
    rclpy.init()
    node = UnifiedBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
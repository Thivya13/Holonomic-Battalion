#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from hb_interfaces.msg import Poses2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import math
import paho.mqtt.client as mqtt
import json
import time

class Task5BManager(Node):
    def __init__(self):
        super().__init__('task_5b_manager')

        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Global_Manager_5B")
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.loop_start()

        self.bot_ids = [0, 2, 4]
        # Assignments for Task 5B
        self.assignments = {
            0: 21,              # Crystal -> Red P3 (Stacker)
            2: 16,              # Frostbite -> Green P4
            4: [12, 30]         # Glacio -> Red P1 (Base Builders)
        } 
        
        # Goals updated with your specific D1 coordinates
        self.goals = {
            0: (1210.0, 1420.0, 0.0),                  # Crystal: Stacking Center Goal
            2: (1012.0, 2047.0, math.radians(-94.0)),  # Frostbite: D2 Green
            4: [(1232.0, 1413.0, math.radians(-8.7)),  # Glacio Drop 1 (Crate 12)
                (1183.0, 1422.0, math.radians(-7.1))]  # Glacio Drop 2 (Crate 30)
        }
        
        self.docks = {
            4: (864.0, 204.0), 
            2: (1568.0, 202.0), 
            0: (1216.0, 205.0) 
        }

        self.dock_yaws = {4: math.pi, 2: math.pi, 0: 0.0}

        # --- NAVIGATION PARAMETERS ---
        self.bot_safety_rad = 720.0    
        self.crate_safety_rad = 480.0  
        self.repulsion_gain = 8500.0   
        self.crit_zone_dist = 400.0    

        self.bot_poses = {0: None, 2: None, 4: None}
        self.last_seen = {0: 0.0, 2: 0.0, 4: 0.0} 
        self.crate_poses = {}
        self.states = {0: "NAV_TO_CRATE", 2: "NAV_TO_CRATE", 4: "NAV_TO_CRATE"}
        
        self.bot4_crate_idx = 0
        self.accumulated_yaw = {0: 0.0, 2: 0.0, 4: 0.0}
        self.prev_yaw = {0: None, 2: None, 4: None}

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub_bot = self.create_subscription(Poses2D, '/bot_pose', self.bot_cb, qos)
        self.sub_crate = self.create_subscription(Poses2D, '/crate_pose', self.crate_cb, qos)
        
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Task 5B: Bot 4 Specific D1 Coords (-8.7, -7.1) Active.")

    def bot_cb(self, msg):
        for p in msg.poses:
            bid = int(p.id)
            if bid in self.bot_poses:
                current_yaw = math.radians(p.w)
                if self.states[bid] == "SWEEPING" and self.prev_yaw[bid] is not None:
                    diff = current_yaw - self.prev_yaw[bid]
                    while diff > math.pi: diff -= 2 * math.pi
                    while diff < -math.pi: diff += 2 * math.pi
                    self.accumulated_yaw[bid] += abs(diff)
                self.bot_poses[bid] = (p.x, p.y, current_yaw)
                self.prev_yaw[bid] = current_yaw
                self.last_seen[bid] = time.time()

    def crate_cb(self, msg):
        for p in msg.poses:
            self.crate_poses[int(p.id)] = (p.x, p.y)

    def control_loop(self):
        now = time.time()
        for bid in self.bot_ids:
            if self.bot_poses[bid] is None or (now - self.last_seen[bid] > 3.0):
                continue 
            x, y, yaw = self.bot_poses[bid]
            state = self.states[bid]

            if bid == 4:
                cid = self.assignments[4][self.bot4_crate_idx]
                target_goal = self.goals[4][self.bot4_crate_idx]
            else:
                cid = self.assignments[bid]
                target_goal = self.goals[bid]

            if state == "NAV_TO_CRATE":
                if cid not in self.crate_poses: continue
                tx, ty = self.crate_poses[cid]
                # High thresholds for sequential red pickups to avoid dragging
                pickup_thresh = 200.0 if (bid == 4 and self.bot4_crate_idx == 0) else (90.0 if bid == 0 else 115.0)
                if (bid == 4 and self.bot4_crate_idx == 1): pickup_thresh = 150.0
                
                if math.hypot(tx-x, ty-y) < pickup_thresh:
                    self.send_to_bot(bid, 0, 0, 0, 135, 0, 0)
                    time.sleep(0.8) 
                    self.execute_arm_down_for_pickup(bid)
                    self.accumulated_yaw[bid] = 0.0 
                    self.states[bid] = "SWEEPING"
                else:
                    self.navigate(bid, tx, ty, math.atan2(ty-y, tx-x), 135, 0, 0)

            elif state == "SWEEPING":
                if self.accumulated_yaw[bid] < 12.8: 
                    self.send_to_bot(bid, 42, 42, 42, 0, 180, 1)
                else:
                    self.send_to_bot(bid, 0, 0, 0, 135, 0, 1)
                    time.sleep(0.5)
                    self.states[bid] = "DELIVER"

            elif state == "DELIVER":
                tx, ty, tw = target_goal
                
                # --- STACKING SYNC ---
                if bid == 0:
                    # Crystal waits until Glacio is returning from its SECOND drop
                    glacio_finished_base = (self.states[4] in ["RETURN", "ALIGN_AT_DOCK", "DONE"] and self.bot4_crate_idx == 1)
                    if not glacio_finished_base:
                        self.navigate(bid, 1250.0, 1000.0, 0.0, 135, 0, 1) # Waiting Point
                        continue

                if math.hypot(tx-x, ty-y) < 55.0:
                    self.states[bid] = "ALIGN_FOR_DROP"
                else:
                    self.navigate(bid, tx, ty, tw, 135, 0, 1)

            elif state == "ALIGN_FOR_DROP":
                target_yaw = target_goal[2]
                yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
                if abs(yaw_err) < 0.04:
                    self.send_to_bot(bid, 0, 0, 0, 135, 0, 1) 
                    time.sleep(0.5)
                    
                    if bid == 0:
                        self.execute_stack_drop_sequence(bid) # (50, 0)
                    else:
                        self.execute_full_drop_sequence(bid) # (0, 180)
                        
                    if bid == 4 and self.bot4_crate_idx == 0:
                        self.bot4_crate_idx = 1
                        self.send_to_bot(bid, 0, 0, 0, 135, 0, 0)
                        time.sleep(0.8)
                        self.states[4] = "NAV_TO_CRATE"
                    else:
                        self.states[bid] = "RETURN"
                else:
                    rot_speed = 28.0 if yaw_err > 0 else -28.0
                    self.send_to_bot(bid, rot_speed, rot_speed, rot_speed, 135, 0, 1)

            elif state == "RETURN":
                tx, ty = self.docks[bid]
                target_yaw = self.dock_yaws[bid]
                if bid == 4 and x > 920: tx, ty = 800.0, 500.0 
                if math.hypot(tx-x, ty-y) < 55.0:
                    self.states[bid] = "ALIGN_AT_DOCK"
                else:
                    self.navigate(bid, tx, ty, target_yaw, 135, 0, 0)

            elif state == "ALIGN_AT_DOCK":
                target_yaw = self.dock_yaws[bid]
                yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
                if abs(yaw_err) < 0.06:
                    self.send_to_bot(bid, 0, 0, 0, 135, 0, 0) 
                    self.states[bid] = "DONE"
                else:
                    rot_speed = 30.0 if yaw_err > 0 else -30.0
                    self.send_to_bot(bid, rot_speed, rot_speed, rot_speed, 135, 0, 0)

    def navigate(self, bid, tx, ty, tw, b, e, g):
        x, y, yaw = self.bot_poses[bid]
        dist_to_target = math.hypot(tx - x, ty - y)
        current_state = self.states[bid]

        k_attr = 1.4
        if current_state == "NAV_TO_CRATE" and dist_to_target < 220.0:
            k_attr = 0.35 
        elif dist_to_target < 200.0: 
            k_attr = 0.85
            
        vx_attr = (tx - x) * k_attr
        vy_attr = (ty - y) * k_attr
        vx_rep, vy_rep = 0.0, 0.0
        in_crit = False

        # --- BOT-BOT AVOIDANCE ---
        for other_id in self.bot_ids:
            if other_id == bid or self.bot_poses[other_id] is None: continue
            ox, oy, _ = self.bot_poses[other_id]
            dist = math.hypot(x - ox, y - oy)
            eff_rad, eff_gain = self.bot_safety_rad, self.repulsion_gain
            if (bid == 2 and other_id == 4) or (bid == 4 and other_id == 2):
                eff_rad, eff_gain = 700.0, self.repulsion_gain * 3.0 
            if current_state in ["RETURN", "ALIGN_AT_DOCK"] and self.states[other_id] == "DONE":
                eff_rad, eff_gain = 180.0, 800.0

            if dist < eff_rad:
                if dist < self.crit_zone_dist: in_crit = True
                force = eff_gain * (1.0/max(dist, 40.0)**2)
                ang = math.atan2(y - oy, x - ox)
                vx_rep += math.cos(ang) * force * 2000.0
                vy_rep += math.sin(ang) * force * 2000.0

        # --- CRATE AVOIDANCE ---
        skip_crate_rep = (current_state == "DELIVER" and dist_to_target < 250.0)
        
        if current_state not in ["ALIGN_FOR_DROP"] and not skip_crate_rep:
            curr_target_cid = self.assignments[4][self.bot4_crate_idx] if bid == 4 else self.assignments[bid]
            for cid, cpose in self.crate_poses.items():
                if current_state == "NAV_TO_CRATE" and cid == curr_target_cid: continue
                local_crate_rad, local_crate_gain = self.crate_safety_rad, self.repulsion_gain

                # Sequential dampening for Bot 4 building the base
                if bid == 4 and cid == 30 and self.bot4_crate_idx == 0:
                    local_crate_rad, local_crate_gain = 300.0, self.repulsion_gain * 0.4
                if bid == 4 and cid == 12 and self.bot4_crate_idx == 1:
                    local_crate_rad, local_crate_gain = 210.0, self.repulsion_gain * 1.0

                dist_c = math.hypot(x - cpose[0], y - cpose[1])
                if dist_c < local_crate_rad:
                    if dist_c < 200.0: in_crit = True
                    force_c = local_crate_gain * (1.0/max(dist_c, 50.0)**2)
                    ang_c = math.atan2(y - cpose[1], x - cpose[0])
                    vx_rep += math.cos(ang_c) * force_c * 1600.0
                    vy_rep += math.sin(ang_c) * force_c * 1600.0

        if in_crit:
            vx_world, vy_world = (vx_attr * 0.01) + vx_rep, (vy_attr * 0.01) + vy_rep
        else:
            vx_world, vy_world = vx_attr + vx_rep, vy_attr + vy_rep

        speed = math.hypot(vx_world, vy_world)
        max_s = 60.0
        if speed > max_s: vx_world, vy_world = (vx_world/speed)*max_s, (vy_world/speed)*max_s

        vx = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        vy = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world
        vw = math.atan2(math.sin(tw - yaw), math.cos(tw - yaw)) * 3.8 
        s1, s2, s3 = (-0.5*vx + 0.866*vy + vw), (-0.5*vx - 0.866*vy + vw), (vx + vw)
        self.send_to_bot(bid, s1, s2, s3, b, e, g)

    def execute_arm_down_for_pickup(self, bid):
        self.send_to_bot(bid, 0, 0, 0, 135, 0, 0)
        time.sleep(0.3)
        self.send_to_bot(bid, 0, 0, 0, 0, 180, 0) 
        time.sleep(2.5) 
        self.send_to_bot(bid, 0, 0, 0, 0, 180, 1) 
        time.sleep(0.6)

    def execute_full_drop_sequence(self, bid):
        self.send_to_bot(bid, 0, 0, 0, 0, 180, 1)
        time.sleep(2.5) 
        time.sleep(3.5) 
        self.send_to_bot(bid, 0, 0, 0, 0, 180, 0) 
        time.sleep(1.5) 
        self.send_to_bot(bid, 0, 0, 0, 135, 0, 0) 
        time.sleep(0.8)
        start_esc = time.time()
        while time.time() - start_esc < 2.0:
            self.send_to_bot(bid, -30.0, -30.0, 60.0, 135, 0, 0)
            time.sleep(0.05)
        self.send_to_bot(bid, 0, 0, 0, 135, 0, 0)

    def execute_stack_drop_sequence(self, bid):
        # Specific angles for Bot 0 Stacking: Base 50, Elbow 0
        self.send_to_bot(bid, 0, 0, 0, 50, 0, 1)
        time.sleep(2.5) 
        time.sleep(3.5) 
        self.send_to_bot(bid, 0, 0, 0, 50, 0, 0) 
        time.sleep(1.5) 
        self.send_to_bot(bid, 0, 0, 0, 135, 0, 0) 
        time.sleep(0.8)
        # Escape backwards to clear the stack
        start_esc = time.time()
        while time.time() - start_esc < 2.0:
            self.send_to_bot(bid, -30.0, -30.0, 60.0, 135, 0, 0)
            time.sleep(0.05)
        self.send_to_bot(bid, 0, 0, 0, 135, 0, 0)

    def send_to_bot(self, bid, s1, s2, s3, b, e, g):
        def friction(v):
            if abs(v) < 0.5: return 0.0
            return (np.sign(v) * 28.0) + v 
        payload = {
            "m1": float(np.clip(friction(s1), -100, 100)),
            "m2": float(np.clip(friction(s2), -100, 100)),
            "m3": float(np.clip(friction(s3), -100, 100)),
            "base": int(b), "elbow": int(e), "grip": int(g)
        }
        self.mqtt_client.publish(f"hb_bot_{bid}/hardware_cmds", json.dumps(payload))

def main():
    rclpy.init()
    try:
        rclpy.spin(Task5BManager())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
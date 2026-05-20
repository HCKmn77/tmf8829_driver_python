#!/usr/bin/env python3
"""ZeroMQ -> ROS2 PointCloud2 bridge for TMF8829.

Usage example:
  python tmf8829_zeromq_client_ros2.py 

This script connects to the existing ZMQ server running on Linux, starts measurement
and republishes incoming measurement frames as sensor_msgs/PointCloud2 messages.
"""
import argparse
import os
import sys
import time
import threading
import struct
import ctypes

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

SCRIPT_DIR = os.path.dirname(__file__)
TMF8829_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TMF8829_DIR not in sys.path:
    sys.path.insert(0, TMF8829_DIR)

from zeromq.tmf8829_zeromq_client import ZeroMqClient
from tmf8829_application_common import Tmf8829AppCommon
from zeromq.tmf8829_zeromq_common import *
from aos_com.register_io import ctypes2Dict


class ZmqRos2Publisher(Node):
    def __init__(self, *, use_local: bool = True, host: str = None, cmd_port: int = 5557, data_port: int = 5558, topic: str = "/tmf8829/pointcloud", frame_id: str = "tmf8829_link", record_frames: int = 0):
        super().__init__("tmf8829_zeromq_ros2_bridge")
        self._topic = topic
        self._frame_id = frame_id
        self._record_frames = int(record_frames)
        self._publisher = self.create_publisher(PointCloud2, self._topic, 10)

        # ZMQ client
        self._client = ZeroMqClient()
        self._use_local = use_local
        self._host = host
        self._cmd_port = int(cmd_port)
        self._data_port = int(data_port)
        self._connection_mode = None
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        # connect and identify
        try:
            # prefer explicit host when provided
            if self._host:
                self._client.connect_host(self._host, self._cmd_port, self._data_port)
                self._connection_mode = 'host'
            else:
                if self._use_local:
                    self._client.connect_local()
                    self._connection_mode = 'local'
                else:
                    self._client.connect_linux()
                    self._connection_mode = 'linux'

            # The ZMQ server opens the TMF8829 hardware before binding its sockets,
            # so give it a short grace period instead of failing on the first startup race.
            identify_attempts = 10
            identify_delay_s = 0.5
            self._device_info = None
            for attempt in range(1, identify_attempts + 1):
                try:
                    self._device_info = self._client.identify()
                    break
                except TimeoutError as exc:
                    if attempt == identify_attempts:
                        raise
                    self.get_logger().warning(
                        f"ZMQ IDENTIFY attempt {attempt}/{identify_attempts} failed: {exc}; retrying"
                    )
                    time.sleep(identify_delay_s)
            self.get_logger().info(f"Identified device: {ctypes2Dict(self._device_info)}")
        except Exception as exc:
            self.get_logger().error(f"ZMQ connect/identify failed: {exc}")
            raise

        # start measurement
        try:
            if not self._client.start_measurement():
                self.get_logger().warning("Measurement not started (server may be logger-only)")
        except Exception as exc:
            self.get_logger().error(f"Start measurement failed: {exc}")

        self._worker.start()

    def _frames_to_points(self, result_frames):
        points = []
        if not result_frames:
            return points

        pixel_results = Tmf8829AppCommon.getFullPixelResult(
            frames=result_frames,
            toMM=True,
            deleteNone=True,
            pointCloud=False,
            distanceToXYZ=True,
        )

        for row in pixel_results:
            for pixel in row:
                for peak in pixel.get("peaks", []):
                    distance = peak.get("distance")
                    if distance is None:
                        continue
                    points.append({
                        "x": float(peak.get("x", 0.0)),
                        "y": float(peak.get("y", 0.0)),
                        "z": float(peak.get("z", 0.0)),
                        "distance": float(distance),
                        "snr": int(peak.get("snr", 0)),
                        "signal": int(peak.get("signal", 0) or 0),
                    })

        if not points:
            return points

        # TMF8829 returns values in millimeters when toMM=True; convert to meters
        for p in points:
            p['x'] = p['x'] / 1000.0
            p['y'] = p['y'] / 1000.0
            p['z'] = p['z'] / 1000.0
            p['distance'] = p['distance'] / 1000.0

        # Remap sensor axes -> ROS (RViz): sensor (x_side, y_up, z_forward)
        # RViz expects (x_forward, y_left?, z_up) conventional: we map
        #   rviz_x = sensor_z (forward)
        #   rviz_y = sensor_x (side)
        #   rviz_z = sensor_y (up)
        for p in points:
            sx, sy, sz = p['x'], p['y'], p['z']
            rvx = sz
            rvy = sx
            rvz = -sy
            p['x'], p['y'], p['z'] = rvx, rvy, rvz

        return points

    def _build_cloud(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.is_dense = False
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="distance", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="snr", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="signal", offset=20, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 24
        msg.row_step = msg.point_step * msg.width
        if points:
            msg.data = b"".join(
                struct.pack(
                    "<ffffII",
                    point["x"],
                    point["y"],
                    point["z"],
                    point["distance"],
                    point["snr"],
                    point["signal"],
                )
                for point in points
            )
        else:
            msg.data = b""

        return msg

    def _worker_loop(self):
        cnt = 0
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                zmq_result_data = self._client.get_result_data()
                header_size = ctypes.sizeof(tmf8829ContainerFrameHeader)
                resultFrame, histoFrames, refFrame = Tmf8829AppCommon.getFramesFromMeasurementResult(zmq_result_data[header_size:])

                points = self._frames_to_points(resultFrame)
                cloud_msg = self._build_cloud(points)
                self._publisher.publish(cloud_msg)
                cnt += 1
                if self._record_frames and cnt >= self._record_frames:
                    break

            except Exception as exc:
                self.get_logger().warning(f"Error in result loop: {exc}")
                time.sleep(0.1)

                # cleanup
        try:
            self._client.stop_measurement()
            self._client.leave()
            if self._connection_mode == 'local':
                self._client.disconnect_local()
            elif self._connection_mode == 'linux':
                self._client.disconnect_linux()
            elif self._connection_mode == 'host':
                # disconnect_host requires host and ports
                try:
                    self._client.disconnect_host(self._host, self._cmd_port, self._data_port)
                except Exception:
                    pass
        except Exception:
            pass

    def destroy_node(self):
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        super().destroy_node()


def _parse_args():
    parser = argparse.ArgumentParser(description="TMF8829 ZMQ → ROS2 PointCloud2 bridge")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--local", action="store_true", help="Connect to local (127.0.0.1) server")
    mode.add_argument("--linux", action="store_true", help="Connect to linux (169.254...) server")
    parser.add_argument("--host", default=None, help="Connect to specific host (overrides --local/--linux). Example: 192.168.56.1")
    parser.add_argument("--cmd-port", type=int, default=5557, help="Command socket port (default: 5557)")
    parser.add_argument("--data-port", type=int, default=5558, help="Data socket port (default: 5558)")
    parser.add_argument("--topic", default="/tmf8829/pointcloud", help="ROS2 topic to publish")
    parser.add_argument("--frame-id", default="tof_sensor_link", help="PointCloud2 header frame_id")
    parser.add_argument("--record-frames", type=int, default=0, help="Stop after N frames (0 = run forever)")
    return parser.parse_args()


def main():
    args = _parse_args()
    # default to local when no flags/host provided
    use_local = args.local or (not args.linux and not args.host)
    host = args.host
    cmd_port = args.cmd_port
    data_port = args.data_port

    # require at least one connection selection
    if (not use_local) and (not args.linux) and (not host):
        print("Specify --local, --linux or --host <addr>")
        sys.exit(1)

    rclpy.init()
    node = None
    try:
        node = ZmqRos2Publisher(use_local=use_local, host=host, cmd_port=cmd_port, data_port=data_port, topic=args.topic, frame_id=args.frame_id, record_frames=args.record_frames)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

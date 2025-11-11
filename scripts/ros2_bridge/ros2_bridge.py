# created by Pascal Aebersold with help from ChatGPT 5.0

import argparse

from isaaclab.app import AppLauncher

import omni.ext
import omni.kit.app
import os

# --- Configure ROS 2 environment ---
# $env:ROS_DISTRO = "humble"
os.environ["ROS_DISTRO"] = "humble"
# $env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp" 
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
# $env:PATH = "$env:PATH;C:\Users\pasca\miniforge3\envs\env_isaaclab\Lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\humble\lib"
isaac_bridge_lib = r"C:\Users\pasca\miniforge3\envs\env_isaaclab\Lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\humble\lib"
os.environ["PATH"] += ";" + isaac_bridge_lib
#  $env:ROS_DOMAIN_ID = "0"
os.environ["ROS_DOMAIN_ID"] = "0"

print("Isaac Sim 5.0.0 ROS 2 (Humble) environment configured")

os.system("isaacsim -s --enable isaacsim.ros2.bridge")

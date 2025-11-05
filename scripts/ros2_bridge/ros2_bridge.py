# created by Pascal Aebersold with help from ChatGPT 5.0

import argparse

from isaaclab.app import AppLauncher

import omni.ext
import omni.kit.app
import os

# --- Configure ROS 2 environment ---
os.environ["ROS_DISTRO"] = "humble"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
isaac_bridge_lib = r"C:\Users\pasca\miniforge3\envs\env_isaaclab\Lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\humble\lib"
os.environ["PATH"] += ";" + isaac_bridge_lib
os.environ["ROS_DOMAIN_ID"] = "0"

print("[✅] Isaac Sim 5.0.0 ROS 2 (Humble) environment configured")

# add argparse arguments
parser = argparse.ArgumentParser(description="This script shows the basic setup of two UR5e robots on a table.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
ui_exts = [
    "omni.kit.window.stage",      # Stage view
    "omni.kit.window.graph",      # Graph editor
    "omni.graph.ui",              # Graph editor backend
    "omni.kit.window.property",   # Property inspector
    "isaacsim.core",              # Core Isaac Sim
    "isaacsim.ros2.bridge"        # ROS 2 bridge name
]

for ext in ui_exts:
    ext_manager.set_extension_enabled_immediate(ext, True)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()

# in docker terminal run: docker run -it --rm --name ros2_jazzy --add-host=host.docker.internal:host-gateway --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp --env ROS_DOMAIN_ID=0 althack/ros2:jazzy-cuda-full-2025-10-03
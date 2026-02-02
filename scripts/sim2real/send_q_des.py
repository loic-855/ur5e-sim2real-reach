import logging
import sys
import time
import socket

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

ROBOT_HOST = "192.168.1.101"
ROBOT_PORT = 30004
ROBOT_PRIMARY_PORT = 30001
config_filename = "scripts/sim2real/URscript/rtde_input.xml"
ur_script_filename = "scripts/sim2real/URscript/impedance_control.script"

keep_running = True
update_statement = True


def send_urscript_file(filepath: str):
    """Load and send a URScript program from a .script file."""
    try:
        with open(filepath, 'r') as f:
            program = f.read()
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ROBOT_HOST, ROBOT_PRIMARY_PORT))
        program = program + "\n"
        s.sendall(program.encode('utf-8'))
        s.close()
        print(f"URScript sent successfully: {filepath}")
        return True
    except Exception as e:
        print(f"Error sending URScript: {e}")
        return False

conf = rtde_config.ConfigFile(config_filename)
output_names, output_types = conf.get_recipe("out")
q_des_names, q_des_types = conf.get_recipe("q_des")
stop_name, stop_type = conf.get_recipe("stop")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

#setup recipes
if not con.send_output_setup(output_names, output_types, frequency=60):
    logging.error("Unable to configure output")
    sys.exit()

setp = con.send_input_setup(q_des_names, q_des_types)
if setp is None:
    logging.error("Unable to configure q_des input")
    sys.exit()
  

# Setup RTDE for stop signal
stop = con.send_input_setup(stop_name, stop_type)
if stop is None:
    logging.error("Unable to configure stop signal")
    sys.exit()

# start data synchronization
if not con.send_start():
    logging.error("Unable to start synchronization")
    sys.exit()


# Send the URScript to the robot
stop.input_bit_register_64 = True
con.send(stop)
print("Sending URScript to robot...")
time.sleep(1)  # Petit délai avant d'envoyer
send_urscript_file(ur_script_filename)
time.sleep(2)  # Attendre que le script démarre sur le robot


# pose for the robot to cycle through
pose_list = [[0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
        [0.3, -1.47, 0.5, -1.27, 0.0, 0.0],
        [0.0, -1.77, 0.0, -1.67, 0.0, 0.3],
        [-0.4, -1.57, 1.77, -0.57, 0.0, 1.57]]



def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        num = i + 24
        sp_list.append(sp.__dict__["input_double_register_%i" % num])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 6):
        num = i + 24
        sp.__dict__["input_double_register_%i" % num] = list[i]
    return sp

# send position to stay at home initially
e = [0.0]*6

list_to_setp(setp, pose_list[0])
con.send(setp)  # IMPORTANT: envoyer setp au robot!
q_des = pose_list[0]
j = 0
moving_to_new_pose = False  # Flag pour éviter le changement immédiat

# Timing pour print à 5Hz
last_print_time = time.time()
print_interval = 1.0 / 5  # 5Hz = affiche tous les 0.2s

# control loop
try:
    while keep_running:
        # receive the current state
        state = con.receive()
        
        if state is None:
            print("Failed to receive state")
            break

        # check if move completed
        for i in range(6):
            e[i] = q_des[i] - state.actual_q[i]
        move_completed = all(abs(e[i]) < 3/180*3.14 for i in range(6))  # Seuil augmenté: 2° → 5°

        # do something...
        if move_completed and not moving_to_new_pose:
            print("Reached pose %d: %s" % (j, q_des))
            time.sleep(0.5)  # Petit délai avant de changer de pose
            j = (j + 1) % len(pose_list)  # len(pose_list) = 4, donc j cycling 0→1→2→3→0→1→2→3...
            q_des = pose_list[j]
            moving_to_new_pose = True
            print("Move to new pose %d: %s" % (j, q_des))
        elif not move_completed:
            moving_to_new_pose = False  # Reset quand on commence à bouger
            
        # Toujours envoyer la position désirée
        list_to_setp(setp, q_des)
        con.send(setp)
        
        # Print à 5Hz
        now = time.time()
        if now - last_print_time >= print_interval:

            formatted_actual_q = ['%.2f' % elem for elem in state.actual_q]
            print(f"Actual Q: {formatted_actual_q}")
            last_print_time = now

except KeyboardInterrupt:   
    keep_running = False
    print("\nStopping URScript on robot...")
    # Send stop signal to URScript via RTDE
    stop.input_bit_register_64 = False
    con.send(stop)
    time.sleep(1)  # Attendre que le script s'arrête proprement
    print("URScript stopped")
except rtde.RTDEException:
    con.disconnect()
    sys.exit()


con.send_pause()

con.disconnect()

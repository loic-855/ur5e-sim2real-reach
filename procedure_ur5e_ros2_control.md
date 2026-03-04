# Procédure de mise en route des robots UR5e avec ROS2

## Configuration du système

| Composant | IP |
|-----------|-----|
| PC ROS2 | 192.168.1.105 |
| Robot Gripper (UR5e) | 192.168.1.101 |
| Robot Screwdriver (UR5e) | 192.168.1.103 |
| OnRobot 2FG7 Gripper | 192.168.1.111 |

## Prérequis

### 1. Firewall - Ouvrir les ports

Les robots UR doivent pouvoir se connecter au PC ROS2 sur les ports 50001-50008.

```bash
sudo ufw allow from 192.168.1.0/24 to any port 50001:50008 proto tcp
```

### 2. URCaps External Control sur les robots

Sur chaque teach pendant, créer/configurer un programme avec le nœud **External Control**:

| Robot | Host IP (PC ROS2) | Custom Port |
|-------|-------------------|-------------|
| Gripper | 192.168.1.105 | 50002 |
| Screwdriver | 192.168.1.105 | 50006 |

---

## Procédure de démarrage

### Étape 1: Lancer le driver ROS2

```bash
cd /home/robots/wwro_ws
source install/local_setup.bash

ros2 launch wwro_startup wwro_control.launch.py \
  gripper_robot_ip:=192.168.1.101 \
  screwdriver_robot_ip:=192.168.1.103 \
  headless_mode:=false


ros2 launch wwro_startup wwro_control.launch.py \
  use_mock_hardware:=true

```

### Étape 2: Sur chaque teach pendant

1. Mettre le robot en **mode REMOTE** (obligatoire)
2. Charger le programme External Control (ex: `ext_ctr_loic.urp`)
3. Appuyer sur **Play ▶**
4. Le robot affiche "External control active" ou similaire

> ⚠️ **Important**: Rester en mode REMOTE après avoir lancé le programme !

### Étape 3: Vérifier les connexions

```bash
# Vérifier que les robots sont connectés
ss -tnp | grep "5000[1-8]"

# Doit afficher des connexions ESTAB vers 192.168.1.101 et 192.168.1.103
```

### Étape 4: Vérifier l'état du programme

```bash
ros2 service call /gripper_dashboard_client/program_state ur_dashboard_msgs/srv/GetProgramState
# Doit afficher: state='PLAYING'

ros2 service call /screwdriver_dashboard_client/program_state ur_dashboard_msgs/srv/GetProgramState
# Doit afficher: state='PLAYING'
```

### Étape 5: Vérifier robot_program_running

```bash
ros2 topic echo /gripper_io_and_status_controller/robot_program_running \
  --qos-reliability reliable --qos-durability transient_local --once
# Doit afficher: data: true
```

---

## Commandes de mouvement

### Mouvement du robot Gripper

```bash
ros2 action send_goal /gripper_scaled_joint_trajectory_controller/follow_joint_trajectory \
control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [gripper_shoulder_pan_joint, gripper_shoulder_lift_joint, gripper_elbow_joint, gripper_wrist_1_joint, gripper_wrist_2_joint, gripper_wrist_3_joint],
    points: [{positions: [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], time_from_start: {sec: 3, nanosec: 0}}]
  }
}"
```

### Mouvement du robot Screwdriver

```bash
ros2 action send_goal /screwdriver_scaled_joint_trajectory_controller/follow_joint_trajectory \
control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [screwdriver_shoulder_pan_joint, screwdriver_shoulder_lift_joint, screwdriver_elbow_joint, screwdriver_wrist_1_joint, screwdriver_wrist_2_joint, screwdriver_wrist_3_joint],
    points: [{positions: [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], time_from_start: {sec: 3, nanosec: 0}}]
  }
}"
```

### Mouvement simultané des deux robots

```bash
ros2 action send_goal /gripper_scaled_joint_trajectory_controller/follow_joint_trajectory \
control_msgs/action/FollowJointTrajectory "{...}" &

ros2 action send_goal /screwdriver_scaled_joint_trajectory_controller/follow_joint_trajectory \
control_msgs/action/FollowJointTrajectory "{...}" &

wait
```

### Contrôle du gripper OnRobot 2FG7

Service: /on_twofg7_grip_external, /on_twofg7_grip_internal, /on_twofg7_release_external, /on_twofg7_release_internal — type wwro_msgs/srv/OnTwofg7.
Type/fields: request gripper_operation {int64 tool_index, float64 width_mm, int64 force_n, int64 speed} → response {bool success}.

Commandes exemples (après avoir sourcé le workspace: source ~/wwro_ws/install/local_setup.bash):

Pour fermer (grip) à 37 mm avec 20 N, tool_index 0:

```bash
ros2 service call /on_twofg7_grip_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 37.0, force_n: 20, speed: 100}}"

ros2 service call /on_twofg7_release_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 100.0, force_n: 0, speed: 100}}"
```


---

## Troubleshooting

### Le robot accepte le goal mais ne bouge pas

1. **Vérifier le firewall**: `sudo ufw status`
2. **Vérifier les connexions**: `ss -tnp | grep "5000[1-8]"`
3. **Vérifier program_state**: doit être `PLAYING`
4. **Vérifier robot_program_running**: doit être `true`

### Le programme External Control ne démarre pas

- Vérifier que le robot est en **mode REMOTE**
- Vérifier que l'IP du PC ROS2 est correcte (192.168.1.105)
- Vérifier que le port correspond (50002 pour gripper, 50006 pour screwdriver)

### Erreur de calibration

Le driver affiche un warning sur la calibration. Pour le corriger:
```bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.1.101 \
  target_filename:=/home/robots/wwro_ws/src/wwro_description/config/gripper_calibration.yaml
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  PC ROS2 (192.168.1.105)                                            │
│  ├── ros2_control_node (ports 50001-50008 en écoute)               │
│  ├── gripper_scaled_joint_trajectory_controller                     │
│  ├── screwdriver_scaled_joint_trajectory_controller                 │
│  └── twofg7_gripper_controller                                      │
│                                                                     │
│         │ TCP (RTDE 500Hz)              │ TCP (RTDE 500Hz)          │
│         ▼                               ▼                           │
│  ┌─────────────────┐             ┌─────────────────┐               │
│  │ Robot Gripper   │             │ Robot Screwdriver│              │
│  │ 192.168.1.101   │             │ 192.168.1.103    │              │
│  │ External Control│             │ External Control │              │
│  │ Port: 50002     │             │ Port: 50006      │              │
│  └─────────────────┘             └─────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Document créé le 14 janvier 2026*


    <!-- Gripper finger joints for visualization -->
    <joint name="gripper_finger_left_joint" type="prismatic">
      <parent link="onrobot_2fg7_gripper"/>
      <child link="gripper_finger_left"/>
      <origin xyz="0 0.02 0.1" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="0" upper="0.025" effort="100" velocity="0.1"/>
    </joint>

    <link name="gripper_finger_left">
      <visual>
        <geometry>
          <box size="0.01 0.01 0.05"/>
        </geometry>
        <material name="blue"/>
      </visual>
    </link>

    <joint name="gripper_finger_right_joint" type="prismatic">
      <parent link="onrobot_2fg7_gripper"/>
      <child link="gripper_finger_right"/>
      <origin xyz="0 -0.02 0.1" rpy="0 0 0"/>
      <axis xyz="-1 0 0"/>
      <limit lower="0" upper="0.025" effort="100" velocity="0.1"/>
      <mimic joint="gripper_finger_left_joint" multiplier="-1"/>
    </joint>

    <link name="gripper_finger_right">
      <visual>
        <geometry>
          <box size="0.01 0.01 0.05"/>
        </geometry>
        <material name="blue"/>
      </visual>
    </link>

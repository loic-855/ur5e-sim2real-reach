#!/usr/bin/env python3
"""
Send target joint positions to UR5e via RTDE input registers.
The impedance_control.script on the robot reads these values continuously.
"""

import argparse
import time
import rtde_io
import rtde_receive



def send_positions(robot_ip: str, q_des: list):
    """Send target positions to robot via input registers 24-29 (RTDE reserved)."""
    rtde_io_ = rtde_io.RTDEIOInterface(robot_ip)
    rtde_receive_ = rtde_receive.RTDEReceiveInterface(robot_ip)
    register_list = [18, 19, 20, 21, 22, 42]
    
    print(f"✓ Connected to {robot_ip}")
    print(f"  Sending target positions: {[round(q, 3) for q in q_des]}")
    
    # Write target positions to input registers 24-29 (RTDE reserved)
    for i in range(6):
        rtde_io_.setInputDoubleRegister(register_list[i], q_des[i])
        print(f"  Register {register_list[i]}: {q_des[i]:.3f} rad")
    
    print(f"\nKeeping connection open... Press Ctrl+C to stop")
    try:
        while True:
            # Monitor current positions
            q_current = rtde_receive_.getActualQ()
            print(f"Current positions: {[round(q, 3) for q in q_current]}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n✓ Stopped")
        rtde_io_.disconnect()
        rtde_receive_.disconnect()
        return True


def interactive_mode(robot_ip: str):
    """Interactive mode: send positions one by one."""
    
    rtde_io_ = rtde_io.RTDEIOInterface(robot_ip)
    rtde_receive_ = rtde_receive.RTDEReceiveInterface(robot_ip)


    
    print(f"✓ Connected to {robot_ip}")
    print("Interactive mode - enter target positions (6 joints)")
    print("Example: 0.0 -1.57 0.0 -1.57 0.0 0.0")
    
    try:
        while True:
            user_input = input("\nEnter positions (or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                break
            
            try:
                q_des = [float(x) for x in user_input.split()]
                if len(q_des) != 6:
                    print("✗ Please enter exactly 6 values")
                    continue
                
                # Write to registers 24-29
                for i in range(6):
                    rtde_io_.setInputDoubleRegister(24 + i, q_des[i])
                
                print(f"✓ Sent: {[round(q, 3) for q in q_des]}")
                
                # Show current state
                q_current = rtde_receive_.getActualQ()
                print(f"  Current: {[round(q, 3) for q in q_current]}")
            except ValueError:
                print("✗ Invalid input. Enter 6 space-separated numbers")
    
    except KeyboardInterrupt:
        print("\n")
    finally:
        rtde_io_.disconnect()
        rtde_receive_.disconnect()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Send target positions to UR5e via RTDE")
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address")
    parser.add_argument("--pos", type=float, nargs=6, 
                        help="Target positions for 6 joints (space-separated)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode (send positions on demand)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.ip)
    elif args.pos:
        send_positions(args.ip, args.pos)
    else:
        print("Error: Specify either --pos or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()

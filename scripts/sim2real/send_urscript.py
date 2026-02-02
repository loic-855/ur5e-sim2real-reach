import socket
import argparse
from pathlib import Path

# initialize variables
robotIP = "192.168.1.101"
PRIMARY_PORT = 30001
SECONDARY_PORT = 30002
REALTIME_PORT = 30003

# URScript single command example
urscript_command = "movej([-0.0,-1.57,0.0,-1.57,0,0],a=0.4, v=0.5, t=0, r=0)"

# URScript full program example (multi-line)
urscript_program = """def my_program():
  movej([-0.0,-1.5,0.1,-1.2,1.8,0],a=1.4, v=1.05)
  sleep(1)
  movej([-0.5,-1.7,0.1,-1.2,1.8,0],a=1.4, v=1.05)
  sleep(1)
    movej([0.0,-1.57,0.0,-1.57,0.0,0],a=1.4, v=1.05)
end
"""

# Creates new line
new_line = "\n"

def send_urscript_command(command: str):
    """
    This function takes the URScript command defined above, 
    connects to the robot server, and sends 
    the command to the specified port to be executed by the robot.

    Args:
        command (str): URScript command
        
    Returns: 
        None
    """
    try:
        # Create a socket connection with the robot IP and port number defined above
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((robotIP, PRIMARY_PORT))

        # Appends new line to the URScript command (the command will not execute without this)
        command = command+new_line
        
        # Send the command
        s.sendall(command.encode('utf-8'))
        
        # Close the connection
        s.close()

    except Exception as e:
        print(f"An error occurred: {e}")

def send_urscript_file(filepath: str):
    """
    Load and send a URScript program from a .script file.

    Args:
        filepath (str): Path to the URScript file
        
    Returns: 
        None
    """
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return
    
    with open(path, 'r') as f:
        program = f.read()
    
    print(f"Sending program from: {filepath}")
    send_urscript_command(program)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send URScript to robot")
    parser.add_argument("--file", "-f", type=str, help="Path to .script file to send")
    parser.add_argument("--program", "-p", action="store_true", help="Send example multi-line program")
    args = parser.parse_args()
    
    if args.file:
        # Send program from file
        send_urscript_file(args.file)
    elif args.program:
        # Send the multi-line program example
        send_urscript_command(urscript_program)
    else:
        # Send single command (default behavior)
        send_urscript_command(urscript_command)
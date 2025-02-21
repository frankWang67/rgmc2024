#!/home/wshf/miniconda3/envs/graspnet/bin/python

import serial
import time

class SerialGripper:
    keys = ["init", "check_init", "set_pos", "get_pos", "turn_off_drop_check"]

    def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=1, wait_time=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.wait_time = wait_time
        self.ser = serial.Serial(port, baudrate, timeout=timeout)

        if not self.check_init():
            self.init()

        self.turn_off_drop_check()

    def get_byte_command(self, key, *args):
        if key == "init":
            return bytes.fromhex("FF FE FD FC 01 08 02 01 00 00 00 00 00 FB")
        if key == "check_init":
            return bytes.fromhex("FF FE FD FC 01 08 02 00 00 00 00 00 00 FB")
        if key == "set_pos":
            pos = int(args[0])
            pos_hex = format(pos, '02X')
            return bytes.fromhex(f"FF FE FD FC 01 06 02 01 00 {pos_hex} 00 00 00 FB")
        if key == "get_pos":
            return bytes.fromhex("FF FE FD FC 01 06 02 00 00 00 00 00 00 FB")
        if key == "turn_off_drop_check":
            return bytes.fromhex("FF FE FD FC 01 15 01 01 00 00 00 00 00 FB")
        
        raise ValueError("Invalid key.")

    def send_command(self, byte_data):
        try:
            self.ser.write(byte_data)
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        except ValueError:
            print("Invalid hex data.")

    def check_init(self):
        byte_data = self.get_byte_command("check_init")
        self.send_command(byte_data)
        time.sleep(self.wait_time)
        data = self.ser.read_all()
        # 返回 ：FF FE FD FC 01 08 02 00 00 01 00 00 00 FB （初始化完成）
        # 返回 ：FF FE FD FC 01 08 02 00 00 00 00 00 00 FB （初始化未完成）
        hex = data.hex()
        print(f"{hex=}")
        res = bool(int(hex[19]))

        return res
    
    def init(self):
        byte_data = self.get_byte_command("init")
        self.send_command(byte_data)
        time.sleep(self.wait_time)
        self.ser.read_all()
        
    def set_pos(self, pos):
        byte_data = self.get_byte_command("set_pos", pos)
        self.send_command(byte_data)
        time.sleep(self.wait_time)
        self.ser.read_all()

    def get_pos(self):
        byte_data = self.get_byte_command("get_pos")
        self.send_command(byte_data)
        time.sleep(self.wait_time)
        data = self.ser.read_all()
        # 返回：FF FE FD FC 01 06 02 00 00 3C 00 00 00 FB （60%位置）
        hex = data.hex()
        print(f"{hex=}")
        pos = int(hex[18:20], 16)

        return pos
    
    def turn_off_drop_check(self):
        byte_data = self.get_byte_command("turn_off_drop_check")
        self.send_command(byte_data)
        time.sleep(self.wait_time)
        self.ser.read_all()

if __name__ == "__main__":
    gripper = SerialGripper()

    time.sleep(1)
    # print(gripper.get_pos())
    # time.sleep(1)
    # gripper.set_pos(100)
    # time.sleep(1)
    # print(gripper.get_pos())
    # time.sleep(1)
    # gripper.set_pos(30)
    # time.sleep(1)
    # print(gripper.get_pos())
    time.sleep(1)
    gripper.set_pos(0)
    # time.sleep(1)
    # print(gripper.get_pos())
import socket
import cv2
import pickle
import struct
import matplotlib.pyplot as plt

# Client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.1.225'  # Server IP address
port = 9999  # Server port
client_socket.connect((host_ip, port))

data = b""
payload_size = struct.calcsize("Q")  # Q: unsigned long long integer (8 bytes)

try:
    while True:
        # Receive the size of the payload
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)  # 4K size buffer
            if not packet:
                break
            data += packet

        if not data:
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive the actual payload (image data)
        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        plt.imshow(frame)
        plt.show()

except:
    client_socket.close()
    cv2.destroyAllWindows()
import socket
import cv2
import pickle
import struct
import matplotlib.pyplot as plt

class Connect():
    def __init__(self):
        try:
            self.host = ""
            self.port = 9999
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("Socket created")
            self.bind_socket()
        except socket.error as msg:
            print("Socket couldn't be created.")

    def bind_socket(self):
        try:
            print("Binding the port " + str(self.port))
            self.s.bind((self.host, self.port))
            self.s.listen(5)
            self.socket_accept()
        except socket.error as msg:
            print("Socket binding error " + str(msg) + " Retrying...")
            self.bind_socket()

    def socket_accept(self):
        conn, address = self.s.accept()
        print("Connection established at IP: " + address[0] + " & Port: " + str(address[1]))
        self.send_command(conn)
        conn.close()

    def send_command(self, conn):
        try:
            cap = cv2.VideoCapture(4)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("Failed to capture image")
                return

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = image[260:310, 200:250]

            sr_fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
            sr_fsrcnn.readModel("SuperResolution/models/FSRCNN/FSRCNN_x4.pb")
            sr_fsrcnn.setModel("fsrcnn", 4)
            result_fsrcnn = sr_fsrcnn.upsample(image)

            a = pickle.dumps(result_fsrcnn)
            b = pickle.dumps(image)
            original_image = struct.pack("Q", len(a)) + a
            supres_image = struct.pack("Q", len(b)) + b
            conn.sendall(original_image)
            conn.sendall(supres_image)
            

        except Exception as e:
            conn.close()
            print("\nConnection closed.")
        

if __name__ == "__main__":
    connect = Connect()

# REAL TIME SUPER RESOLUTION 
import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(4)
ret, frame = cap.read()
if not ret:
	print("Failed to capture image")
	cap.release()
	exit()
	
cap.release()

image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = image[260:310, 200:250]

# # EDSR
# sr_edsr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr_edsr.readModel("SuperResolution/models/EDSR/EDSR_x4.pb")
# sr_edsr.setModel("edsr",4)

# start_edsr = time.time()
# result_edsr = sr_edsr.upsample(image)
# end_edsr = time.time()

# # ESPCN 
# sr_espcn = cv2.dnn_superres.DnnSuperResImpl_create()
# sr_espcn.readModel("SuperResolution/models/ESPCN/ESPCN_x4.pb")
# sr_espcn.setModel("espcn",4)

# start_espcn = time.time()
# result_espcn = sr_espcn.upsample(image)
# end_espcn = time.time()

# # FSRCNN 
# sr_fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
# sr_fsrcnn.readModel("SuperResolution/models/FSRCNN/FSRCNN_x4.pb")
# sr_fsrcnn.setModel("fsrcnn",4)

# start_fsrcnn = time.time()
# result_fsrcnn = sr_fsrcnn.upsample(image)
# end_fsrcnn = time.time()

# LapSRN 
sr_lapsrn = cv2.dnn_superres.DnnSuperResImpl_create()
sr_lapsrn.readModel("SuperResolution/models/LapSRN/LapSRN_x4.pb")
sr_lapsrn.setModel("lapsrn",4)

start_lapsrn = time.time()
result_lapsrn = sr_lapsrn.upsample(image)
end_lapsrn = time.time()


# Plot of original pre-trained models
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title(f"Original image\nwith shape {image.shape}")

# plt.subplot(1, 2, 2)
# plt.imshow(result_edsr)
# plt.title(f"EDSR upsampled image\nwith shape {result_edsr.shape}\nwith time {end_edsr-start_edsr} seconds")

# plt.subplot(1, 2, 2)
# plt.imshow(result_espcn)
# plt.title(f"ESPCN upsampled image\nwith shape {result_espcn.shape}\nwith time {end_espcn-start_espcn} seconds")

# plt.subplot(1, 2, 2)
# plt.imshow(result_fsrcnn)
# plt.title(f"FSRCNN upsampled image\nwith shape {result_fsrcnn.shape}\nwith time {end_fsrcnn-start_fsrcnn} seconds")

plt.subplot(1, 2, 2)
plt.imshow(result_lapsrn)
plt.title(f"LapSRN upsampled image\nwith shape {result_lapsrn.shape}\nwith time {end_lapsrn-start_lapsrn} seconds")

plt.tight_layout()
plt.show()

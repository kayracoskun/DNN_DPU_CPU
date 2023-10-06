# SUPER RESOLUTION 
# using pre-trained models (.pb)
# https://learnopencv.com/super-resolution-in-opencv/

import cv2
import matplotlib.pyplot as plt

# Read image
# img = cv2.imread("SuperResolution/images/AI-Courses-By-OpenCV-Github.jpg")
img = cv2.imread("SuperResolution/images/kayra_bedirhan.png")

# Cropout OpenCV logo
# img = img[:80,850:]

# EDSR
sr_edsr = cv2.dnn_superres.DnnSuperResImpl_create()
sr_edsr.readModel("SuperResolution/models/EDSR/EDSR_x4.pb")
sr_edsr.setModel("edsr",4)
result_edsr = sr_edsr.upsample(img)

# ESPCN 
sr_espcn = cv2.dnn_superres.DnnSuperResImpl_create()
sr_espcn.readModel("SuperResolution/models/ESPCN/ESPCN_x4.pb")
sr_espcn.setModel("espcn",4)
result_espcn = sr_espcn.upsample(img)

# FSRCNN 
sr_fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
sr_fsrcnn.readModel("SuperResolution/models/FSRCNN/FSRCNN_x4.pb")
sr_fsrcnn.setModel("fsrcnn",4)
result_fsrcnn = sr_fsrcnn.upsample(img)

# LapSRN 
sr_lapsrn = cv2.dnn_superres.DnnSuperResImpl_create()
sr_lapsrn.readModel("SuperResolution/models/LapSRN/LapSRN_x4.pb")
sr_lapsrn.setModel("lapsrn",4)
result_lapsrn = sr_lapsrn.upsample(img)

# LapSRN Our Model
sr_lapsrn_trained = cv2.dnn_superres.DnnSuperResImpl_create()
sr_lapsrn_trained.readModel("SuperResolution/models/LapSRN/kefir_x4.pb")
sr_lapsrn_trained.setModel("lapsrn",4)
result_lapsrn_trained = sr_lapsrn_trained.upsample(img)

# Resized image
resized = cv2.resize(img,dsize=None,fx=4,fy=4)

# Plot of original pre-trained models
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Original image (LR)
axs[0, 0].imshow(img[:,:,::-1])
axs[0, 0].set_title("Original Image")

# OpenCV upscaled
axs[0, 1].imshow(resized[:,:,::-1])
axs[0, 1].set_title("OpenCV upscaled")

# EDSR upscaled
axs[0, 2].imshow(result_edsr[:,:,::-1])
axs[0, 2].set_title("EDSR upscaled")

# ESPCN upscaled
axs[1, 0].imshow(result_espcn[:,:,::-1])
axs[1, 0].set_title("ESPCN upscaled")

# FSRCNN upscaled
axs[1, 1].imshow(result_fsrcnn[:,:,::-1])
axs[1, 1].set_title("FSRCNN upscaled")

# LapSRN upscaled
axs[1, 2].imshow(result_lapsrn[:,:,::-1])
axs[1, 2].set_title("LapSRN upscaled")

plt.tight_layout()
plt.show()

# Plot of our trained model
fig1, axs1 = plt.subplots(1, 3, figsize=(12, 8))

# Original image (LR)
axs1[0].imshow(img[:,:,::-1])
axs1[0].set_title("Original Image")

# LapSRN upscaled
axs1[1].imshow(result_lapsrn[:,:,::-1])
axs1[1].set_title("LapSRN upscaled")

# LapSRN upscaled (trained)
axs1[2].imshow(result_lapsrn_trained[:,:,::-1])
axs1[2].set_title("Our Trained LapSRN upscaled")

plt.tight_layout()
plt.show()
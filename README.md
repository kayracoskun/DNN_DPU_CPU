# DNN_DPU_CPU
Deep Neural Network models, training and inference codes. Implementations on both Python and C++. Tests on CPU, GPU and DPU.

### Super Resolution
Four models are available for super resolution:
- EDSR: https://github.com/Saafke/EDSR_Tensorflow
- ESPCN: https://github.com/fannymonori/TF-ESPCN
- FSRCNN: https://github.com/Saafke/FSRCNN_Tensorflow
- LapSRN: https://github.com/fannymonori/TF-LapSRN

Dependencies:
- matplotlib pyplot
- opencv

OpenCV version should be 4.3 and higher. To check OpenCV version:
```
python
import cv2
print(cv2.__version__)
```
To install OpenCV or upgrade the version:

```
pip install pip install opencv-contrib-python
pip install opencv-contrib-python --upgrade
```

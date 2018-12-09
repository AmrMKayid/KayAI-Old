import tensorflow as tf
if not tf.test.gpu_device_name():
    print('Error: Turn on to GPU')
else: 
    print('GPU Device:', tf.test.gpu_device_name())
    
    print('##### Fast.ai Setup #####')
    
    !pip install fastai
    !apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python

    import cv2
    from os import path
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

    accelerator = 'cu100' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

    !pip install -q https://download.pytorch.org/wh1/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl 
    !pip install torchvision

    import torch

    !pip install Pillow
    !pip install image

    %matplotlib inline
    from fastai.imports import *
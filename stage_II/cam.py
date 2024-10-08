import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import torch
from qdnet.models.resnest import Resnest
import torch.nn as nn
from resnest.torch import resnest50
from resnest.torch import resnest101
from resnest.torch import resnest200
from resnest.torch import resnest269

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM, \
                             LayerCAM    

# from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

import os

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use-cuda', action='store_true', default=True,
#                         help='Use NVIDIA GPU acceleration')
#     parser.add_argument('--image-path', type=str, default='./data/img/0male/0(2).jpg',
#                         help='Input image path')
#     parser.add_argument('--aug_smooth', action='store_true',
#                         help='Apply test time augmentation to smooth the CAM')
#     parser.add_argument('--eigen_smooth', action='store_true',
#                         help='Reduce noise by taking the first principle componenet'
#                         'of cam_weights*activations')
#     parser.add_argument('--method', type=str, default='gradcam',
#                         choices=['gradcam', 'gradcam++', 
#                                  'scorecam', 'xgradcam',
#                                  'ablationcam', 'eigencam', 
#                                  'eigengradcam', 'layercam'],
#                         help='Can be gradcam/gradcam++/scorecam/xgradcam'
#                              '/ablationcam/eigencam/eigengradcam/layercam')

#     #Add parameters
#     parser.add_argument('--result-path', type=str, default='./data/img/0male/0(2).jpg',
#                         help='Output image path')

#     parser.add_argument('--pre-target-now', type=int, default=None,
#                         help='predict type')

#     args = parser.parse_args()
#     args.use_cuda = args.use_cuda and torch.cuda.is_available()
#     if args.use_cuda:
#         print('Using GPU for acceleration')
#     else:
#         print('Using CPU for computation')

#     return args

methods = \
    {"gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM}

def compute_CAM(
    model, 
    image_path,
    method = 'gradcam', 
    use_cuda = True,
    aug_smooth = False,
    eigen_smooth = False,
):

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.enet.layer4[-1]

    cam = methods[method](model=model,
                               target_layer=target_layer,
                               use_cuda=use_cuda)

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

    # Only generate the CAM for target 1 (positive)
    target_category = 1

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=aug_smooth,
                        eigen_smooth=eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # # Stack CAM weights on the original images.
    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor, target_category=target_category)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    
    if use_cuda:
        torch.cuda.empty_cache()
    
    return cam_image, grayscale_cam

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

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./data/img/0male/0(2).jpg',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    #Add parameters
    parser.add_argument('--result-path', type=str, default='./data/img/0male/0(2).jpg',
                        help='Output image path')

    parser.add_argument('--pre-target-now', type=int, default=None,
                        help='predict type')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}



    # model = models.resnet101(pretrained=False)

    # model = Resnest("resnest101")
    model = eval("resnest101")(pretrained=False)
    model = Resnest(
        enet_type = "resnest101",
        out_dim = int(3),
        drop_nums = int(1),
        pretrained = False,
        metric_strategy = False
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Change the model to 200 epochs
    model_file = './resnest101/weight/best_fold.pth'
    model.load_state_dict(torch.load(model_file), strict=True)

    # raw_state_dict = torch.load(model_file)
    # from collections import OrderedDict
    #
    # state_dict = {}
    # for k,v in raw_state_dict.items():
    #     if k.startswith('enet.'):
    #         namekey = k[5:]
    #     else:
    #         namekey = k
    #     state_dict[namekey] = v
    # model.load_state_dict(state_dict, strict=True)


    # try:  # single GPU model_file
    #     model.load_state_dict(torch.load(model_file), strict=True)
    # except:  # multi GPU model_file
    #     state_dict = torch.load(model_file)
    #     state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    #     model.load_state_dict(state_dict, strict=True)


    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.enet.layer4[-1]

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.

    target_category = None
    # target_category = args.pre_target_now
    # target_category = 1

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)

    #modify
    grayscale_cam_ori = grayscale_cam
    grayscale_cam[np.where(grayscale_cam < 10/255)] = 0
    grayscale_cam[np.where(grayscale_cam >= 10 / 255)] = 1

    cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam.jpg'), cam_image)
    cv2.imwrite(os.path.join(args.result_path, f'{args.method}_gb.jpg'), gb)
    cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_gb.jpg'), cam_gb)
    cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam.jpg'), np.uint8(grayscale_cam * 255))
    cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam_ori.jpg'),
                np.uint8(grayscale_cam_ori * 255))


    # if args.pre_target_now == 1:
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam.jpg'), cam_image)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_gb.jpg'), gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_gb.jpg'), cam_gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam.jpg'), np.uint8(grayscale_cam * 255))
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam_ori.jpg'), np.uint8(grayscale_cam_ori * 255))
    #
    #     # 1
    #     cam_image_1 = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #
    #     # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    #     cam_image_1 = cv2.cvtColor(cam_image_1, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_1.jpg'), cam_image_1)
    #
    #     # 2
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_2.jpg'), cam_image)
    #
    #     # 3
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_3.jpg'), cam_image_1)
    #
    #
    # else:
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam.jpg'), cam_image)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_gb.jpg'), gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_gb.jpg'), cam_gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam.jpg'), np.uint8(grayscale_cam * 0))
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam_ori.jpg'), np.uint8(grayscale_cam_ori * 255))
    #
    #     # 1
    #     cam_image_1 = show_cam_on_image(rgb_img, grayscale_cam * 0, use_rgb=True)
    #
    #     # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    #     cam_image_1 = cv2.cvtColor(cam_image_1, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_1.jpg'), cam_image_1)
    #
    #     # 2
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_2.jpg'), cam_image_1)
    #
    #     # 3
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_3.jpg'), cam_image)

    # grayscale_cam[np.where(grayscale_cam < 10/255)] = 0
    # grayscale_cam[np.where(grayscale_cam >= 10 / 255)] = 1
    # if args.pre_target_now == 1:
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam.jpg'), cam_image)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_gb.jpg'), gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_gb.jpg'), cam_gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam.jpg'), np.uint8(grayscale_cam * 255))
    # else:
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam.jpg'), cam_image)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_gb.jpg'), gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_cam_gb.jpg'), cam_gb)
    #     cv2.imwrite(os.path.join(args.result_path, f'{args.method}_grayscale_cam.jpg'), np.uint8(grayscale_cam * 0))
    #
    #
    #

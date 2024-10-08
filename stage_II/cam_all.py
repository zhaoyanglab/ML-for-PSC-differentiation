import os, argparse
import torch
import cv2
import tqdm
import numpy as np
from qdnet.conf.config import load_yaml
from qdnet.dataset.dataset import get_df
from cam import compute_CAM
from qdnet.models.resnest import Resnest

def init_model(statedict, use_cuda = True):
    # model = Resnest("resnest101")
    #model = eval("resnest101")(pretrained=False)
    model = Resnest(
        enet_type = "resnest101",
        out_dim = 2,
        drop_nums = 1,
        pretrained = False,
        metric_strategy = False
    )
    
    model.load_state_dict(statedict)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    return model

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path',default='conf/resnest101.yaml' , help='config file path')
parser.add_argument('--ignore_predicted', action = 'store_true', help='whether to ignore images that have been predicted before')
# parameters for CAM
parser.add_argument('--use-cuda', action='store_true',
                    help='Use NVIDIA GPU acceleration')
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
args = parser.parse_args()
config = load_yaml(args.config_path, args)
args.use_cuda = args.use_cuda and torch.cuda.is_available()
if args.use_cuda:
        print('Using GPU for acceleration')
else:
    print('Using CPU for computation')

df_test, _ = get_df( config["data_dir"], config["auc_index"] , stage = "test")
test_path = df_test['filepath']
pre_target = np.load('resnest101/oofs/best_pred_target.npy')

resultpath = 'img_result/'
os.makedirs(resultpath, exist_ok=True)

model_file = './resnest101/weight/best_fold.pth'
model_state_dict = torch.load(model_file)
model = init_model(model_state_dict, args.use_cuda)

for i in tqdm.tqdm(range(len(test_path))):
    
    resultpath_folder = os.path.join(
        resultpath, 
        "%d_%s" % (i, os.path.splitext(os.path.split(test_path[i])[1])[0])
    )
    os.makedirs(resultpath_folder, exist_ok=True)
    
    if args.ignore_predicted and (len(os.listdir(resultpath_folder))>0): # has been predicted before, ignore. 
        continue

    # print('Processing (%d/%d) %s' % (i+1, len(test_path), test_path[i]))  
    model = init_model(model_state_dict, args.use_cuda)
    cam_image, grayscale_cam = compute_CAM(
        model = model, 
        image_path = test_path[i], 
        method = args.method,
        use_cuda= args.use_cuda, 
        aug_smooth= args.aug_smooth, 
        eigen_smooth = args.eigen_smooth,
    )
    
    binary_cam = (grayscale_cam >= 10/255).astype(np.uint8)
    if (pre_target[i] == 0):
        binary_cam[:, :] = 0

    cv2.imwrite(os.path.join(resultpath_folder, f'{args.method}_cam.jpg'), cam_image)
    cv2.imwrite(
        os.path.join(resultpath_folder, f'{args.method}_cam_binary.jpg'), 
        binary_cam * 255
    )
    cv2.imwrite(
        os.path.join(resultpath_folder, f'{args.method}_cam_grayscale.jpg'),
        np.uint8(grayscale_cam * 255)
    )
    



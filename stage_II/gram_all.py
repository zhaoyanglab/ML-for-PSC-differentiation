import os
from qdnet.conf.config import load_yaml
import argparse
from qdnet.dataset.dataset import get_df
import numpy as np

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path',default='conf/resnest101.yaml' , help='config file path')
# parser.add_argument('--n_splits', help='n_splits', type=int)
args = parser.parse_args()
config = load_yaml(args.config_path, args)

_, df_test, _ = get_df( config["data_dir"], config["auc_index"]  )
# filepath = r'test_img/0\0_CD23'
resultpath = r'img_result/img/test\image_1_22_bestepochs'

#resultpath exists？
isExists=os.path.exists(resultpath)
if not isExists:
    os.makedirs(resultpath)

# filelist = os.listdir(filepath)
test_path = df_test['filepath']
pre_target = np.load('resnest101/oofs/best_pred_target.npy')

for i in range(len(test_path)):
    file_on = test_path[i]
    pre_target_now = pre_target[i]
    file_on = os.path.split(file_on)
    file_on = file_on[1]
    resultpath_subpath , _ = os.path.splitext(file_on)
    resultpath_subpath = f"{i}_{resultpath_subpath}"
    resultpath_new = os.path.join(resultpath, resultpath_subpath)
    test_path_i = '"' + test_path[i] + '"'
    if not os.path.exists(resultpath_new):
        os.makedirs(resultpath_new)
        cmd_command = f'python cam.py --image-path {test_path_i} --result-path {resultpath_new} --pre-target-now {pre_target_now} --method gradcam'
        os.system(cmd_command)
        # os.popen(cmd_command)
    elif len(os.listdir(resultpath_new))>0 :
        continue
    else:
        cmd_command = f'python cam.py --image-path {test_path_i} --result-path {resultpath_new} --pre-target-now {pre_target_now} --method gradcam'
        os.system(cmd_command)


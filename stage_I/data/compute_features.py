import cv2, os
import pandas as pd
import numpy as np
from algutils import get_bag_of_features, get_optical_flow, get_fractal_dimension

def compute_features(imgs):

    features = []
    for func in (get_optical_flow, 
                 get_bag_of_features,
                 get_fractal_dimension,
                ):
        features.extend(func(imgs))
        
    return dict(features)

if __name__ == "__main__":
    
    for batch in ("CD01-1", "CD01-2", "CD01-3", "CD01-4"):
        
        if not os.path.exists(batch):
            os.mkdir(batch)
    
        table = []

        for S_id in range(1, 97): # iterate over each well
            
            print("Computing batch %s, S%d ..." % (batch, S_id))
            
            # obtain the image stream
            imgs = np.stack([cv2.imread("./image/%s/S%d/T%d.png" % (batch, S_id, i), cv2.IMREAD_GRAYSCALE) 
                             for i in range(1, 11)])
            
            # compute the features
            features = compute_features(imgs)
            
            features["S_id"] = S_id
            features["batch_name"] = batch
            
            table.append(features)
        
        df = pd.DataFrame(table)
        
        df.to_pickle("%s/%s_features.pkl" % (batch, batch))
        df.to_csv("%s/%s_features.csv" % (batch, batch))
        


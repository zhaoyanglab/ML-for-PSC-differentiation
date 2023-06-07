import numpy as np

def compute_differentiation_efficiency_index(img, thres = 0.5):
    '''
        Compute the differentiaiton efficiency index (i.e. total fluorescence intensity) 
        of a fluorescence image.
    '''
    
    h, w = img.shape
    return img[img > thres].sum() / (h * w)
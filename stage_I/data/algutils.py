import cv2
import numpy as np

def get_optical_flow(imgs):
    
    def compute_optical_flow(img1, img2, thres = 4, winsize = 16):
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, winsize, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return mag[mag > thres].mean()

    res = []
    for j in range(imgs.shape[0] - 1):
        res.append( ( "optical_flow_%d" % j, compute_optical_flow(imgs[j], imgs[j+1]) ))

    return res

from skimage.filters.rank import entropy
from skimage.morphology import disk

def get_bag_of_features(imgs):

    width = imgs.shape[1]
    roi = np.zeros((width, width), dtype = np.uint8)
    r = int(width / 2 * 0.9)
    tmp = disk(r).astype(np.uint8)
    roi[width // 2 - r : width // 2 + r + 1, width // 2 - r : width // 2 + r + 1] = tmp
    del tmp, r

    res = []

    # local entropy
    masks = []
    ele = disk(8)
    thres = 3
    for j in range(imgs.shape[0]):
        img = imgs[j]
        entr_img = entropy(img, ele)
        mask = ((entr_img > thres) & roi).astype(np.bool)
        masks.append(mask)

        mean_entropy = entr_img[mask].mean()
        res.append( ("local_entropy_%d" % j, mean_entropy) )

    # density
    for j in range(imgs.shape[0]):
        density = imgs[j][masks[j]].mean() / 255
        res.append( ("cell_brightness_%d" % j, density) )
    
    def compute_circumference(mask):
        w = mask.shape[0]
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        C = 0
        for contour in contours:
            C += cv2.arcLength(contour, closed = True)
        return C / w

    # area, circumference, area-circumference ratio
    areas, Cs = [], []
    for j in range(imgs.shape[0]):
        area = masks[j].sum() / (width * width)
        circumference = compute_circumference((masks[j] * 255).astype(np.uint8))
        res.append( ("area_%d" % j, area) )
        res.append( ("circumference_%d" % j, circumference) )
        res.append( ("A_C_ratio_%d" % j, area / circumference) )

        areas.append(area)
        Cs.append(circumference)

    return res

import math

def _get_Nr(gray_img, logscales):
    h, w = gray_img.shape[:2]
    M = min(h,w)
    Nr = np.zeros(logscales.shape, dtype = np.float64)
    for i in range(logscales.shape[0]):      
        s = round(np.e ** logscales[i])
        H = 255*s/M
        box_num = 0 
        for row in range(h//s):
            for col in range(w//s):
                nr = math.ceil((np.max(gray_img[row*s:(row+1)*s, col*s:(col+1)*s])-np.min(gray_img[row*s:(row+1)*s, col*s:(col+1)*s]))/H +1)
                box_num += nr
        Nr[i] = box_num
    return Nr

def differential_box_counting(gray_img):
    h, w = gray_img.shape[:2]
    M = min(h,w)

    logscales = np.linspace(np.log(2), np.log(M//20), num = 16)
    Nr = _get_Nr(gray_img, logscales)
    
    coeffs = np.polyfit(np.log(M) - np.log(np.round(np.e ** logscales)), np.log(Nr), 1)
    return coeffs[0]

def get_fractal_dimension(imgs):

    width = imgs.shape[1]
    roi = np.zeros((width, width), dtype = np.uint8)
    r = int(width / 2 * 0.9)
    tmp = disk(r).astype(np.uint8)
    roi[width // 2 - r : width // 2 + r + 1, width // 2 - r : width // 2 + r + 1] = tmp
    del tmp, r

    res = []
    for j in range(imgs.shape[0]):
        img = imgs[j].copy()
        img[roi == 0] = 127
        res.append( ( "fractal_dimension_%d" % j, differential_box_counting(img) ))
    return res
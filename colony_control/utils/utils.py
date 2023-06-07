import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
from skimage.filters import rank
from utils.local_features import BagOfKeypointsRepresentation

# Area, Circumference
def compute_area_circumference(mask_dir):
    
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8) * 255
    width = mask.shape[0]
    
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    C = 0
    for contour in contours:
        
        contour = contour[:, 0, :]
        N = len(contour)
        for i in range(N):
            x0, y0 = contour[i, :]
            x1, y1 = contour[(i+1) % N, :]
            if ((x0 == x1) and ((x0 == 0) or (x0 == width - 1))) or ((y0 == y1) and ((y0 == 0) or (y0 == width - 1))):
                continue
            C += np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
    Area = (mask > 0).sum() / (width * width)
    Circumference = C / width

    return {
        "Area" : Area, 
        "Circumference" : Circumference, 
    }

# Area/Circumference Ratio, Solidity, Convexity, Circularity
def compute_solidity_convexity(mask_dir):

    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE) 
    width = mask.shape[0]
    
    total_weight = 0.
    
    solidity = 0.
    convexity = 0.
    A_C_ratio = 0.
    circularity = 0.
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,connectivity = 8)
    
    if (num_labels == 1):
        # If there is no region, compute the result for a circle with radius r by letting r -> 0
        return {
            "Solidity" : 1, 
            "Convexity": 1, 
            "Area/Circumference Ratio": 0,
            "Circularity": 1, 
        }
    
    for i in range(1, num_labels):
        y, x, w, h, num_pixels = stats[i]
        region = (labels[x:x+h, y:y+w] == i).astype(np.uint8)
        
        contours, _ = cv2.findContours(region * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        Areas = [cv2.contourArea(contour) for contour in contours]
        idx = np.argsort(Areas)[-1]
        cvx_hull = cv2.convexHull(contours[idx])
        cvx_hull_area = cv2.contourArea(cvx_hull)
        cvx_hull_circumference = cv2.arcLength(cvx_hull, closed = True) 
        
        area = (region>0).sum()
        circumference = sum([cv2.arcLength(contour, closed = True) for contour in contours])
        
        total_weight += area
        
        solidity += (area / cvx_hull_area) * area
        convexity += (cvx_hull_circumference / circumference) * area
        A_C_ratio += (area / circumference) * area
        circularity += (4 * np.pi * area / (circumference ** 2)) * area
    
    solidity /= total_weight
    convexity /= total_weight
    A_C_ratio /= (width * total_weight)
    circularity /= total_weight

    return {
        "Solidity" : solidity, 
        "Convexity" : convexity,
        "Area/Circumference Ratio": A_C_ratio,
        "Circularity": circularity
    }

# Spacing
def compute_spacing(mask_dir):

    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    width = mask.shape[0]

    background_region = 1 - mask//255

    if background_region.sum() == 0:

        Spacing = 0

    else:

        Spacing = 0
        tot_weight = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background_region * 255,connectivity = 8)
        for i in range(1, num_labels):
            y, x, w, h, num_pixels = stats[i]
            sub_region = (labels[x:x+h, y:y+w] == i).astype(np.uint8)

            skeleton = morphology.skeletonize(sub_region)
            dis = ndimage.distance_transform_edt(sub_region)

            Spacing += dis[skeleton > 0].max() / width * num_pixels
            tot_weight += num_pixels

        Spacing /= tot_weight
        Spacing *= 2

    return {
        "Spacing" : Spacing
    }

# Max, Min, Mean, Min/Max, Std of Centrod-Contour Distances (CCD)
def compute_CCD_features(mask_dir): 

    def fill_contour(contour):
    
        '''
        contour: of shape (N, 2)

        densely fill the contour returned by opencv.
        '''

        lastx, lasty = (contour[0, 0], contour[0, 1])
        res = [(lastx, lasty)]
        i = 1
        N = len(contour)
        for i in range(N+1):

            x, y = contour[i%N, :]
            while (abs(x - lastx) > 1.5) or (abs(y - lasty)> 1.5):

                if (abs(x - lastx) > 1.5):
                    lastx += np.sign(x - lastx)
                if (abs(y - lasty) > 1.5):
                    lasty += np.sign(y - lasty)
                res.append((lastx, lasty))
            res.append((x, y))
            lastx, lasty = x, y

        return np.asarray(res)
    
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    width = mask.shape[0]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,connectivity = 8)
    N = num_labels - 1
    
    Displacement_Max = 0
    Displacement_Min = 0
    Displacement_MinMaxRatio = 0
    Displacement_Mean = 0
    Displacement_Std = 0  
    Total_Area = 0
    
    for i in range(N):
        
        y, x, w, h, num_pixels = stats[i+1]
        region = (labels[x:x+h, y:y+w] == i+1).astype(np.uint8)
        region_area = num_pixels / (width * width)
        contours, _ = cv2.findContours(region * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        centroid = centroids[i + 1, :]
        contours = [fill_contour(contour[:, 0, :])
                    for contour in contours
                   ]
        r = np.concatenate([contour - centroid[None, :] 
                            for contour in contours
                           ], axis = 0)
        r_norm = np.linalg.norm(r, axis = 1) / width
        Displacement_Max += r_norm.max() * num_pixels
        Displacement_Min += r_norm.min() * num_pixels
        Displacement_MinMaxRatio += r_norm.min() / r_norm.max() * num_pixels
        Displacement_Mean += r_norm.mean() * num_pixels
        Displacement_Std += r_norm.std() * num_pixels
        Total_Area += num_pixels

    Displacement_Max /= Total_Area
    Displacement_Min /= Total_Area
    Displacement_MinMaxRatio /= Total_Area
    Displacement_Mean /= Total_Area
    Displacement_Std /= Total_Area
    
    return {
        "Max Centroid-Contour Distances" : Displacement_Max, 
        "Min Centroid-Contour Distances" : Displacement_Min, 
        "Min/Max Ratio Centroid-Contour Distances" : Displacement_MinMaxRatio, 
        "Mean of Centroid-Contour Distances" : Displacement_Mean, 
        "Std of Centroid-Contour Distances" : Displacement_Std,
    }

# Total Variation
def compute_total_variation(brightfield_dir):

    img = cv2.imread(brightfield_dir, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    TV = np.abs(np.diff(img, axis = 0)).mean() + np.abs(np.diff(img, axis = 1)).mean()
    return {"Total Variation": TV}

# Hu Moment 1-7
def compute_Hu_moment(brightfield_dir):

    img = cv2.imread(brightfield_dir, cv2.IMREAD_GRAYSCALE)
    img = img / 255

    # Moments
    moments = cv2.moments(img) 
    moments = cv2.HuMoments(moments)[:, 0]
    moments[moments != 0] = - np.sign(moments[moments != 0]) * np.log(np.abs(moments[moments != 0]))
    Hu_Moments = moments

    return {
        "Hu Moment 1": Hu_Moments[0], 
        "Hu Moment 2": Hu_Moments[1], 
        "Hu Moment 3": Hu_Moments[2], 
        "Hu Moment 4": Hu_Moments[3], 
        "Hu Moment 5": Hu_Moments[4], 
        "Hu Moment 6": Hu_Moments[5], 
        "Hu Moment 7": Hu_Moments[6],
    }

# SIFT 1-256, ORB 1-64
sift = BagOfKeypointsRepresentation(filename="./utils/feature_extractor_sift.pkl")
orb = BagOfKeypointsRepresentation(filename="./utils/feature_extractor_orb.pkl")
def compute_local_features(brightfield_dir):
    
    sift_res = sift.transform(brightfield_dir)
    orb_res = orb.transform(brightfield_dir)
    
    return dict([
        ("SIFT %d" % (i+1), v) for i, v in enumerate(sift_res)
    ] + [
        ("ORB %d" % (i+1), v) for i, v in enumerate(orb_res)
    ])

# Local Entropy, Cell Brightness, Cell Contrast
def compute_cell_region_grayscale_feature(brightfield_dir, mask_dir):

    img = cv2.imread(brightfield_dir, cv2.IMREAD_GRAYSCALE)
    entr_img = rank.entropy(img, morphology.disk(5))
    img = img / 255
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    
    brightness = img[mask > 0].mean() / np.median(img.ravel())
    local_entropy = entr_img[mask > 0].mean()
    contrast = img[mask > 0].std()
   
    return {
        "Cell Brightness" : brightness / np.median(img.ravel()), 
        "Local Entropy" : local_entropy, 
        "Contrast": contrast, 
    }

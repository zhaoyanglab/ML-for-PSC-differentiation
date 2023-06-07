import sklearn
from sklearn.cluster import KMeans
import cv2
import pickle
import numpy as np
import tqdm

def get_feature_descriptor(des_type):

    if des_type == "SIFT":
        return cv2.SIFT_create()
    elif des_type == "SURF":
        return cv2.xfeatures2d.SURF_create()
    elif des_type == "ORB":
        return cv2.ORB_create()
    else:
        raise Exception("Unknown descriptor type %s." % des_type)

class BagOfKeypointsRepresentation:
    
    '''
        Local feature extractor.
    '''
    
    def __init__(self, filename = None, candidates_for_each = 100, clusters = 256, random_state = 12345, des_type = "SIFT"):
        
        if (filename is not None):
            self.load(filename)

        else:
            self.des_type = des_type
            self.sift = get_feature_descriptor(des_type)
            self.candidates_for_each = candidates_for_each
            self.clusters = clusters
            self.kmeans = KMeans(n_clusters=self.clusters, random_state=random_state)
        
    def fit(self, img_list):
        
        '''
            Args: 
            ------
            img_list: a list of str
                The directories of the training images. 
        '''
        
        points = []
        
        print("Compute %s ..." % self.des_type)
        for filename in tqdm.tqdm(img_list):
            
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            
            keypoints = self.sift.detect(img, None)
            randomly_selected = np.random.choice(np.arange(len(keypoints)), size = (self.candidates_for_each))
            keypoints = [ keypoints[i] for i in randomly_selected ]
            
            _, des = self.sift.compute(img, keypoints)
            points.append(des)
            
        points = np.concatenate(points, axis = 0)
        
        print("Perform Kmeans ...")
        self.kmeans.fit(points)
        
        print("Done.")
    
    def transform(self, img_list):
        
        '''
            Args:
            --------
            img_list: a list of str
                The directories of the input images. 
            
            Returns:
            --------
            features: 2D numpy array, of shape (# images, clusters, )
                The feature vectors for all the input images.
            
            Note:
            --------
            You can also pass a single image to the function. 
            The returned value will be a 1D numpy array of shape (clusters, ).
        '''
        
        if (isinstance(img_list, str)):
            
            img = cv2.imread(img_list, cv2.IMREAD_GRAYSCALE)
            _, des = self.sift.detectAndCompute(img, None)
            
            cluster_idx = self.kmeans.predict(des)
            return np.histogram(cluster_idx, bins = range(self.clusters + 1), density = True)[0]
        
        else:
            res = []

            for filename in tqdm.tqdm(img_list):

                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                _, des = self.sift.detectAndCompute(img, None)

                cluster_idx = self.kmeans.predict(des)
                res.append( np.histogram(cluster_idx, bins = range(self.clusters + 1), density = True)[0] )

            return np.stack(res, axis = 0)
    
    def save(self, filename):
        
        sift = self.sift
        try:
            self.sift = None
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        except:
            print("[Error] Fail to save as %s" % filename)
        self.sift = sift
    
    def load(self, filename):
        
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
        
        self.des_type = tmp.des_type
        self.candidates_for_each = tmp.candidates_for_each
        self.clusters = tmp.clusters
        self.kmeans = tmp.kmeans
        self.sift = get_feature_descriptor(self.des_type)
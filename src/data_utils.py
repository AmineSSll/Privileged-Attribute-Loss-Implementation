import numpy as np
import skimage.io as sk
from skimage.filters import gaussian
import sys

# ======================================================= #
# ================ DATA WRITING FUNCTIONS =============== #
# ======================================================= #


# https://www.tensorflow.org/tutorials/load_data/images


def download_dataset_to_dir():
    pass

def save_model_to_dir():
    pass
    
# ======================================================= #
# ================ DATA LOADING FUNCTIONS =============== #
# ======================================================= #

def load_img():
    pass

def load_imgs():
    pass

def load_label():
    pass

def load_labels():
    pass

def load_data():
    pass
    # return (train_imgs, train_labels), (test_imgs, test_labels)

# ======================================================= #
# =============== DATA PROCESSING FUNCTIONS ============= #
# ======================================================= #

def img_resize(img, width, height):
    pass

def create_augmented_img():
    pass

def process_imgs(imgs):
    # img_resize(img, width, height)
    pass


def process_landmarks_file(filename, im_h, im_w, sigma):
    
    # Open file and read all lines
    f = open(filename, 'r')
    Lines = f.readlines()
    
    # List of heatmaps for each landmark
    heatmaps = list()
    
    for line in Lines:
        
        # Remove line skip
        line = line.strip()
        
        # x and y coordinates of landmarks
        x, y = line.split()
        x = int(x)
        y = int(y)
        
        # Landmark coordinates is set to 1, everything else is 0
        heatmap = np.zeros((im_h,im_w), dtype = np.float)
        heatmap[x,y] = 1
        
        # Apply Gaussian filter and add it to the heatmaps list
        heatmap = gaussian(heatmap, sigma)
        print(np.max(heatmap))
        heatmaps.append(heatmap)
        
        
    
    # Change list to ndarray and sum
    heatmaps = np.array(heatmaps)
    print(heatmaps.shape)
    facial_landmarks_prior = np.sum(heatmaps, axis = 0)
    print(facial_landmarks_prior.shape)
    
    return facial_landmarks_prior
        
    
    

def process_prior_heatmaps(imgs):
    pass



    
if __name__ == '__main__':
    
    
    sys.path.append('../')
    
    filename = '../data/RAFDB/raw/landmarks/test_0004_aligned.txt'
    
    im = process_landmarks_file(filename, 100, 100, 1)
    sk.imshow(im)
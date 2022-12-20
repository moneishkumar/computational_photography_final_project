import cv2
import scipy.io
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d
import numpy as np
import pdb 
import copy 
import argparse

def wiener_filter(img, kernel, K):
    '''
    Applies weiner deconvolution to the img 

    input args:
    img - image on which the non-blind deconvolution has to be performed
    kernel - blur kernel
    K- is the SNR

    output:
    dummy - deconvolved image 
    '''
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    #pdb.set_trace()
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy


def gamma_encoding(image):
    '''
    Gamma encodes the image

    inputs args:
    image - image that needs to be gammas encoded

    outputs args:
    image - gamma encoded image
    '''
    #Non-lenearization
    C_linear_below = image <= 0.0031308
    C_linear_above = image > 0.0031308

    image[C_linear_below] = 12.92*image[C_linear_below]
    image[C_linear_above] = (1+0.055)*((image[C_linear_above])**(1/2.4)) - 0.055

    return image

def gammaCorrection(src, gamma = 2.2):
    '''
    apples gamma correction to src

    input args: 
    src - src image
    gamma - gamma factor 

    output:
    src

    '''
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def calculate_gradients(images_deconvolved):
    '''
    Calcuates the L1 norm of the gradient for the deconvoled images

    input args: list of deconvoled images
    
    output: list of norms for all the deconvoled images
    '''
    norms = []
    for i in range(images_deconvolved.shape[0]):
        dy, dx = np.gradient(images_deconvolved[i])
        norm = np.sqrt(dx**2 + dy**2)
        norms.append(norm)
    
    norms = np.array(norms)
    
    return norms

def calculate_depth(norms):
    '''
    Caculates the depth of each pixel based on the min L1 norm
    
    input args: L1 norms for each deconvoled image

    output - depth map 
    
    '''
    norms = []
    depth = np.zeros_like((norms.shape[0], norms.shape[1]))

    k = 20
    for i in range(norms.shape[1]-k):
        for j in range(norms.shape[2]-k):
            depth[i:i+k, j:j+k] = np.argmin(np.sum(norms[:,i:i+k, j:j+k], axis=(1,2))) 

    return depth

def calculate_depth_mask(norms, mask, gray, images_deconvolved):
    '''
    Caculates the depth of each pixel based on the min L1 norm only for the masked regions
    
    input args: 
    norms - L1 norms for each deconvoled image
    mask - mask for the ROI

    output - depth map 
    '''
    norms = []
    bbox1, bbox2, bbox3 = mask
    depth = np.zeros_like((norms.shape[0], norms.shape[1]))

    for i in range(3):
        bbox = bbox1
        temp = np.sum(norms[i,bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])

        if(temp < min1):
            gray[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = images_deconvolved[i][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            min1 = temp
        bbox = bbox2
        temp = np.sum(norms[i,bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])

        if(temp < min2):
            min2 = temp
            gray[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = images_deconvolved[i][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
        bbox = bbox3
        temp = np.sum(norms[i,bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])

        if(temp < min3):
            min3 = temp
            gray[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = images_deconvolved[i][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
    return depth

def main(args):
    data_path = "./../data/"

    img = cv2.imread(data_path+'/beer_coke_inp.jpg', cv2.IMREAD_UNCHANGED)
    
    with open(args.segmentation_masks, 'rb') as f:
        masks = np.load(f, allow_pickle=True)

    # convert BGR to RGB
    img = img[:,:,::-1]

    # Use linear image
    img = gammaCorrection(img)
    
    '''
    This code is for gray scale images,
    To obtain the RGB images replace 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    with: 

    gray = img[:,:,c]

    for all cannels and concatenate the all the channels later.
    '''

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with open(args.kernels, 'rb') as f:
        kernel = np.load(f, allow_pickle=True)
    
    #pdb.set_trace()

    images_deconvolved = []

    # Iterate over all kernels
    for i in range(len(kernel)):
        img_de = wiener_filter(gray, kernel[i], K = 0.05)
        images_deconvolved.append(img_de)

    images_deconvolved = np.array(images_deconvolved)

    # Calculate gradients for the deconvolved images
    norms = calculate_gradients(images_deconvolved)

    # Calculate Depth by taking patches and measuting sharpness 

    depth = calculate_depth(norms)
    #depth = calculate_depth_with_mask(norms, masks,gray, images_deconvolved)

    plt.imshow(depth)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', help='path to coded aperture image')
    parser.add_argument('--kernels',help='path to calibrated kernels')
    parser.add_argument('--segmentation_mask',help='path to calibrated kernels')

    args = parser.parse_args()

    main(args)
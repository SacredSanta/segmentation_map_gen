#%%
'''
240903

'''


#%%  import package
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from lib.grayscale_conversion import *
from lib.preprocess_pkg import mask_ellipse
import struct 
import pandas as pd 
from joblib import Parallel, delayed, cpu_count

num_cores = cpu_count()

#%%  1. =============================================================

img = cv2.imread("./SW/MAETEL_SAMPLE.jpeg", cv2.IMREAD_COLOR)

img_np = np.array(img)
img1 = img[150:1050, 150:1050]

plt.imshow(img1)

#%% extract window images ------------------------------------------------------------

ix = 1 # shifting 간격
iy = 1

N = 20 # block 크기

blocks = np.array([np.newaxis, np.newaxis, np.newaxis])

def cut_box(row, col):
    global img1      
    global N  
    return img1[row:row+N, col:col+N]

# Read the entire file
results = Parallel(n_jobs=num_cores, verbose=10)(delayed(cut_box)(row, col) 
                                                 for row in range(0, img1.shape[0]-N, iy) 
                                                 for col in range(0, img1.shape[1]-N, ix))

results = np.array(results)

#%% 1.2 save file ------------------------------------------------------------
np.savez("./data/test3_dt", results=results)

#%% 1.3 load file ------------------------------------------------------------
a = np.load("./data/test2_dt.npz")
results = a["results"]






#%% 2. preprocessing =============================================================
from lib.grayscale_conversion import *

test = rgb_to_grayscale_gamma_corrected(results[0], gamma=2.2)

#%% 2-1 fft  ------------------------------------------------------------
test_fft = np.fft.fftshift(np.fft.fft2(test).reshape(20,20))


plt.imshow(abs(test_fft))


#%% 2-2 bpf  ------------------------------------------------------------
from lib.preprocess_pkg import mask_ellipse

sz = np.shape(test_fft)
f_msk = mask_ellipse([2, 2]) - mask_ellipse([0.5, 0.5])
sub_img2 = np.real(np.fft.ifft2(np.fft.ifftshift(test_fft * f_msk)))
sub_img3 = np.divide(np.max(np.max(sub_img2)) - sub_img2, np.max(np.max(sub_img2)))
THOLD = np.mean(sub_img3) - np.std(sub_img3)
sub_img4 = (sub_img3 < THOLD) * 1
sub_img4 = sub_img4.reshape(1, 20, 20, 1)


plt.imshow(sub_img4[0])


#%% 3-1 best image ============================================================
best_img = np.zeros([20,20])

r = 5
for rr in range(r+1):
    for theta in range(0, 360):
        best_img[int(10+rr*np.sin(theta)), int(10+rr*np.cos(theta))] = 1
        
pixelcounts = np.sum(best_img == 1)
#%% 3-2 compare with best_img ~ windowed img ---------------------------------------------
'''
from lib.img_similarity import *

simil, _ = calculate_ssim(best_img, sub_img4[0,:,:,0])

print(simil)
'''
plt.figure()

plt.subplot(1,3,1)
plt.imshow(best_img)
plt.title("Best img")

plt.subplot(1,3,2)
plt.imshow(sub_img4[0])
plt.title("windowed img")

tf_img = (best_img.astype(bool) & sub_img4[0,:,:,0].astype(bool))

plt.subplot(1,3,3)
plt.imshow(tf_img)
plt.title("overlapped part")

score = pixelcounts - np.sum(tf_img == 1)
print("score(높을 수록 겹치는게 없음) : ", score)

#%% 3-3 save img with T/F ----------------------------------------
score_thres = 10

num = 1
if score <= score_thres:
    cv2.imwrite(f'./data/True/true_{num}.png', sub_img4[0,:,:,0])
else:
    cv2.imwrite(f'./data/False/false_{num}.png', sub_img4[0,:,:,0])
    
    
    
    



#%% Final ==============================================================

# init
def make_bestimg(r=5):
    best_img = np.zeros([20,20])
    for rr in range(r+1):
        for theta in range(0, 360):
            best_img[int(10+rr*np.sin(theta)), int(10+rr*np.cos(theta))] = 1
    
    return best_img, np.sum(best_img == 1)


def makedata(idx):
    global best_img, pixelcounts, data_len, results
    
    num = np.random.choice(range(0, data_len), 1, replace=False)[0]
    
    test = rgb_to_grayscale_gamma_corrected(results[num], gamma=2.2)
    test_fft = np.fft.fftshift(np.fft.fft2(test).reshape(20,20))
    sz = np.shape(test_fft)
    f_msk = mask_ellipse([2, 2]) - mask_ellipse([0.5, 0.5])
    sub_img2 = np.real(np.fft.ifft2(np.fft.ifftshift(test_fft * f_msk)))
    sub_img3 = np.divide(np.max(np.max(sub_img2)) - sub_img2, np.max(np.max(sub_img2)))
    THOLD = np.mean(sub_img3) - np.std(sub_img3)
    sub_img4 = (sub_img3 < THOLD) * 1
    sub_img4 = sub_img4.reshape(1, 20, 20, 1)
    
    tf_img = (best_img.astype(bool) & sub_img4[0,:,:,0].astype(bool))
    score = pixelcounts - np.sum(tf_img == 1)
    
    if score <= score_thres:
        cv2.imwrite(f'./data/True/true_{num}.png', (sub_img4[0,:,:,0]*255).astype(np.uint8))
    else:
        cv2.imwrite(f'./data/False/false_{num}.png', (sub_img4[0,:,:,0]*255).astype(np.uint8))

    cv2.imwrite(f'./data/ori/ori_{num}.png', results[num])


best_img, pixelcounts = make_bestimg(r=3)
score_thres = 10


#%% data make

iter = 100
data_len = len(results)
dt_results = Parallel(n_jobs=num_cores, verbose=10)(delayed(makedata)(idx) 
                                                       for idx in range(iter))
                                                 

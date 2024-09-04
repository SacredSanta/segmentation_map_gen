#%%
import numpy as np
import cv2

def gamma_decompress(value, gamma=2.2):
    return value ** gamma

def gamma_compress(value, gamma=2.2):
    return value ** (1/gamma)

def rgb_to_grayscale_gamma_corrected(img, gamma=2.2):
    #b, g, r = cv2.split(img)  
    # 감마 확장을 적용하여 감마 압축된 RGB 값을 선형 공간으로 변환
    r_ = gamma_decompress(img[:,:,0]/255.0, gamma)
    g_ = gamma_decompress(img[:,:,1]/255.0, gamma)
    b_ = gamma_decompress(img[:,:,2]/255.0, gamma)
    
    
    # 선형 RGB 값을 사용하여 선형 밝기 값을 계산
    y_ = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
    
    # 필요한 경우 선형 밝기 값을 다시 감마 압축하여 그레이스케일 값으로 변환
    y_n = gamma_compress(y_, gamma)
        
    return y_n*255



if __name__ == '__main__':
    img = np.load('../data/test_dt.npz')
    img = img['results']
    
    a = rgb_to_grayscale_gamma_corrected(img[0])
    
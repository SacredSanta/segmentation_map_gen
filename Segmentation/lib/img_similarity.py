import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_histogram_similarity(img1, img2):
    # 히스토그램 계산
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # 히스토그램 비교
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return hist_similarity

def calculate_ssim(img1, img2):
    # SSIM 계산
    similarity_index, _ = ssim(img1, img2, full=True)
    return similarity_index

def combined_similarity(img1, img2, weight_hist=0.5, weight_ssim=0.5):
    # 히스토그램 유사도 계산
    hist_similarity = calculate_histogram_similarity(img1, img2)

    # SSIM 유사도 계산
    ssim_similarity = calculate_ssim(img1, img2)

    # 가중 평균하여 종합적인 유사도 계산
    combined_similarity = (weight_hist * hist_similarity) + (weight_ssim * ssim_similarity)
    return combined_similarity


if __name__ == '__main__':
    # 이미지 불러오기
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')

    # 종합적인 유사도 계산
    similarity = combined_similarity(img1, img2)

    print(f"Combined Similarity: {similarity}")
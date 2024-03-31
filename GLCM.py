import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_normalized_gray_histogram(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算灰度直方图
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # 删除灰度值为0的统计数据
    histogram = histogram[1:]
    
    # 归一化直方图
    histogram /= histogram.sum()
    
    return histogram

def plot_histogram(histogram):
    plt.figure()
    plt.title("Normalized Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Normalized Frequency")
    plt.plot(histogram)
    plt.xlim([0, 255])
    plt.show()

def save_histogram_plot(histogram, filename):
    plt.figure()
    plt.title("Tumor Grayscale Histogram")
    plt.xlabel("Grayscale Value")
    plt.ylabel("Normalized Frequency")
    
    x = np.arange(len(histogram))
    
    plt.plot(histogram)
    
    # 设置区间[100, 175]内的曲线为红色
    mask = (x >= 120) & (x <= 175)
    plt.plot(x[mask], histogram[mask], color='r')
    
    plt.xlim([0, 255])
    
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # 读取图像
    file_path = '/root/common-dir/selfdataset/glioma_dataset/class3/Threeclass_allpng'

    for img_name in os.listdir(file_path):
        img_path = os.path.join(file_path,img_name)
        image = cv2.imread(img_path)
        # 计算并归一化灰度直方图
        normalized_histogram = compute_normalized_gray_histogram(image)
        # 保存直方图到指定地址
        save_path = f'selfdataset/glioma_dataset/class3/normalized_histogram/{img_name}_histogram.png'
        save_histogram_plot(normalized_histogram, save_path)


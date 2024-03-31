# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy
from torchvision import transforms

# 椒盐噪声
def SaltAndPepper(src,percetage):
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randR=np.random.randint(0,src.shape[0]-1)
        randG=np.random.randint(0,src.shape[1]-1)
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0:
            SP_NoiseImg[randR,randG,randB]=0
        else:
            SP_NoiseImg[randR,randG,randB]=255
    return SP_NoiseImg

# 高斯噪声
def addGaussianNoise(image,percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

# 昏暗
def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy

# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated

# 翻转
def mirror(image, direction):
    # direction参数可选值为0（沿x轴翻转），1（沿y轴翻转），-1（同时沿x和y轴翻转）
    mirrored = cv2.flip(image, direction)
    return mirrored
    
# 图片文件夹路径
# file_dir = 'selfdataset/glioma_dataset/class3/Threeclass_allpng_delete_apart_crop'
# sava_dir = 'selfdataset/glioma_dataset/class3/Threeclass_allpng_delete_apart_crop_aug10'
file_dir = 'selfdataset/mri_t1_classification_dataset2/train'
save_dir = 'selfdataset/mri_t1_classification_dataset2/train_aug'
os.makedirs(save_dir, exist_ok=True)
i = 0
for dir in os.listdir(file_dir):
    path = os.path.join(file_dir,dir)
    for img_name in os.listdir(path):
        img_path = os.path.join(path,img_name)
        # img_path = file_dir + '/' + img_name
        img = cv2.imread(img_path)
        sava_dir = save_dir + '/' + dir
        os.makedirs(sava_dir, exist_ok=True)
        cv2.imwrite(sava_dir + '/' + img_name, img)
        # img = cv2.resize(img, (432, 432))
        rotated_90 = rotate(img, 90)
        # cv2.imwrite(sava_dir + '/' + img_name[0:-4] + '_r90.png', rotated_90)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_r90.png', rotated_90)
        
        # rotated__90 = rotate(img, -90)
        # cv2.imwrite(sava_dir + '/' + img_name[0:-4] + '_r_90.png', rotated__90)
        rotated45 = rotate(img, 45)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_r45.png', rotated45)
        # rotated__45 = rotate(img, -45)
        # cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_r_45.png', rotated__45)
        rotated_135 = rotate(img, 135)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_r135.png', rotated_135)


        # 镜像
        flipped_img = mirror(img,0)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_fli_0.png', flipped_img)
        flipped_img = mirror(img,1)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_fli.png', flipped_img)
        flipped_img = mirror(img,-1)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_fli_2.png', flipped_img)


        # # 增加噪声
        img_salt = SaltAndPepper(img, 0.3)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_salt.png', img_salt)
        img_gauss = addGaussianNoise(img, 0.3)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_noise.png',img_gauss)

        #变亮、变暗
        img_darker = darker(img)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_darker.png', img_darker)
        img_brighter = brighter(img)
        cv2.imwrite(sava_dir + '/' + img_name.split(".")[0] + '_brighter.png', img_brighter)

        # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
        # #      cv2.GaussianBlur(图像，卷积核，标准差）
        # cv2.imwrite(sava_dir + '/' + img_name[0:-4] + '_blur.png',blur)
        i = i +1
        print(i)
import os
import numpy as np
# import matplotlib.pyplot as plt
import h5py
import random
from skimage import io
import csv
import nibabel as nib
from PIL import Image
import numpy as np
import cv2

# 输入的glioma原始数据地址
data_dir = 'selfdataset/glioma_nii'
# 输出切片存放地址
outdata_dir_t1 = '/root/common-dir/selfdataset/glioma_dataset/class3/Threeclass_allpng'
os.makedirs(outdata_dir_t1,exist_ok=True)
# 存放数据信息文件
csv_dir = 'selfdataset/Threeclass_label.csv'

if __name__ == "__main__":
    

    with open(csv_dir, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过标题行
        data = [(row[0],row[1]) for row in csv_reader]

    for folder_name,label in data:
        file_path_t1 = os.path.join(data_dir,folder_name,f'{folder_name}_t1w.nii.gz')
        nii_data = nib.load(file_path_t1)
        nii_data = nii_data.get_fdata()

        # 抽取冠状位方位切片
        for slice_index in range(nii_data.shape[2]):
            slice_HW = nii_data[:, :, slice_index]
            
            # 将数据转换为 [0, 255] 范围的整数类型
            image = ((slice_HW - np.min(slice_HW)) / (np.max(slice_HW) - np.min(slice_HW)) * 255).astype(np.uint8)

            # 将灰度图像转换为RGB格式
            image_rgb = np.stack((image,) * 3, axis=-1)

            # 将NumPy数组转换为PIL Image对象
            pil_image = Image.fromarray(image_rgb)

            # 保存图像
            image_filename = f"{folder_name}_HW_{slice_index}_{label}.png"
            image_path = os.path.join(outdata_dir_t1, image_filename)
            pil_image.save(image_path)

         
        # 抽取矢状位方位切片
        for slice_index in range(nii_data.shape[0]):
            slice_WD = nii_data[slice_index,:, :]
            
            # 将数据转换为 [0, 255] 范围的整数类型
            image = ((slice_WD - np.min(slice_WD)) / (np.max(slice_WD) - np.min(slice_WD)) * 255).astype(np.uint8)

            # 将灰度图像转换为RGB格式
            image_rgb = np.stack((image,) * 3, axis=-1)

            # 将NumPy数组转换为PIL Image对象
            pil_image = Image.fromarray(image_rgb)

            # 保存图像
            image_filename = f"{folder_name}_WD_{slice_index}_{label}.png"
            image_path = os.path.join(outdata_dir_t1, image_filename)
            pil_image.save(image_path)

        # 抽取轴状位方位切片
        for slice_index in range(nii_data.shape[1]):
            slice_HD = nii_data[:, slice_index, :]
            
            # 将数据转换为 [0, 255] 范围的整数类型
            image = ((slice_HD - np.min(slice_HD)) / (np.max(slice_HD) - np.min(slice_HD)) * 255).astype(np.uint8)

            # 将灰度图像转换为RGB格式
            image_rgb = np.stack((image,) * 3, axis=-1)

            # 将NumPy数组转换为PIL Image对象
            pil_image = Image.fromarray(image_rgb)

            # 保存图像
            image_filename = f"{folder_name}_HD_{slice_index}_{label}.png"
            image_path = os.path.join(outdata_dir_t1, image_filename)
            pil_image.save(image_path)
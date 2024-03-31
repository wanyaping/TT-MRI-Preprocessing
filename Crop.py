from PIL import Image
import os
import cv2

def get_unique_image_sizes(folder_path):
    unique_sizes = set()
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    unique_sizes.add((width, height))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return unique_sizes

def count_files_with_hw(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        if "HD" in filename:
            count += 1
    return count


# 输入和输出文件夹路径
input_folder = "/root/common-dir/selfdataset/glioma_dataset/class3/Threeclass_allpng"
output_folder = "/root/common-dir/selfdataset/glioma_dataset/class3/Threeclass_allpng_crop"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 PNG 图像
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        _, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 计算最大轮廓的边界框
        x, y, w, h = cv2.boundingRect(max_contour)

        # 裁剪图像
        cropped_image = image[y:y+h, x:x+w]

        # 保存裁剪后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped_image)

        print(f"Saved {output_path}")

print("裁剪完成")

# 将路径下的所有视频文件重命名为保留前7个字符
import os

def rename_videos_in_directory(directory_path):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory_path):
        # 构建新的文件名，保留原文件名的前7个字符，并加上原文件的扩展名
        new_filename = filename[:7] + filename[-4:]
        # 构建完整的文件路径
        old_file_path = os.path.join(directory_path, filename)
        new_file_path = os.path.join(directory_path, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)

# 调用函数，传入视频文件所在的目录路径
rename_videos_in_directory('dataset/video')
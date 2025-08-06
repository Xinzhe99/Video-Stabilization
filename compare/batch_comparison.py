"""
批量比较工具 - 用于处理多组视频比较
"""
import os
import glob
from video_stabilization_comparison import VideoStabilizationComparison

def batch_compare_videos(video_directory, output_directory="output"):
    """
    批量比较视频稳定算法
    
    参数:
        video_directory: 包含视频文件的目录
        output_directory: 输出目录
    """
    
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    
    # 查找视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv','*.m4v']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_directory, ext)))
        video_files.extend(glob.glob(os.path.join(video_directory, ext.upper())))
    
    if not video_files:
        print(f"在目录 {video_directory} 中没有找到视频文件")
        return
        
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    # 创建比较器
    comparator = VideoStabilizationComparison()
    
    # 添加所有视频
    for video_path in video_files:
        filename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(filename)[0]
        comparator.add_video(video_path, name_without_ext)
    
    # 生成输出文件名
    output_filename = f"batch_comparison_{len(video_files)}_videos.mp4"
    output_path = os.path.join(output_directory, output_filename)
    
    # 执行比较
    comparator.compare_videos(output_path, preview=True, save_video=True)

if __name__ == "__main__":
    # 使用示例
    video_dir = input("请输入视频文件夹路径: ").strip()
    if os.path.exists(video_dir):
        batch_compare_videos(video_dir)
    else:
        print("指定的目录不存在！")
import cv2
import numpy as np
import os
import time
from pathlib import Path

class VideoStabilizationComparison:
    def __init__(self):
        self.videos = []
        self.caps = []
        self.labels = []
        self.total_frames = 0
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0
        
    def add_video(self, video_path, label=None):
        """添加要比较的视频"""
        if not os.path.exists(video_path):
            print(f"错误：视频文件不存在 - {video_path}")
            return False
            
        # 创建VideoCapture对象
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 - {video_path}")
            return False
            
        # 获取视频信息
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"添加视频: {video_path}")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {frame_count}")
        
        # 如果是第一个视频，设置基准参数
        if len(self.videos) == 0:
            self.total_frames = frame_count
            self.fps = fps
            self.frame_width = width
            self.frame_height = height
        else:
            # 检查视频参数是否一致
            if frame_count != self.total_frames:
                print(f"警告：视频帧数不一致 ({frame_count} vs {self.total_frames})")
            if fps != self.fps:
                print(f"警告：视频帧率不一致 ({fps} vs {self.fps})")
                
        # 生成标签
        if label is None:
            label = Path(video_path).stem
            
        self.videos.append(video_path)
        self.caps.append(cap)
        self.labels.append(label)
        
        return True
        
    def resize_frame(self, frame, target_width, target_height):
        """调整帧大小，保持宽高比"""
        h, w = frame.shape[:2]
        
        # 计算缩放比例
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整大小
        resized = cv2.resize(frame, (new_w, new_h))
        
        # 创建目标大小的画布并居中放置
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
        
    def add_label_to_frame(self, frame, label):
        """在帧的左上角添加标签"""
        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text_color = (255, 255, 255)  # 白色
        bg_color = (0, 0, 0)  # 黑色背景
        
        # 获取文本大小
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # 绘制背景矩形
        cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), bg_color, -1)
        
        # 绘制文本
        cv2.putText(frame, label, (15, 15 + text_h), font, font_scale, text_color, font_thickness)
        
        return frame
        
    def create_comparison_grid(self, frames):
        """创建比较网格"""
        num_videos = len(frames)
        if num_videos == 0:
            return None
            
        # 计算网格布局
        if num_videos == 1:
            cols, rows = 1, 1
        elif num_videos == 2:
            cols, rows = 2, 1
        elif num_videos <= 4:
            cols, rows = 2, 2
        elif num_videos <= 6:
            cols, rows = 3, 2
        elif num_videos <= 9:
            cols, rows = 3, 3
        else:
            cols, rows = 4, 3  # 最多支持12个视频
            
        # 计算每个子窗口的大小
        sub_width = self.frame_width // cols
        sub_height = self.frame_height // rows
        
        # 创建输出画布
        output_height = sub_height * rows
        output_width = sub_width * cols
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 排列帧
        for i, frame in enumerate(frames):
            if i >= cols * rows:
                break
                
            row = i // cols
            col = i % cols
            
            # 调整帧大小
            resized_frame = self.resize_frame(frame, sub_width, sub_height)
            
            # 添加标签
            labeled_frame = self.add_label_to_frame(resized_frame, self.labels[i])
            
            # 放置到画布上
            y1 = row * sub_height
            y2 = y1 + sub_height
            x1 = col * sub_width
            x2 = x1 + sub_width
            
            canvas[y1:y2, x1:x2] = labeled_frame
            
        return canvas
        
    def compare_videos(self, output_path="comparison_output.mp4", preview=True, save_video=True):
        """执行视频比较"""
        if len(self.videos) == 0:
            print("错误：没有添加任何视频文件")
            return False
            
        print(f"\n开始比较 {len(self.videos)} 个视频...")
        print("视频列表:")
        for i, (video_path, label) in enumerate(zip(self.videos, self.labels)):
            print(f"  {i+1}. {label}: {video_path}")
            
        # 计算输出视频的尺寸
        num_videos = len(self.videos)
        if num_videos <= 2:
            cols, rows = min(num_videos, 2), 1
        elif num_videos <= 4:
            cols, rows = 2, 2
        elif num_videos <= 6:
            cols, rows = 3, 2
        elif num_videos <= 9:
            cols, rows = 3, 3
        else:
            cols, rows = 4, 3
            
        output_width = (self.frame_width // cols) * cols
        output_height = (self.frame_height // rows) * rows
        
        # 设置视频编写器
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (output_width, output_height))
            if not writer.isOpened():
                print(f"错误：无法创建输出视频文件 - {output_path}")
                return False
            print(f"输出视频: {output_path}")
            print(f"输出分辨率: {output_width}x{output_height}")
            
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frames = []
                valid_frames = 0
                
                # 读取所有视频的当前帧
                for cap in self.caps:
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                        valid_frames += 1
                    else:
                        # 如果某个视频结束了，使用黑色帧
                        black_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                        cv2.putText(black_frame, "视频结束", 
                                  (self.frame_width//2-60, self.frame_height//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        frames.append(black_frame)
                
                # 如果所有视频都结束了，退出循环
                if valid_frames == 0:
                    break
                    
                # 创建比较网格
                comparison_frame = self.create_comparison_grid(frames)
                if comparison_frame is None:
                    break
                    
                # 保存到输出视频
                if save_video and writer is not None:
                    writer.write(comparison_frame)
                    
                # 实时预览
                if preview:
                    # 如果输出分辨率太大，缩放显示
                    display_frame = comparison_frame
                    if comparison_frame.shape[1] > 1920:  # 如果宽度超过1920
                        scale = 1920 / comparison_frame.shape[1]
                        new_width = 1920
                        new_height = int(comparison_frame.shape[0] * scale)
                        display_frame = cv2.resize(comparison_frame, (new_width, new_height))
                        
                    cv2.imshow("视频稳定算法比较", display_frame)
                    
                    # 按键控制
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # 按'q'退出
                        print("\n用户手动停止比较")
                        break
                    elif key == ord(' '):  # 按空格暂停/继续
                        cv2.waitKey(0)
                        
                frame_count += 1
                
                # 显示进度
                if frame_count % 30 == 0:  # 每30帧显示一次进度
                    elapsed = time.time() - start_time
                    progress = (frame_count / self.total_frames) * 100
                    fps_current = frame_count / elapsed if elapsed > 0 else 0
                    print(f"\r处理进度: {progress:.1f}% ({frame_count}/{self.total_frames}) - "
                          f"处理速度: {fps_current:.1f} FPS", end="", flush=True)
                          
        except KeyboardInterrupt:
            print("\n\n用户中断处理...")
            
        finally:
            # 清理资源
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
        elapsed_total = time.time() - start_time
        print(f"\n\n处理完成！")
        print(f"总处理时间: {elapsed_total:.2f} 秒")
        print(f"平均处理速度: {frame_count/elapsed_total:.2f} FPS")
        if save_video:
            print(f"输出文件已保存: {output_path}")
            
        return True
        
    def __del__(self):
        """析构函数，释放所有VideoCapture对象"""
        for cap in self.caps:
            if cap.isOpened():
                cap.release()

def main():
    """主函数示例"""
    # 创建比较器
    comparator = VideoStabilizationComparison()
    
    # 示例：添加视频文件（请替换为您的实际文件路径）
    video_files = [
        ("original_video.mp4", "原始视频"),
        ("stabilized_kalman.mp4", "卡尔曼滤波"),
        ("stabilized_opencv.mp4", "OpenCV稳定"),
        ("stabilized_custom.mp4", "自定义算法"),
    ]
    
    print("=== 视频稳定算法比较工具 ===\n")
    
    # 添加视频文件
    added_count = 0
    for video_path, label in video_files:
        if os.path.exists(video_path):
            if comparator.add_video(video_path, label):
                added_count += 1
        else:
            print(f"跳过不存在的文件: {video_path}")
            
    if added_count == 0:
        print("错误：没有找到任何有效的视频文件！")
        print("请确保视频文件存在并且路径正确。")
        return
        
    print(f"\n成功添加 {added_count} 个视频文件")
    
    # 开始比较
    print("\n控制说明:")
    print("- 按 'q' 键退出")
    print("- 按 '空格' 键暂停/继续")
    print("- 关闭预览窗口也会停止处理\n")
    
    comparator.compare_videos(
        output_path="stabilization_comparison.mp4",
        preview=True,      # 是否实时预览
        save_video=True    # 是否保存输出视频
    )

if __name__ == "__main__":
    main()
import cv2
import numpy as np
from video_stabilizer import VideoStabilizer

def main():
    # 创建视频稳定器对象
    stabilizer = VideoStabilizer()
    
    # 初始化摄像头（0表示默认摄像头）
    # cap = cv2.VideoCapture(0)
    
    # 也可以使用视频文件：
    cap = cv2.VideoCapture(r'D:\pycharmproject\meshflow-master\videos\video-2\video-2.m4v')
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 读取第一帧
    ret, frame_1 = cap.read()
    if not ret:
        print("无法读取第一帧")
        return
    
    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('stabilized_output_kalman.avi', fourcc, 30.0, 
                         (frame_1.shape[1], frame_1.shape[0]))
    
    print("开始视频稳定化处理... 按 'q' 退出")
    
    while True:
        try:
            ret, frame_2 = cap.read()
            
            if not ret:
                break
            
            # 进行视频稳定化
            stabilized_frame = stabilizer.stabilize(frame_1, frame_2)
            
            # 写入输出视频
            out.write(stabilized_frame)
            
            # # 显示稳定后的视频
            # cv2.imshow("Stabled", stabilized_frame)
            
            # 按'q'退出
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            # 更新帧
            frame_1 = frame_2.copy()
            
        except Exception as e:
            print(f"处理出错: {e}")
            # 出错时重新读取帧
            ret, frame_1 = cap.read()
            if not ret:
                break
    
    # 清理资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频稳定化完成！")

if __name__ == "__main__":
    main()
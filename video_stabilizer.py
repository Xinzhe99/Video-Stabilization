import cv2
import numpy as np
import math

class VideoStabilizer:
    def __init__(self):
        # 卡尔曼滤波器参数
        self.Q1 = 0.004
        self.R1 = 0.5
        
        # 显示对比结果的开关
        self.test = True
        
        # 边界裁剪参数
        self.HORIZONTAL_BORDER_CROP = 30
        
        # 初始化变量
        self.k = 1
        
        # 误差初始化
        self.err_scale_x = 1.0
        self.err_scale_y = 1.0
        self.err_theta = 1.0
        self.err_trans_x = 1.0
        self.err_trans_y = 1.0
        
        # 卡尔曼滤波器的Q和R参数
        self.Q_scale_x = self.Q1
        self.Q_scale_y = self.Q1
        self.Q_theta = self.Q1
        self.Q_trans_x = self.Q1
        self.Q_trans_y = self.Q1
        
        self.R_scale_x = self.R1
        self.R_scale_y = self.R1
        self.R_theta = self.R1
        self.R_trans_x = self.R1
        self.R_trans_y = self.R1
        
        # 累积变换参数
        self.sum_scale_x = 0.0
        self.sum_scale_y = 0.0
        self.sum_theta = 0.0
        self.sum_trans_x = 0.0
        self.sum_trans_y = 0.0
        
        # 卡尔曼滤波器状态
        self.scale_x = 0.0
        self.scale_y = 0.0
        self.theta = 0.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        
        # 变换矩阵
        self.smoothed_mat = np.zeros((2, 3), dtype=np.float64)
        
    def stabilize(self, frame_1, frame_2):
        """主要的稳定化函数"""
        # 转换为灰度图
        frame1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
        
        # 计算垂直边界
        vert_border = int(self.HORIZONTAL_BORDER_CROP * frame_1.shape[0] / frame_1.shape[1])
        
        # 特征点检测
        features1 = cv2.goodFeaturesToTrack(frame1_gray, 
                                           maxCorners=200, 
                                           qualityLevel=0.01, 
                                           minDistance=30)
        
        if features1 is None or len(features1) < 10:
            return frame_1
            
        # 光流追踪
        features2, status, err = cv2.calcOpticalFlowPyrLK(frame1_gray, 
                                                          frame2_gray, 
                                                          features1, 
                                                          None)
        
        # 筛选有效的特征点
        good_features1 = []
        good_features2 = []
        
        for i in range(len(status)):
            if status[i] == 1:
                good_features1.append(features1[i])
                good_features2.append(features2[i])
        
        if len(good_features1) < 10:
            return frame_1
            
        good_features1 = np.array(good_features1, dtype=np.float32)
        good_features2 = np.array(good_features2, dtype=np.float32)
        
        # 估计仿射变换
        try:
            # 注意：OpenCV 4.x中estimateRigidTransform已被弃用，使用estimateAffinePartial2D
            affine_matrix, _ = cv2.estimateAffinePartial2D(good_features1, good_features2)
            
            if affine_matrix is None:
                return frame_1
                
        except:
            return frame_1
        
        # 提取变换参数
        dx = affine_matrix[0, 2]
        dy = affine_matrix[1, 2]
        da = math.atan2(affine_matrix[1, 0], affine_matrix[0, 0])
        
        if abs(math.cos(da)) > 1e-6:
            ds_x = affine_matrix[0, 0] / math.cos(da)
            ds_y = affine_matrix[1, 1] / math.cos(da)
        else:
            ds_x = 1.0
            ds_y = 1.0
        
        sx = ds_x
        sy = ds_y
        
        # 累积变换参数
        self.sum_trans_x += dx
        self.sum_trans_y += dy
        self.sum_theta += da
        self.sum_scale_x += ds_x
        self.sum_scale_y += ds_y
        
        # 卡尔曼滤波（第一帧跳过）
        if self.k == 1:
            self.k += 1
        else:
            self.kalman_filter()
        
        # 计算差值
        diff_scale_x = self.scale_x - self.sum_scale_x
        diff_scale_y = self.scale_y - self.sum_scale_y
        diff_trans_x = self.trans_x - self.sum_trans_x
        diff_trans_y = self.trans_y - self.sum_trans_y
        diff_theta = self.theta - self.sum_theta
        
        # 应用平滑参数
        ds_x = ds_x + diff_scale_x
        ds_y = ds_y + diff_scale_y
        dx = dx + diff_trans_x
        dy = dy + diff_trans_y
        da = da + diff_theta
        
        # 创建平滑变换矩阵
        self.smoothed_mat[0, 0] = sx * math.cos(da)
        self.smoothed_mat[0, 1] = sx * (-math.sin(da))
        self.smoothed_mat[1, 0] = sy * math.sin(da)
        self.smoothed_mat[1, 1] = sy * math.cos(da)
        self.smoothed_mat[0, 2] = dx
        self.smoothed_mat[1, 2] = dy
        
        # 应用变换
        smoothed_frame = cv2.warpAffine(frame_1, self.smoothed_mat, 
                                       (frame_2.shape[1], frame_2.shape[0]))
        
        # 裁剪边界以消除黑边
        smoothed_frame = smoothed_frame[vert_border:smoothed_frame.shape[0]-vert_border,
                                      self.HORIZONTAL_BORDER_CROP:smoothed_frame.shape[1]-self.HORIZONTAL_BORDER_CROP]
        
        # 调整大小回原始尺寸
        smoothed_frame = cv2.resize(smoothed_frame, (frame_2.shape[1], frame_2.shape[0]))
        
        # 显示对比结果
        if self.test:
            canvas = np.zeros((frame_2.shape[0], frame_2.shape[1]*2+10, frame_2.shape[2]), dtype=np.uint8)
            
            canvas[:, :smoothed_frame.shape[1]] = frame_1
            canvas[:, smoothed_frame.shape[1]+10:smoothed_frame.shape[1]*2+10] = smoothed_frame
            
            if canvas.shape[1] > 1920:
                canvas = cv2.resize(canvas, (canvas.shape[1]//2, canvas.shape[0]//2))
            
            cv2.imshow("Compared", canvas)
        
        return smoothed_frame
    
    def kalman_filter(self):
        """卡尔曼滤波器实现"""
        frame_1_scale_x = self.scale_x
        frame_1_scale_y = self.scale_y
        frame_1_theta = self.theta
        frame_1_trans_x = self.trans_x
        frame_1_trans_y = self.trans_y
        
        frame_1_err_scale_x = self.err_scale_x + self.Q_scale_x
        frame_1_err_scale_y = self.err_scale_y + self.Q_scale_y
        frame_1_err_theta = self.err_theta + self.Q_theta
        frame_1_err_trans_x = self.err_trans_x + self.Q_trans_x
        frame_1_err_trans_y = self.err_trans_y + self.Q_trans_y
        
        gain_scale_x = frame_1_err_scale_x / (frame_1_err_scale_x + self.R_scale_x)
        gain_scale_y = frame_1_err_scale_y / (frame_1_err_scale_y + self.R_scale_y)
        gain_theta = frame_1_err_theta / (frame_1_err_theta + self.R_theta)
        gain_trans_x = frame_1_err_trans_x / (frame_1_err_trans_x + self.R_trans_x)
        gain_trans_y = frame_1_err_trans_y / (frame_1_err_trans_y + self.R_trans_y)
        
        self.scale_x = frame_1_scale_x + gain_scale_x * (self.sum_scale_x - frame_1_scale_x)
        self.scale_y = frame_1_scale_y + gain_scale_y * (self.sum_scale_y - frame_1_scale_y)
        self.theta = frame_1_theta + gain_theta * (self.sum_theta - frame_1_theta)
        self.trans_x = frame_1_trans_x + gain_trans_x * (self.sum_trans_x - frame_1_trans_x)
        self.trans_y = frame_1_trans_y + gain_trans_y * (self.sum_trans_y - frame_1_trans_y)
        
        self.err_scale_x = (1 - gain_scale_x) * frame_1_err_scale_x
        self.err_scale_y = (1 - gain_scale_y) * frame_1_err_scale_x
        self.err_theta = (1 - gain_theta) * frame_1_err_theta
        self.err_trans_x = (1 - gain_trans_x) * frame_1_err_trans_x
        self.err_trans_y = (1 - gain_trans_y) * frame_1_err_trans_y
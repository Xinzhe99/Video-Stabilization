import cv2
import numpy as np
import math

class VideoStabilizer:
    def __init__(self, feature_type='gftt'):
        # 卡尔曼滤波器参数
        self.Q1 = 0.004  # 卡尔曼滤波器过程噪声Q的初始值，影响滤波器对新数据的敏感度
        self.R1 = 0.5    # 卡尔曼滤波器观测噪声R的初始值，影响滤波器对观测数据的信任度
        
        # 显示对比结果的开关
        self.test = True  # 是否显示原始帧与稳定后帧的对比画面，True为显示
        
        # 边界裁剪参数
        self.HORIZONTAL_BORDER_CROP = 45  # 水平方向裁剪的像素数，用于去除变换后产生的黑边
        
        # 初始化帧计数器
        self.k = 1  # 当前处理的帧编号，第一帧不做滤波
        
        # 误差初始化
        self.err_scale_x = 1.0  # x方向缩放的估计误差协方差
        self.err_scale_y = 1.0  # y方向缩放的估计误差协方差
        self.err_theta = 1.0    # 旋转角度的估计误差协方差
        self.err_trans_x = 1.0  # x方向平移的估计误差协方差
        self.err_trans_y = 1.0  # y方向平移的估计误差协方差
        
        # 卡尔曼滤波器的Q（过程噪声）和R（观测噪声）参数
        self.Q_scale_x = self.Q1  # x方向缩放的过程噪声Q
        self.Q_scale_y = self.Q1  # y方向缩放的过程噪声Q
        self.Q_theta = self.Q1    # 旋转角度的过程噪声Q
        self.Q_trans_x = self.Q1  # x方向平移的过程噪声Q
        self.Q_trans_y = self.Q1  # y方向平移的过程噪声Q
        
        self.R_scale_x = self.R1  # x方向缩放的观测噪声R
        self.R_scale_y = self.R1  # y方向缩放的观测噪声R
        self.R_theta = self.R1    # 旋转角度的观测噪声R
        self.R_trans_x = self.R1  # x方向平移的观测噪声R
        self.R_trans_y = self.R1  # y方向平移的观测噪声R
        
        # 累积变换参数（从第一帧到当前帧的总变换量）
        self.sum_scale_x = 0.0  # 累积x方向缩放因子
        self.sum_scale_y = 0.0  # 累积y方向缩放因子
        self.sum_theta = 0.0    # 累积旋转角度（弧度）
        self.sum_trans_x = 0.0  # 累积x方向平移量
        self.sum_trans_y = 0.0  # 累积y方向平移量
        
        # 卡尔曼滤波器当前状态（平滑后的参数）
        self.scale_x = 0.0   # 当前帧x方向缩放（滤波后）
        self.scale_y = 0.0   # 当前帧y方向缩放（滤波后）
        self.theta = 0.0     # 当前帧旋转角度（滤波后）
        self.trans_x = 0.0   # 当前帧x方向平移（滤波后）
        self.trans_y = 0.0   # 当前帧y方向平移（滤波后）
        
        # 平滑变换矩阵（2x3仿射矩阵，存储平滑后的变换参数）
        self.smoothed_mat = np.zeros((2, 3), dtype=np.float64)
        
        self.feature_type = feature_type  # 特征点检测类型，可选'gftt'、'orb'、'sift'，用于选择特征点提取算法
        
    def stabilize(self, frame_1, frame_2):
        """主要的稳定化函数"""
        # 转换为灰度图
        frame1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        # shape参数提前缓存
        h1, w1 = frame_1.shape[0], frame_1.shape[1]
        h2, w2 = frame_2.shape[0], frame_2.shape[1]
        c2 = frame_2.shape[2] if len(frame_2.shape) > 2 else 1

        # 计算垂直边界等比例地换算到垂直方向
        vert_border = int(self.HORIZONTAL_BORDER_CROP * h1 / w1)
        
        # 特征点检测（根据 feature_type 选择）
        if self.feature_type == 'gftt':
            features1 = cv2.goodFeaturesToTrack(frame1_gray, 
                                               maxCorners=200, 
                                               qualityLevel=0.01, 
                                               minDistance=30)
        elif self.feature_type == 'orb':
            orb = cv2.ORB_create(nfeatures=200)
            keypoints = orb.detect(frame1_gray, None)
            if not keypoints:
                features1 = None
            else:
                features1 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        elif self.feature_type == 'sift':
            try:
                sift = cv2.SIFT_create(nfeatures=200)
            except AttributeError:
                sift = cv2.xfeatures2d.SIFT_create(nfeatures=200)
            keypoints = sift.detect(frame1_gray, None)
            if not keypoints:
                features1 = None
            else:
                features1 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        else:
            raise ValueError(f"未知的特征点检测类型: {self.feature_type}")
        
        if features1 is None or len(features1) < 10:
            return frame_1
        
        # 光流追踪
        features2, status, err = cv2.calcOpticalFlowPyrLK(frame1_gray, 
                                                          frame2_gray, 
                                                          features1, 
                                                          None)
        # print(features2.shape, status.shape) #(200, 1, 2) (200,)
        # 筛选有效的特征点
        # 创建布尔掩码，标记哪些特征点被成功跟踪（status==1），flatten()保证是一维数组，便于后续索引
        mask = (status == 1).flatten()
        # 根据掩码筛选出被成功跟踪的特征点在第一帧中的位置
        good_features1 = features1[mask]
        # 根据掩码筛选出被成功跟踪的特征点在第二帧中的位置
        good_features2 = features2[mask]
        
        if len(good_features1) < 10:
            return frame_1
        
        # 将列表转换为NumPy数组，并指定数据类型为float32，便于后续OpenCV计算
        if good_features1.dtype != np.float32:
            good_features1 = np.array(good_features1, dtype=np.float32)
        if good_features2.dtype != np.float32:
            good_features2 = np.array(good_features2, dtype=np.float32)
        # print(good_features1.shape, good_features2.shape)
        # 估计仿射变换
        try:
            # 注意：OpenCV 4.x中estimateRigidTransform已被弃用，使用estimateAffinePartial2D
            affine_matrix, _ = cv2.estimateAffinePartial2D(good_features1, good_features2)
            # print(affine_matrix.shape) #(2, 3)
            if affine_matrix is None:
                return frame_1
        except Exception:
            return frame_1
        
        # 仿射变换矩阵 affine_matrix 说明：
        # 形状：2x3
        # 结构如下：
        # [ a11  a12  tx ]
        # [ a21  a22  ty ]
        # 其中：
        #   a11, a12, a21, a22 控制旋转、缩放（和剪切，如果是 estimateAffine2D）
        #   tx, ty 控制平移
        # 作用：
        #   [x', y'] = [a11 a12; a21 a22] * [x, y] + [tx, ty]
        #   即 x' = a11*x + a12*y + tx
        #      y' = a21*x + a22*y + ty

        # 提取变换参数
        dx = affine_matrix[0, 2]  # tx，x 方向的平移量
        dy = affine_matrix[1, 2]  # ty，y 方向的平移量
        da = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])  # 旋转角度，a21 和 a11 用于计算旋转

        # 如果旋转角度的余弦值不是极小（避免除以0），则计算去除旋转影响后的缩放因子
        cos_da = math.cos(da)
        if abs(cos_da) > 1e-6:
            ds_x = affine_matrix[0, 0] / cos_da  # 计算x方向的缩放因子
            ds_y = affine_matrix[1, 1] / cos_da  # 计算y方向的缩放因子
        else:
            ds_x = 1.0  # 若余弦值过小，直接设缩放为1，避免数值不稳定
            ds_y = 1.0
        
        sx = ds_x  # 最终x方向缩放因子
        sy = ds_y  # 最终y方向缩放因子
        
        # 累积变换参数
        self.sum_trans_x += dx      # 累积x方向的平移量，用于记录整体的水平移动
        self.sum_trans_y += dy      # 累积y方向的平移量，用于记录整体的垂直移动
        self.sum_theta   += da      # 累积旋转角度（弧度），用于记录整体的旋转变化
        self.sum_scale_x += ds_x    # 累积x方向的缩放因子，用于记录整体的水平缩放变化
        self.sum_scale_y += ds_y    # 累积y方向的缩放因子，用于记录整体的垂直缩放变化
        
        # 卡尔曼滤波（第一帧跳过）
        if self.k == 1:
            self.k += 1  # 第一帧不做滤波，直接跳过，k加1
        else:
            # 从第二帧开始，应用卡尔曼滤波进行参数平滑，传入当前参数，返回滤波后结果
            self.scale_x, self.scale_y, self.theta, self.trans_x, self.trans_y = \
                self.kalman_filter(self.scale_x, self.scale_y, self.theta, self.trans_x, self.trans_y)  # 更新为滤波后结果
        
        # 计算差值
        diff_scale_x = self.scale_x - self.sum_scale_x  # 当前帧x方向平滑缩放与累积缩放的差值
        diff_scale_y = self.scale_y - self.sum_scale_y  # 当前帧y方向平滑缩放与累积缩放的差值
        diff_trans_x = self.trans_x - self.sum_trans_x  # 当前帧x方向平滑平移与累积平移的差值
        diff_trans_y = self.trans_y - self.sum_trans_y  # 当前帧y方向平滑平移与累积平移的差值
        diff_theta   = self.theta   - self.sum_theta    # 当前帧平滑旋转角与累积旋转角的差值
        
        # 应用平滑参数
        ds_x = ds_x + diff_scale_x  # 用平滑后的缩放修正当前缩放
        ds_y = ds_y + diff_scale_y  # 用平滑后的缩放修正当前缩放
        dx   = dx   + diff_trans_x  # 用平滑后的平移修正当前平移
        dy   = dy   + diff_trans_y  # 用平滑后的平移修正当前平移
        da   = da   + diff_theta    # 用平滑后的旋转修正当前旋转
        
        # 创建平滑变换矩阵
        self.smoothed_mat[0, 0] = sx * math.cos(da)        # 仿射矩阵第一行第一列，包含x缩放和旋转
        self.smoothed_mat[0, 1] = sx * (-math.sin(da))     # 仿射矩阵第一行第二列，包含x缩放和旋转
        self.smoothed_mat[1, 0] = sy * math.sin(da)        # 仿射矩阵第二行第一列，包含y缩放和旋转
        self.smoothed_mat[1, 1] = sy * math.cos(da)        # 仿射矩阵第二行第二列，包含y缩放和旋转
        self.smoothed_mat[0, 2] = dx                      # 仿射矩阵第一行第三列，x方向平移
        self.smoothed_mat[1, 2] = dy                      # 仿射矩阵第二行第三列，y方向平移
        
        # 应用变换，指定更快的插值方式
        smoothed_frame = cv2.warpAffine(frame_1, self.smoothed_mat, (w2, h2), flags=cv2.INTER_LINEAR)
        
        # 裁剪边界以消除黑边
        smoothed_frame = smoothed_frame[vert_border:smoothed_frame.shape[0]-vert_border,
                                       self.HORIZONTAL_BORDER_CROP:smoothed_frame.shape[1]-self.HORIZONTAL_BORDER_CROP]  # 裁剪边界，去除上下和左右的黑边
        
        # 调整大小回原始尺寸，指定更快的插值方式
        smoothed_frame = cv2.resize(smoothed_frame, (w2, h2), interpolation=cv2.INTER_LINEAR)  # 将裁剪后的图像缩放回原始帧大小
        
        # 显示对比结果
        if self.test:
            # 复用canvas，只有尺寸变化时才重新分配
            if not hasattr(self, 'canvas') or self.canvas.shape != (h2, w2*2+10, c2):
                self.canvas = np.zeros((h2, w2*2+10, c2), dtype=np.uint8)
            canvas = self.canvas
            canvas[:, :smoothed_frame.shape[1]] = frame_1  # 左侧放原始帧
            canvas[:, smoothed_frame.shape[1]+10:smoothed_frame.shape[1]*2+10] = smoothed_frame  # 右侧放平滑后帧
            if canvas.shape[1] > 1920:
                canvas = cv2.resize(canvas, (canvas.shape[1]//2, canvas.shape[0]//2), interpolation=cv2.INTER_LINEAR)  # 如果画布太宽则缩小一半，适应屏幕显示
            cv2.imshow("Compared", canvas)  # 显示对比画面
        
        return smoothed_frame  # 返回平滑处理后的帧
    
    def kalman_filter(self, scale_x, scale_y, theta, trans_x, trans_y):
        """卡尔曼滤波器实现，传入当前参数，返回滤波后参数"""
        # 记录当前帧的各参数
        frame_1_scale_x = scale_x  # 当前帧x方向缩放
        frame_1_scale_y = scale_y  # 当前帧y方向缩放
        frame_1_theta = theta      # 当前帧旋转角度
        frame_1_trans_x = trans_x  # 当前帧x方向平移
        frame_1_trans_y = trans_y  # 当前帧y方向平移
        
        # 预测各参数的误差协方差（加上过程噪声Q）
        frame_1_err_scale_x = self.err_scale_x + self.Q_scale_x  # 预测x缩放误差
        frame_1_err_scale_y = self.err_scale_y + self.Q_scale_y  # 预测y缩放误差
        frame_1_err_theta = self.err_theta + self.Q_theta        # 预测旋转误差
        frame_1_err_trans_x = self.err_trans_x + self.Q_trans_x  # 预测x平移误差
        frame_1_err_trans_y = self.err_trans_y + self.Q_trans_y  # 预测y平移误差
        
        # 计算各参数的卡尔曼增益
        gain_scale_x = frame_1_err_scale_x / (frame_1_err_scale_x + self.R_scale_x)  # x缩放卡尔曼增益
        gain_scale_y = frame_1_err_scale_y / (frame_1_err_scale_y + self.R_scale_y)  # y缩放卡尔曼增益
        gain_theta = frame_1_err_theta / (frame_1_err_theta + self.R_theta)          # 旋转卡尔曼增益
        gain_trans_x = frame_1_err_trans_x / (frame_1_err_trans_x + self.R_trans_x)  # x平移卡尔曼增益
        gain_trans_y = frame_1_err_trans_y / (frame_1_err_trans_y + self.R_trans_y)  # y平移卡尔曼增益
        
        # 更新参数的估计值（滤波后结果）
        new_scale_x = frame_1_scale_x + gain_scale_x * (self.sum_scale_x - frame_1_scale_x)  # x缩放
        new_scale_y = frame_1_scale_y + gain_scale_y * (self.sum_scale_y - frame_1_scale_y)  # y缩放
        new_theta = frame_1_theta + gain_theta * (self.sum_theta - frame_1_theta)            # 旋转
        new_trans_x = frame_1_trans_x + gain_trans_x * (self.sum_trans_x - frame_1_trans_x)  # x平移
        new_trans_y = frame_1_trans_y + gain_trans_y * (self.sum_trans_y - frame_1_trans_y)  # y平移
        
        # 更新误差协方差
        self.err_scale_x = (1 - gain_scale_x) * frame_1_err_scale_x  # x缩放误差协方差
        self.err_scale_y = (1 - gain_scale_y) * frame_1_err_scale_y  # y缩放误差协方差
        self.err_theta = (1 - gain_theta) * frame_1_err_theta        # 旋转误差协方差
        self.err_trans_x = (1 - gain_trans_x) * frame_1_err_trans_x  # x平移误差协方差
        self.err_trans_y = (1 - gain_trans_y) * frame_1_err_trans_y  # y平移误差协方差
        
        # 返回滤波后的参数
        return new_scale_x, new_scale_y, new_theta, new_trans_x, new_trans_y
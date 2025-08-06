"""
交互式比较工具 - 提供更友好的用户界面
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from video_stabilization_comparison import VideoStabilizationComparison

class VideoComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频稳定算法比较工具")
        self.root.geometry("600x400")
        
        self.comparator = VideoStabilizationComparison()
        self.video_list = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="视频稳定算法比较工具", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 视频列表框架
        list_frame = ttk.LabelFrame(main_frame, text="视频列表", padding="5")
        list_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 视频列表
        self.video_listbox = tk.Listbox(list_frame, height=8)
        self.video_listbox.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.video_listbox.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # 按钮
        ttk.Button(button_frame, text="添加视频", command=self.add_video).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="删除选中", command=self.remove_video).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="清空列表", command=self.clear_videos).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="上移", command=self.move_up).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="下移", command=self.move_down).pack(side=tk.LEFT, padx=(0, 5))
        
        # 选项框架
        options_frame = ttk.LabelFrame(main_frame, text="输出选项", padding="5")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 预览选项
        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="实时预览", variable=self.preview_var).grid(row=0, column=0, sticky=tk.W)
        
        # 保存选项
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="保存视频", variable=self.save_var).grid(row=0, column=1, sticky=tk.W)
        
        # 输出文件路径
        ttk.Label(options_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.output_path_var = tk.StringVar(value="comparison_output.mp4")
        ttk.Entry(options_frame, textvariable=self.output_path_var, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        ttk.Button(options_frame, text="浏览", command=self.browse_output).grid(row=1, column=2, pady=(5, 0))
        
        # 开始比较按钮
        self.compare_button = ttk.Button(main_frame, text="开始比较", command=self.start_comparison)
        self.compare_button.grid(row=4, column=0, columnspan=3, pady=10)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="就绪")
        self.status_label.grid(row=6, column=0, columnspan=3)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        
    def add_video(self):
        file_paths = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("MP4文件", "*.mp4"),
                ("AVI文件", "*.avi"),
                ("M4V文件", "*.m4v"),
                ("所有文件", "*.*")
            ]
        )
        
        for file_path in file_paths:
            if file_path not in [item[0] for item in self.video_list]:
                # 获取文件名作为标签
                filename = file_path.split('/')[-1].split('\\')[-1]
                label = filename.rsplit('.', 1)[0]  # 去掉扩展名
                
                self.video_list.append((file_path, label))
                self.video_listbox.insert(tk.END, f"{label} - {file_path}")
                
        self.update_status(f"已添加 {len(self.video_list)} 个视频")
        
    def remove_video(self):
        selection = self.video_listbox.curselection()
        if selection:
            index = selection[0]
            del self.video_list[index]
            self.video_listbox.delete(index)
            self.update_status(f"已删除视频，剩余 {len(self.video_list)} 个")
            
    def clear_videos(self):
        self.video_list.clear()
        self.video_listbox.delete(0, tk.END)
        self.update_status("已清空视频列表")
        
    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("AVI文件", "*.avi"), ("所有文件", "*.*")]
        )
        if file_path:
            self.output_path_var.set(file_path)
            
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def start_comparison(self):
        if len(self.video_list) < 2:
            messagebox.showwarning("警告", "至少需要添加2个视频文件才能进行比较！")
            return
            
        # 禁用按钮
        self.compare_button.config(state='disabled')
        self.progress.start()
        
        # 在新线程中执行比较
        threading.Thread(target=self.run_comparison, daemon=True).start()
        
    def run_comparison(self):
        try:
            self.update_status("正在初始化比较器...")
            
            # 创建新的比较器
            comparator = VideoStabilizationComparison()
            
            # 添加视频
            for video_path, label in self.video_list:
                self.update_status(f"添加视频: {label}")
                if not comparator.add_video(video_path, label):
                    messagebox.showerror("错误", f"无法添加视频: {video_path}")
                    return
                    
            self.update_status("开始视频比较...")
            
            # 执行比较
            success = comparator.compare_videos(
                output_path=self.output_path_var.get(),
                preview=self.preview_var.get(),
                save_video=self.save_var.get()
            )
            
            if success:
                self.update_status("比较完成！")
                messagebox.showinfo("完成", "视频比较已完成！")
            else:
                self.update_status("比较失败")
                messagebox.showerror("错误", "视频比较失败")
                
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            
        finally:
            # 重新启用按钮
            self.progress.stop()
            self.compare_button.config(state='normal')

    def move_up(self):
        selection = self.video_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        idx = selection[0]
        # 交换video_list
        self.video_list[idx-1], self.video_list[idx] = self.video_list[idx], self.video_list[idx-1]
        # 交换Listbox显示
        temp = self.video_listbox.get(idx)
        self.video_listbox.delete(idx)
        self.video_listbox.insert(idx-1, temp)
        self.video_listbox.selection_clear(0, tk.END)
        self.video_listbox.selection_set(idx-1)
        self.update_status("已上移视频")

    def move_down(self):
        selection = self.video_listbox.curselection()
        if not selection or selection[0] == len(self.video_list)-1:
            return
        idx = selection[0]
        # 交换video_list
        self.video_list[idx+1], self.video_list[idx] = self.video_list[idx], self.video_list[idx+1]
        # 交换Listbox显示
        temp = self.video_listbox.get(idx)
        self.video_listbox.delete(idx)
        self.video_listbox.insert(idx+1, temp)
        self.video_listbox.selection_clear(0, tk.END)
        self.video_listbox.selection_set(idx+1)
        self.update_status("已下移视频")

def main():
    root = tk.Tk()
    app = VideoComparisonGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
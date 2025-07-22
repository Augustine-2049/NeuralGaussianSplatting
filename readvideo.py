'''
1. 读取视频
2. 读取视频的每一帧
3. 读取视频的每一帧的denoiser, feature, unet
4. 将denoiser, feature, unet显示在窗口中
5. 窗口中显示denoiser, feature, unet的细节
6. 窗口中显示denoiser, feature, unet的细节的坐标
7. 窗口中显示denoiser, feature, unet的细节的坐标对应的原始图像

'''

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

# 全局变量
image_id = None
current_photo = None
dir = r"G:\Lab\NGSassist\output\2fa108db-a\videos"
dir_list = os.listdir(dir)
#unet_data = np.load(unet_dir)
main_images = []
small_images = []
denoisers = []
features = []
#unets = []

# 得到目录下所有npz结尾的文件
dir_list = os.listdir(dir)
npz_list = [f for f in dir_list if f.endswith('.npz')]
np_names = [npz_file.split(".")[0] for npz_file in npz_list]
idx_names = {i: np_name for i, np_name in enumerate(np_names)}
names_idx = {np_name: i for i, np_name in enumerate(np_names)}
posidx_names = {
    "denoisers": 3,
    "features": 1,
    "aggregations": 5,
    "images": 7,
}

# 将文件读到字典中
data_dict = {}
for np_name in np_names:
    data = np.load(os.path.join(dir, np_name + ".npz"))
    data_dict[np_name] = data

# 将data组成列表，像上面这样
data_list = {np_name: []for np_name in np_names} 
for np_name, npz_data in data_dict.items():
    for key in npz_data.keys():
        data_list[np_name].append(npz_data[key])



# 创建主窗口
root = tk.Tk()
root.title("Image Viewer")

# 滑动条选择帧
frame_var = tk.IntVar(value=0)
slider = tk.Scale(root, from_=0, to=len(data_list["images"]) - 1, orient=tk.HORIZONTAL, variable=frame_var, command=lambda x: update_image())
slider.pack()


# Canvas 显示主图片
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack(side=tk.LEFT)

# 右侧坐标显示
coord_frame = tk.Frame(root)
coord_frame.pack(side=tk.RIGHT)
coord_label = tk.Label(coord_frame, text="X: , Y: ")
coord_label.pack()

# Toplevel 窗口显示小图
toplevel_list = [tk.Toplevel(root) for _ in range(len(npz_list))]
small_labels = []
for i, toplevel in enumerate(toplevel_list):
    toplevel_list[i].withdraw()
    toplevel_list[i].title(np_names[i])
    small_labels.append(tk.Label(toplevel))
    small_labels[i].pack()


# toplevel1 = tk.Toplevel(root)
# toplevel1.withdraw()  # 初始隐藏
# toplevel1.title("denoiser")#np_names[0])
# print(np_names[0])
# small_label1 = tk.Label(toplevel1)
# small_label1.pack()

# toplevel2 = tk.Toplevel(root)
# toplevel2.withdraw()  # 初始隐藏
# toplevel2.title(np_names[1])
# print(np_names[1])
# small_label2 = tk.Label(toplevel2)

# small_label2.pack()
# toplevel3 = tk.Toplevel(root)
# toplevel3.withdraw()  # 初始隐藏
# toplevel3.title(npz_list[2].split(".")[0])
# small_label3 = tk.Label(toplevel3)
# small_label3.pack()




# 将 NumPy 数组转换为 PhotoImage
def array_to_photo(arr):
    # 确保arr:uint8 格式
    img = Image.fromarray(arr)
    return ImageTk.PhotoImage(img)

# 更新主图片
def update_image(val=None):
    global image_id, current_photo
    frame_idx = frame_var.get()
    main_img = data_list["images"][frame_idx]
    photo = array_to_photo(main_img)
    if image_id:
        canvas.delete(image_id)
    image_id = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    current_photo = photo

def enlarge_pixels(img, scale=10):
    """将每个像素放大 scale 倍（非插值，纯像素复制）"""
    if img.ndim == 2:
        return np.kron(img, np.ones((scale, scale), dtype=img.dtype))
    else:
        print(img.shape, "Input array must be 2D or 3D")
        return img
        raise ValueError("Input array must be 2D or 3D")

def enlarge_rgb(img, scale=10):
    # img: (H, W, 3)
    img = np.repeat(img, scale, axis=0)  # 沿高度放大
    img = np.repeat(img, scale, axis=1)  # 沿宽度放大
    return img


# 鼠标移动事件
def on_motion(event):
    x, y = event.x, event.y
    if 0 <= x < 400 and 0 <= y < 400:
        coord_label.config(text=f"X: {x}, Y: {y}")
        frame_idx = frame_var.get()
        small_pics = {}
        if "denoisers" in data_list.keys():
            small_pics["denoisers"] = data_list["denoisers"][frame_idx][y, x]
        if "features" in data_list.keys():
            small_pics["features"] = data_list["features"][frame_idx][y, x]
        if "aggregations" in data_list.keys():
            # small_pics["all_aggregations"] = data_list["all_aggregations"][frame_idx][y, x]
            small_pics["aggregations"] = data_list["aggregations"][frame_idx][max(y-10, 0):min(y+10, 400), max(x-10, 0):min(x+10, 400)]
            # print(small_pics["all_aggregations"].shape)
        if "images" in data_list.keys():
            small_pics["images"] = data_list["images"][frame_idx][max(y-10, 0):min(y+10, 400), max(x-10, 0):min(x+10, 400)]


        enlarged_pics = {}
        for key in small_pics.keys():
            if small_pics[key].ndim == 2:
                enlarged_pics[key] = enlarge_pixels(small_pics[key], scale=25)
            elif small_pics[key].ndim == 3:
                enlarged_pics[key] = enlarge_rgb(small_pics[key], scale=10)
            else:
                print(small_pics[key].ndim, "Input array must be 2D or 3D")
        #enlarged_unet = enlarge_pixels(small_unet, scale=10)
        small_photos = {}

        for key in enlarged_pics.keys():
            small_photos[key] = array_to_photo(enlarged_pics[key])
        #small_unet_photo = array_to_photo(enlarged_unet)

        for i, toplevel in enumerate(toplevel_list):
            small_labels[i].config(image=small_photos[idx_names[i]])
            small_labels[i].image = small_photos[idx_names[i]]  # 防止垃圾回收
            toplevel_list[i].geometry(
                f"+{event.x_root + windows_pos[posidx_names[idx_names[i]]][0]}+{event.y_root + windows_pos[posidx_names[idx_names[i]]][1]}")




        # small_label1.config(image=small_photos[0])
        # small_label1.image = small_photos[0]  # 防止垃圾回收
        # small_label2.config(image=small_photos[1])
        # small_label2.image = small_photos[1]  # 防止垃圾回收
        # #small_label3.config(image=small_unet_photo)
        # #small_label3.image = small_unet_photo  # 防止垃圾回收
        # 定位 Toplevel 窗口
        # toplevel1.geometry(f"+{event.x_root + 30}+{event.y_root + 20}")
        # toplevel2.geometry(f"+{event.x_root - 150}+{event.y_root + 20}")
        # toplevel3.geometry(f"+{event.x_root - 150}+{event.y_root + 150}")
    else:
        coord_label.config(text="X: , Y: ")

windows_pos = [ [-100, 50],
                [0, 50], 
                [100, 50],
                [-200, 300],
                [0, 300], 
                [200, 300],
                [-200, 500],
                [0, 500], 
                [200, 500],
            ]





# 显示/隐藏 Toplevel 窗口
def on_enter(event):
    for toplevel in toplevel_list:
        toplevel.deiconify()
    # toplevel1.deiconify()
    # toplevel2.deiconify()

    #toplevel3.deiconify()
def on_leave(event):
    for toplevel in toplevel_list:
        toplevel.withdraw()
    # toplevel1.withdraw()
    # toplevel2.withdraw()
    #toplevel3.with

# 绑定事件
canvas.bind("<Motion>", on_motion)
canvas.bind("<Enter>", on_enter)
canvas.bind("<Leave>", on_leave)

# 初始显示第一帧
update_image()

# 启动主循环
root.mainloop()






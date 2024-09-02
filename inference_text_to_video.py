import torch
from model import Model
print(torch.cuda.is_available())  # 應該返回 True
print(torch.version.cuda)          # 應該顯示 12.4
print(torch.cuda.get_device_name(torch.cuda.current_device())) # GPU名稱
model = Model(device = "cuda", dtype = torch.float16)
prompt = "A horse galloping on a street"
# params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8, "chunk_size": 4}
out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)
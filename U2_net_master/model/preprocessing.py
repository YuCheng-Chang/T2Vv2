import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image


class RescaleT(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,image):
    # 檢查輸入是否為 PyTorch 張量
        if torch.is_tensor(image):
            # 將張量轉換為 NumPy 數組，並調整維度順序
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
            # 現在 image 的形狀應該是 (512, 512, 3)
            resized = transform.resize(image, (self.output_size, self.output_size), mode='constant')
            
            # 將調整大小後的圖像轉回 PyTorch 張量格式
            # resized_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
            return resized

            # img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
            # # lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
            
            # return img

    
    
class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag

    def __call__(self, image):
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)
        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
        # tmpImg = image
        tmpImg = tmpImg.transpose((2, 0, 1))# ndarray: channel x H x W

        return  torch.from_numpy(tmpImg).float()

# normalize the predicted SOD probability map
def normPred(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def resizePred(pred, output_dim):
    predict = pred.squeeze()  # 移除多餘的維度
    predict_np = predict.cpu().data.numpy()

    # 將預測結果轉換為 PIL Image（保持單通道）
    im = Image.fromarray((predict_np * 255).astype(np.uint8), mode='L')
    
    height, width = output_dim
    imo = im.resize((width, height), resample=Image.BILINEAR)
    
    # 轉換回 PyTorch tensor，保持單通道
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    imo = transform(imo).cuda()
    
    return imo.squeeze(0)  # 確保輸出是 2D tensor
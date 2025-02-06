import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import h5py
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import AdamW
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image
import os
import numpy as np
from sklearn.model_selection import train_test_split

from torchvision import transforms

class RainDataset(Dataset):
    def __init__(self, input_paths, target_paths):
        super().__init__()
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        input_image = Image.open(self.input_paths[idx]).convert("RGB")
        target_image = Image.open(self.target_paths[idx]).convert("RGB")
        
        input_image = self.transform(input_image)
        target_image = self.transform(target_image)
        return input_image, target_image

# GAN --------------------------------------------------------------------------------------------------------------------------------------- #
import torch
from torch import nn
from torch.nn import functional as F

        
class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fx = self.conv(x)
        
        if self.isn is not None:
            fx = self.isn(fx)
            
        fx = self.lrelu(fx)
        return fx
    
    
class Discriminator(nn.Module):
    """Basic Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(3, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x):
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        
        return fx
    
    
class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(6, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        
        return fx
    
import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)
        
    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
            
        return fx

    
class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)       
        
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
            
    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx

    
class Generator(nn.Module):
    """Encoder-Decoder model"""
    def __init__(self,):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(512, 512, dropout=True)
        self.decoder6 = DecoderBlock(512, 512, dropout=True)
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
 
        d8 = self.decoder8(e8)
        d7 = self.decoder7(d8)
        d6 = self.decoder6(d7)
        d5 = self.decoder5(d6)
        d4 = self.decoder4(d5)
        d3 = self.decoder3(d4)
        d2 = F.relu(self.decoder2(d3))
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)
    
    
class UnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model"""
    def __init__(self,):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder6 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder5 = DecoderBlock(2*512, 512)
        self.decoder4 = DecoderBlock(2*512, 256)
        self.decoder3 = DecoderBlock(2*256, 128)
        self.decoder2 = DecoderBlock(2*128, 64)
        self.decoder1 = nn.ConvTranspose2d(2*64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)
    



#CNN--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision

import numpy as np
import cv2
import random
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=False):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1), 
            nn.ReLU()
        )
        
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU()
        )
        
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 16, row, col))
        c = Variable(torch.zeros(batch_size, 16, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list



#Transformer----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
import os
import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.backends import cudnn

from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as T
from torchvision.transforms import RandomCrop 

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x



class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)



class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class Restormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_refinement=4,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])

        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
class RainDataset(Dataset):
    def __init__(self, input_dir, target_dir, patch_size=None, length=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_names = os.listdir(self.input_dir)
        self.patch_size = patch_size
        self.length = length or len(self.image_names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index %= len(self.image_names)
        input_path = os.path.join(self.input_dir, self.image_names[index])
        target_path = os.path.join(self.target_dir, self.image_names[index])
    
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
    
        input_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).float() / 255.0
    
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        if self.patch_size:
            ph, pw = self.patch_size, self.patch_size
            th = torch.randint(0, h - ph + 1, (1,)).item()
            tw = torch.randint(0, w - pw + 1, (1,)).item()
            input_tensor = input_tensor[:, th:th + ph, tw:tw + pw]
            target_tensor = target_tensor[:, th:th + ph, tw:tw + pw]
    
        return input_tensor, target_tensor, self.image_names[index], h, w


class Args:
	def __init__(self):
            self.data_path = '/kaggle/input/rain13kdataset/train/train/'
            self.save_path = 'results'
            self.num_blocks = [2, 3, 3, 4]
            self.num_heads = [1, 2, 4, 8]
            self.channels = [48, 96, 192, 384]
            self.expansion_factor = 2.66
            self.num_refinement = 4
            self.num_iter = 50
            self.lr = 1e-4
            self.batch_size = 16
            self.workers = 4
            self.model_file = None
args = Args()

class DummyModel:
    def predict(self, img):
        return img

model1 = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor)
model2 = UnetGenerator()
model3 = PReNet()

model1.load_state_dict(torch.load('D:\\CS331\\ThuyetTrinh\\model_bs4_epoch50.pth', map_location=torch.device('cpu')))
model2.load_state_dict(torch.load('D:\CS331\ThuyetTrinh\generator_bs4_alpha300_epoch50.pth', map_location=torch.device('cpu')))
model3.load_state_dict(torch.load('D:\\CS331\\ThuyetTrinh\\bs1\\PReNet_bs1_epoch50.pth', map_location=torch.device('cpu')))

model1.to('cpu')
model1.eval()
model2.to('cpu')
model2.eval()
model3.to('cpu')
model3.eval()

def process_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    return transform(image).unsqueeze(0).to('cpu')

def process_image_GAN(model, image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)), 
    ])
    input_tensor = transform(image).unsqueeze(0).to('cpu') 

    with torch.no_grad(): 
        output_tensor = model(input_tensor).squeeze(0).cpu() 

    output_image = transforms.ToPILImage()(output_tensor)
    return output_image

def process_image_CNN(model, image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    input_tensor = transform(image).unsqueeze(0).to('cpu')

    with torch.no_grad():
        output_tensor, _ = model(input_tensor)
        output_tensor = output_tensor.squeeze(0).cpu()


    output_image = transforms.ToPILImage()(output_tensor)
    return output_image

def process_image_Transformer(model, input_image):
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert('RGB')
    
    input_image = input_image.resize((256, 256))
    width, height = input_image.size
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    input_image = input_image.resize((new_width, new_height))

    input_tensor = ToTensor()(input_image).unsqueeze(0).to('cpu')

    model.eval()
    with torch.no_grad():
        denoised_tensor = torch.clamp(model(input_tensor), 0, 1).squeeze(0).cpu()

    to_pil = ToPILImage()
    denoised_image = to_pil(denoised_tensor)
    return denoised_image

def get_image_size(image):
    return image.size

def resize_with_aspect_ratio(image, target_size):
    if isinstance(image, torch.Tensor):
        image = convert_to_pil(image)
    
    width, height = image.size
    target_width, target_height = target_size

    width_ratio = target_width / width
    height_ratio = target_height / height

    scale_factor = min(width_ratio, height_ratio)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

def show_images_in_row(images, titles, original_size):
    col1, col2, col3 = st.columns(3)
    
    resized_images = [resize_with_aspect_ratio(image, original_size) for image in images]

    with col1:
        st.subheader(titles[0])
        st.image(resized_images[0], use_container_width=True)
        
    with col2:
        st.subheader(titles[1])
        st.image(resized_images[1], use_container_width=True)
        
    with col3:
        st.subheader(titles[2])
        st.image(resized_images[2], use_container_width=True)

def convert_to_pil(tensor_image):
    if isinstance(tensor_image, torch.Tensor):
        tensor_image = tensor_image.squeeze(0) 
        tensor_image = tensor_image.permute(1, 2, 0).detach().cpu().numpy()  
        tensor_image = (tensor_image * 255).astype(np.uint8) 
        return Image.fromarray(tensor_image)
    elif isinstance(tensor_image, Image.Image):
        return tensor_image
    else:
        raise ValueError("Input is neither a tensor nor a PIL image")

st.title("Image Prediction with 3 Models")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    original_size = get_image_size(image)

    processed_image = process_image(image)

    output_image_model1 = process_image_Transformer(model1, image)
    output_image_model2 = process_image_GAN(model2, image)
    output_image_model3 = process_image_CNN(model3, image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    st.subheader("Model Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Restormer Prediction")
        st.image(output_image_model1, use_container_width=True)

    with col2:
        st.subheader("Pix2Pix Prediction")
        st.image(output_image_model2, use_container_width=True)

    with col3:
        st.subheader("PReNet Prediction")
        st.image(output_image_model3, use_container_width=True)

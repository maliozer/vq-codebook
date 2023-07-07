print("Installing CLIP...")
!git clone https://github.com/openai/CLIP
print("Installing Python Libraries for AI...")
!git clone https://github.com/CompVis/taming-transformers
!pip install transformers
!pip install ftfy regex tqdm omegaconf pytorch-lightning
!pip install kornia
!pip install einops
!pip install wget
!pip install pytorch-lightning

print("Installing libraries for metadata management...")
!pip install stegano
!apt install exempi
!pip install python-xmp-toolkit
!pip install imgtag
!pip install pillow==7.1.2

print("Installing Python libraries for creating videos...")
!pip install imageio-ffmpeg
!mkdir steps

print("Installation completed.")

#%%

!cp -R /content/drive/MyDrive/data/vqgan/2023-05-02T01-17-30_pororo2/images/* /content/images
!unzip /content/drive/MyDrive/research/storycont/blip_images.zip -d /content/

#%%


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.nn import functional as F

import sys
sys.path.append('./taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan



#%%

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

#%%



def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        print(config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


#%%

MODEL_PATH = "/content/drive/MyDrive/data/vqgan/2023_pororo2/checkpoints/last.ckpt"
MODEL_CONFIG_PATH = "/content/drive/MyDrive/data/vqgan/2023_pororo2/configs/2023-05-02T01-17-30-project.yaml"

model = load_vqgan_model(MODEL_CONFIG_PATH, MODEL_PATH)

def read_img(image_path):
    image = Image.open(image_path)

    image=(np.asarray(image)-127.5)/127.5
    tensor_image=torch.tensor(image,dtype=torch.float32)
    tensor_image = tensor_image.reshape(1,tensor_image.shape[0], tensor_image.shape[1], tensor_image.shape[2])
    # tensor_image = tensor_image.type(torch.FloatTensor)
    tensor_image = tensor_image.permute(0,3,1,2)
    return tensor_image

# example ground image
image_tensor = read_img("/content/blip_files/Pororo_ENGLISH1_1_ep10_19.jpg")


#%% 

# model.encode make the same thing as the function below
# returns quant, emb_loss, info (info has code indices)
def vector_quantize(x, codebook):
  d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
  indices = d.argmin(-1)
  x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
  return replace_grad(x_q, x).movedim(3, 1), indices

# synt z
z_q, indices = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight)


#%%

# encode image to quantized latent
z_quant, emb_loss, info = model.encode(image_tensor)
z_quant = z_quant.movedim(1,3).shape

# info[2] has indices for codebook to use as condition

# decode from z
z_dec = model.decode(z_quant)
z_dec = torch.clamp(z_dec, min=-1, max=1)
z_dec = z_dec.permute(0,2,3,1).detach().numpy()

z_dec=(((z_dec+1)/2)*255).astype("uint8")
plt.imshow(z_dec[0])

# codebook
print(model.quantize.embedding.weight.shape)
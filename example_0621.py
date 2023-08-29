# 由于默认huggingface缓存目录空间不足，更换为自定义文件缓存目录
# 也可以使用环境变量直接在终端输入‘export HF_HOME=/data/ML_document/.cache/huggingface’
import os
import time
os.environ["HF_HOME"] = "/data/ML_document/.cache/huggingface"

# Copied from 'https://github.com/dbolya/tomesd' on  23.06.23
import torch, tomesd
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Apply ToMe with a 50% merging ratio
tomesd.apply_patch(pipe, ratio=1.0) # Can also use pipe.unet in place of pipe here

time_start = time.time()
image = pipe("a photo of an astronaut riding a green dog on mars").images[0]
time_end = time.time()
print(f"the time of generate the pic is {time_end - time_start} second!")
image.save("astronaut.png")

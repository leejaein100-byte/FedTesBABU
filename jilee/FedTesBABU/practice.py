import os, torch
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count=", torch.cuda.device_count())
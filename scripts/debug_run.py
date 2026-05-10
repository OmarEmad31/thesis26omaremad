import pandas as pd
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
print("Pandas OK")
import torch
print("Torch OK")
import timm
print("Timm OK")
df = pd.read_csv("video_audit_report.csv")
print(f"CSV OK, rows: {len(df)}")
device = "cpu"
model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device)
print("Model OK")

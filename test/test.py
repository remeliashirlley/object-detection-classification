#python3 detect.py --weights /Users/remeliashirlley/Desktop/final_model/best.pt --img 416 --conf 0.6 --source /Users/remeliashirlley/Desktop/MDP_IR/images/23. Red Letter T/01092022_015049.jpg

import torch
import numpy as np
import cv2
from time import time
import glob
import os
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

model = torch.hub.load('/Users/remeliashirlley/Desktop/final_model/yolov5', 'custom', path='/Users/remeliashirlley/Desktop/final_model/best.pt', force_reload=True, source='local')
imageName='/Users/remeliashirlley/Desktop/MDP_IR/test_images/rpi images/1. Target/01092022_001508.jpg'
results = model(imageName)
#for imageName in glob.glob('/Users/remeliashirlley/Desktop/MDP_IR/test_images/rpi images/33. Red Arrow Down/*.jpg'): #assuming JPG
#display(Image(filename=imageName))
#print("\n")
#results = model(imageName)

# Results
#results.save(results.save(save_dir='/Users/remeliashirlley/Desktop/final_model/runs/detect/exp') )  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()
results.xyxy[0]  # im predictions (tensor)

results.pandas().xyxy[0]  # im predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

results.pandas().xyxy[0]=results.pandas().xyxy[0].sort_values(by='confidence', ascending=False)

results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
pred_class=results.pandas().xyxy[0]['name'].iloc[0] #return str

if pred_class =='Bullseye': print(True)
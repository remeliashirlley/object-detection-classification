import torch
import time
from imutils import paths
from PIL import Image
import os
import torchvision
# import cv2
from datetime import datetime


def prediction(imageName):
    start=time.time()
    model = torch.hub.load('/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/yolov5', 'custom', path='/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/best.pt', force_reload=True, source='local')
    results = model(imageName)
    results.files[0]=datetime.now().strftime("%y%m%d%H%M%s")+'.jpg'
    results.save(save_dir='/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/exp/')  # or .show(), .save(), .crop(), .pandas(), etc.
    
    try:
        output=results.pandas().xyxy[0]
        output=output[output['name']!='Bullseye'].sort_values(by='confidence', ascending=False)
        pred_class=output['name'].iloc[0] #return str

        id={'1':'11', '2':'12', '3':'13', '4':'14', '5':'15', '6':'16', '7':'17', '8':'18', '9':'19', 
            'A':'20', 'B':'21', 'C':'22', 'D':'23', 'E':'24', 'F':'25', 'G':'26', 'H':'27', 'S': '28',
            'T':'29', 'U':'30', 'V':'31', 'W':'32', 'X':'33', 'Y':'34', 'Z':'35', 'Up':'36', 'Down':'37',
            'Right':'38', 'Left':'39', 'Circle':'40', 'Bullseye':'Bullseye'}
        print("Model: ", time.time()-start)
        return pred_class
    except: return 'Error'
    
    

def stiching():
    start=time.time()
    image_folder = r'/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/exp'
    imagePaths = list(paths.list_images(image_folder))
    images = [Image.open(x) for x in imagePaths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.show()
    #new_im.save('/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/stitched/output.jpg', format='JPEG')
    print("Stiching: ", time.time()-start)

if __name__ == "__main__":
    start=time.time()
    count=0
    # for i in os.listdir('/Users/remeliashirlley/Desktop/fordata'):
    #     image=os.path.join('/Users/remeliashirlley/Desktop/fordata', i)
    #     print(image)
    #     prediction(image)
    # while count<2:
    print(prediction('/Users/remeliashirlley/Desktop/01102022_103739.jpg'))
        # count+=1

    # if count==2:
    #     stiching()
    print(time.time()-start)
   
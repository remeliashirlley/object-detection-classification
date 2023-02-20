### CONNECT TO MDP30 WIFI!!!

### CHANGE MAC NETWORK ADDRESS!!!

import imagezmq
import torch
from imutils import paths
from PIL import Image
from datetime import datetime
import os

class wifireceiver():
       
    def __init__(self):
        self.imgHub=imagezmq.ImageHub(open_port='tcp://192.168.30.15:5555') #socket programming
        self.count=0

    def receive(self):
        rpi_name,image=self.imgHub.recv_image()
        self.imgHub.send_reply(bytes(self.prediction(image), 'utf-8'))
        # self.count+=1

    def prediction(self, imageName):
        model = torch.hub.load('/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/yolov5', 'custom', path='/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/best.pt', force_reload=True, source='local')
        results = model(imageName)
        results.files[0]=datetime.now().strftime("%y%m%d%H%M%s")+'.jpg'
        # results.save(save_dir='/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/exp/')  # or .show(), .save(), .crop(), .pandas(), etc.

        try:
            output=results.pandas().xyxy[0]
            print(output)
            if self.count==0:
                output=output[(output['name']=='Left') | (output['name']=='Right')]
                print(output['name'])
            elif self.count==1:
                output=output[(output['name']=='Left2') | (output['name']=='Right2')]
                print(output['name'])
            output=output.sort_values(by='confidence', ascending=False)
            pred_class=output['name'].iloc[0] #return str
            print(pred_class)

            # id={'1':'11', '2':'12', '3':'13', '4':'14', '5':'15', '6':'16', '7':'17', '8':'18', '9':'19', 
            #     'A':'20', 'B':'21', 'C':'22', 'D':'23', 'E':'24', 'F':'25', 'G':'26', 'H':'27', 'S': '28',
            #     'T':'29', 'U':'30', 'V':'31', 'W':'32', 'X':'33', 'Y':'34', 'Z':'35', 'Up':'36', 'Down':'37',
            #     'Right':'38', 'Left':'39', 'Circle':'40', 'Bullseye':'Bullseye'}
            id={'Left':'l', 'Right':'r', 'Left2':'l', 'Right2':'r'}
            res=id[pred_class]
            results.save(save_dir='/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/exp/')  # or .show(), .save(), .crop(), .pandas(), etc.
            self.count+=1
            return res
        except: return 'Error'
        
    def stitching(self):
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
        new_im.save('/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/stitched/output.jpg', format='JPEG')

def main(x):
    a=wifireceiver()
    while(a.count<x): a.receive()
    if (a.count==x): a.stitching()

if __name__=='__main__':
    dir = '/Users/remeliashirlley/Desktop/MDP_IR/models/final_model/runs/detect/exp/'
    for file in os.listdir(dir): os.remove(os.path.join(dir, file))
    main(2)
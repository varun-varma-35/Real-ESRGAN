import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=2)
    model.load_weights('weights/RealESRGAN_x2.pth', download=True)
    file_path = 'inputs/lr_slide.jpeg'
    image = Image.open(file_path).convert('RGB')
    sr_image = model.predict(image)
    sr_image.save('results/hr_slide.png')

if __name__ == '__main__':
    main()
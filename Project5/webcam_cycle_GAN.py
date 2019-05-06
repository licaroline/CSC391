"""
Quick program for utilizing pretrained Cycle GANs on webcam input. 

Example usage:
python3 test2.py --dataroot datasets --name style_cezanne_pretrained --model test --no_dropout --gpu_ids -1 --preprocess none
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import util
from data.base_dataset import get_transform
from torchvision import transforms
from PIL import Image
import cv2
import torch

# hard-coded values which can be adjusted
scaling_factor = 0.30   # 1 is original image size, 0.5 is half; adjust based on desired resolution
fps = 1                 # desired FPS

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    transform = get_transform(opt, grayscale=(opt.input_nc == 1))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    if opt.eval:
        model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)

    while True:
        # get input from webcam
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(frame, 'RGB')

        # transform frame to desired format
        transformed = transform(image)
        x, y, z = transformed.size()
        data = {'A': transformed.view(1, x, y, z), 'A_paths': ['']}

        model.set_input(data)
        model.test()
        
        # get results from model
        visuals = model.get_current_visuals()

        im_data = visuals['fake_B']
        im = util.tensor2im(im_data)
        im = cv2.resize(im, None, fx=1.0/scaling_factor, fy=1.0/scaling_factor, interpolation=cv2.INTER_AREA)
        
        # display results
        cv2.imshow('CycleGAN', im)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
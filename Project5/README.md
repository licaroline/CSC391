## Final Project


### About the code
This script makes use of the [PyTorch implementation of CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Example usage: 
```bash
python3 test2.py --dataroot datasets --name style_cezanne_pretrained --model test --no_dropout --gpu_ids -1 --preprocess none
```
Whatever you put in `--dataroot` doesnt really matter, the code will utilize images taken from the webcam rather than images found in whatever directory you put there. 

There are a few other pretrained models provided in the CycleGAN implementation that can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3).

Due to limitations in computing power, I lowered the resolution and FPS for images captured from the webcam, however these parameters can be adjusted in the script if desired.
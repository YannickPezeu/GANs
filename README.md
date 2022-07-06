# GANs
Generative adversarial network learning to draw MNIST

## 1 TF_GAN is an implementation that can be found on the tensorflow site which uses a dense network to produce the images. 

## 2 Conv_GAN is a custom implementation I realized with pytorch using convolution layers.

The results can be seen on my website: http://www.pezeuy.com and the superiority of the convolutional version is obvious.

To reproduce the results just clone the repo, navigate to TFGAN and run python TFGAN.py. Then navigate to Conv_GAN and run python Conv_GAN.py 
Then you can run create_gif.py to create the gif animation of the network training.


simpleGAN | ![simpleGan](https://github.com/YannickPezeu/GANs/blob/master/simpleGan.png) 
--- | --- 
convGAN | ![convGan](https://github.com/YannickPezeu/GANs/blob/master/convGan.png) 

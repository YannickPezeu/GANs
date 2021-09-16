import imageio
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt


imageio.imread("Conv_GAN/results4/image_grid72.png")

def create_gif(path):
    print("create_gif")
    mypath = "results"
    paths = sorted(Path(path).iterdir(), key=os.path.getmtime)
    print("paths")
    print(type(paths))
    print(type([1,2]))
    onlyfiles = [ f for f in paths if (isfile(f))  ]
    print(onlyfiles)
    images = []
    for filename in onlyfiles:
        if 'png' in str(filename):
            images.append(imageio.imread(filename))

    last_image = images[-1]
    for i in range(len(images)):
        images.append(last_image)
    try:
        os.mkdir('results_gif')
    except:
        print('can\'t create dir')
    imageio.mimsave('results_gif/'+ path +'.gif', images)
    print('results_gif/'+ path + '.gif')


path = "simple_GAN/result_images2"
create_gif(path)

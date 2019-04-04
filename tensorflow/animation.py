from PIL import Image
import numpy as np
import imageio
import os

ipath = "iterations/iteration_%d.png"

images = []
nb_images = len(next(os.walk('iterations'))[2])

for i in range(nb_images):
    images.append(imageio.imread(ipath%(i*1000)))
imageio.mimsave('animation/animation.gif', images, fps=1)

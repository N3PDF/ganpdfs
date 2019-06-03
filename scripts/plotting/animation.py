#!/usr/bin/env python3
from PIL import Image
import numpy as np
import imageio
import os

ipath = "iterations/pdf_generated_at_training_%d.png"

images = []
nb_images = len(next(os.walk('iterations'))[2])

for i in range(1,nb_images+1):
    images.append(imageio.imread(ipath%(i*1000)))
imageio.mimsave('animation.gif', images, fps=1)

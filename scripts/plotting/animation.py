#!/usr/bin/env python3
import os
import imageio
import argparse
from PIL import Image
import numpy as np

try:
    import imageio
    from PIL import Image
except:
    print("Either *imageio* or *PIL* is missing!")
    raise

def animate_plot(ipath, nb_images, fps):
    images = []
    for i in range(1,nb_images+1):
        images.append(imageio.imread(ipath%(i*1000)))
    imageio.mimsave('animation.gif', images, fps=fps)

def main(args):
    """
    Plot the generated PDFs for each iterations and
    save the result as a gif.
    """

    # Check the input parameters
    if not os.path.exists(args.folder):
        raise Exception(f'{args.folder} does not exist.')
    input_folder = args.folder.strip('/')
    if args.fps == None:
        fps = 1
    elif args.fps > 0:
        fps = args.fps
    else:
        raise Exception(f'{args.fps} is not valid. Value should be positive.')

    # Set the path
    ipath = "{0}/pdf_generated_at_training_%d.png".format(input_folder)
    # Count the number of images
    nb_images = len(next(os.walk(input_folder))[2])
    # Plot the animation
    animate_plot(ipath, nb_images, fps)


if __name__ == "__main__":
    """
    Read command line argument
    """
    parser = argparse.ArgumentParser(description='Animate GAN plots')
    parser.add_argument('folder', help='Take as input the *iterations* folder.')
    parser.add_argument('--fps', default=None, type=int, help='Number of frame/s.')
    args = parser.parse_args()
    main(args)


import morphsnakes
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread
import os
import matplotlib.pyplot as plt

path = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\a"
outPath = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\b"

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T**2, 0))
    u = np.float_(phi > 0)
    return u

def evolve_visual(msnake, output, levelset=None, num_iters=20, background=None):
    """
    Visual evolution of a morphological snake.

    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    background : array-like, optional
        If given, background will be shown behind the contours instead of
        msnake.data.
    """
    from matplotlib import pyplot as ppl

    if levelset is not None:
        msnake.levelset = levelset

    # Prepare the visual environment.
    fig = ppl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(output)
    ax1 = fig.add_subplot(1, 3, 2)
    if background is None:
        ax1.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax1.imshow(background, cmap=ppl.cm.gray)
    ax1.contour(msnake.levelset, [0.5], colors='r')

    ax2 = fig.add_subplot(1, 3, 3)
    ax_u = ax2.imshow(msnake.levelset)
    ppl.pause(0.001)

    # Iterate.
    for i in range(num_iters):
        # Evolve.
        msnake.step()

#         Update figure.
        del ax1.collections[0]
        ax1.contour(msnake.levelset, [0.5], colors='r')
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()
        ppl.pause(0.001)

    # Return the last levelset.
    return msnake.levelset

def test_1():
    # Load the image.
    img = imread("output.jpg")[..., 0]/255.0

    # g(I)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.41, balloon=1)
    mgac.levelset = circle_levelset(img.shape, (148, 200), 20)

    # Visual evolution.
    plt.figure()
    evolve_visual(mgac, img, num_iters=150, background=img)
    mpimg.imsave('crop.jpg',res)


def test_2():
    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)
    # Load the image.
        imgcolor = imread(input_path)/255.0
        img = rgb2gray(imgcolor)
    
        # g(I)
        gI = morphsnakes.gborders(img, alpha=1000, sigma=2)
    
        # Morphological GAC. Initialization of the level-set.
        mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-1)
        mgac.levelset = circle_levelset(img.shape, (200, 250), 210, scalerow=0.75)
    
        # Visual evolution.
        plt.figure()
        res = evolve_visual(mgac, img,num_iters=320, background=imgcolor)
        fullpath = os.path.join(outPath, 'f'+image_path)
        mpimg.imsave(fullpath,res)


def test_3():
    # Load the image.
    imgcolor = imread("output.jpg")/255.0
    img = rgb2gray(imgcolor)

    # MorphACWE does not need g(I)

    # Morphological ACWE. Initialization of the level-set.
    macwe = morphsnakes.MorphACWE(img, smoothing=3, lambda1=1, lambda2=1)
    macwe.levelset = circle_levelset(img.shape, (200, 170), 25)

    # Visual evolution.
    plt.figure()
    evolve_visual(macwe, img, num_iters=100, background=imgcolor)


if __name__ == '__main__':
    print("""""")
    # test_1()
    test_2()
    # test_3()
    plt.show()

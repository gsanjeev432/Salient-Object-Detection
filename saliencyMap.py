# numpy for all calculations
import numpy as np
# for additional image calculations
from scipy import ndimage


def getSaliencyMapNumpy(labImage, scales=3):
    (height, width, channels) = labImage.shape
    minimumDimension = min(width, height)

    # saliency map
    saliencyMap = np.zeros(shape=(height, width))

    # calculate neighbourhood means for every scale and channel
    for s in range(0, scales):
        offset = np.round(minimumDimension / (2 ** (s + 1))).astype(int)
        radius = offset * 2 + 1
        filterMask = np.pad(np.ones((height, width)), offset, mode='constant', constant_values=0)
        filterFix = ndimage.uniform_filter(filterMask, radius, mode='constant', cval=0.0)
        filterFix = filterFix[offset:-offset, offset:-offset]
        for c in range(0, channels):
            saliencyMap += (labImage[:, :, c] - ndimage.uniform_filter(labImage[:, :, c], radius, mode='constant',
                                                                       cval=0.0) / filterFix) ** 2

    return saliencyMap

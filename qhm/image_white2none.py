"""image_white2none.m -- make the white background of a PNG transparent.

`regionprops(alpha,'BoundingBox')` on a 0/1 label image is just the bounding box
of the nonzero region, computed directly here. MATLAB's `img2 = img` keeps img's
uint8 class, so the conv2 results are rounded and saturated before comparison --
reproduced, because it changes which pixels end up transparent.
"""

import numpy as np


def image_white2none(filenamein, filenameout, *varargin):
    from PIL import Image
    from scipy.signal import convolve2d

    # example useage: image_white2none('2.png','2t.png');

    # read in .png file with alpha layer

    method1 = False
    recenter = True

    threshold = 255
    if len(varargin) > 0:
        threshold = varargin[0]
        if len(varargin) > 1:
            if varargin[1] is True:
                method1 = True

    img = np.asarray(Image.open(str(filenamein)).convert('RGB'))

    alpha = np.ones((img.shape[0], img.shape[1]))

    if method1:
        # not work do not know why rgb_sum = double(img(:,:,1))+...;
        # alpha( find(rgb_sum)>=threshold ) = 0;

        alpha[(img[:, :, 0] >= threshold) & (img[:, :, 1] >= threshold)
              & (img[:, :, 2] >= threshold)] = 0
        # alpha( find( img(:,:,1)==255&... ) ) = 0;
    else:
        ksize = 2   # 10

        A = threshold * np.ones((img.shape[0], img.shape[1]))
        B = np.ones((ksize, ksize)) / (ksize * ksize)

        S = convolve2d(A, B, mode='same')

        img2 = img.copy()
        for k in range(3):
            # because matlab always padded 0.
            c = convolve2d(255.0 - img[:, :, k].astype(np.float64), B, mode='same')
            img2[:, :, k] = np.clip(np.rint(c), 0, 255).astype(np.uint8)

        inv = 255 - img2.astype(np.int16)   # uint8 arithmetic, no wraparound
        alpha[(inv[:, :, 0] >= threshold) & (inv[:, :, 1] >= threshold)
              & (inv[:, :, 2] >= threshold)] = 0

    if recenter:
        rows, cols = np.nonzero(alpha)
        # BoundingBox = [minCol-.5, minRow-.5, nCols, nRows], then ceil()
        min_i = int(cols.min())                      # 0-based first column
        max_i = min_i + int(cols.max() - cols.min() + 1)
        min_j = int(rows.min())
        max_j = min_j + int(rows.max() - rows.min() + 1)

        max_i = min(max_i, img.shape[1] - 1)
        max_j = min(max_j, img.shape[0] - 1)

        img2 = img[min_j:max_j + 1, min_i:max_i + 1, :]
        alpha2 = alpha[min_j:max_j + 1, min_i:max_i + 1]
        # imshow(img2);

        _imwrite_alpha(img2, alpha2, filenameout)
    else:
        _imwrite_alpha(img, alpha, filenameout)


def _imwrite_alpha(img, alpha, filename):
    from PIL import Image
    a = np.clip(np.rint(np.asarray(alpha, dtype=np.float64) * 255), 0, 255)
    rgba = np.dstack([np.asarray(img, dtype=np.uint8), a.astype(np.uint8)])
    Image.fromarray(rgba, mode='RGBA').save(str(filename))

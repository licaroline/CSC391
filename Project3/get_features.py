from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hog, blob_log, blob_dog, blob_doh, local_binary_pattern
from skimage import data,  exposure, io
from skimage.color import rgb2gray, label2rgb
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.transform import rotate

plt.rcParams['font.size'] = 9
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def match_gabor(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def match_lbp(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

def get_HOG(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    print(fd.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.figure(2)
    plt.plot(fd)

    plt.show()

def get_blobs(image, threshold = 0.3):
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
            'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()

def get_gabor(agr, non):
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    shrink = (slice(0, None, 3), slice(0, None, 3))
    agriculture = img_as_float(agr)[shrink]
    non_agriculture = img_as_float(non)[shrink]

    image_names = ('agriculture', 'nonagriculture')
    images = (agriculture, non_agriculture)

    # prepare reference features
    ref_feats = np.zeros((2, len(kernels), 2), dtype=np.double)
    ref_feats[0, :, :] = compute_feats(agriculture, kernels)
    ref_feats[1, :, :] = compute_feats(non_agriculture, kernels)

    print('Rotated images matched against references using Gabor filter banks:')

    print('original: agriculture, rotated: 30deg, match result: ', end='')
    feats = compute_feats(ndi.rotate(agriculture, angle=190, reshape=False), kernels)
    print(image_names[match_gabor(feats, ref_feats)])

    print('original: agrictulture, rotated: 70deg, match result: ', end='')
    feats = compute_feats(ndi.rotate(agriculture, angle=70, reshape=False), kernels)
    print(image_names[match_gabor(feats, ref_feats)])

    print('original: non-agriculture, rotated: 145deg, match result: ', end='')
    feats = compute_feats(ndi.rotate(non_agriculture, angle=145, reshape=False), kernels)
    print(image_names[match_gabor(feats, ref_feats)])

    # Plot a selection of the filter bank kernels and their responses.
    results = []
    kernel_params = []
    for theta in (0, 1):
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, [power(img, kernel) for img in images]))

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(5, 6))
    plt.gray()

    fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

    axes[0][0].axis('off')

    # Plot original images
    for label, img, ax in zip(image_names, images, axes[0][1:]):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis('off')

    for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
        # Plot Gabor kernel
        ax = ax_row[0]
        ax.imshow(np.real(kernel), interpolation='nearest')
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot Gabor responses with the contrast normalized for each filter
        vmin = np.min(powers)
        vmax = np.max(powers)
        for patch, ax in zip(powers, ax_row[1:]):
            ax.imshow(patch, vmin=vmin, vmax=vmax)
            ax.axis('off')

    plt.show()

def get_lbp(agr, non):
    refs = {
        'agriculture': local_binary_pattern(agr, n_points, radius, METHOD),
        'nonagriculture': local_binary_pattern(non, n_points, radius, METHOD)
    }

    # classify rotated textures
    print('Rotated images matched against references using LBP:')
    print('original: agriculture, rotated: 30deg, match result: ',
        match_lbp(refs, rotate(agr, angle=30, resize=False)))
    print('original: agriculture, rotated: 70deg, match result: ',
        match_lbp(refs, rotate(agr, angle=70, resize=False)))
    print('original: nonagriculture, rotated: 145deg, match result: ',
        match_lbp(refs, rotate(non, angle=145, resize=False)))

    # plot histograms of LBP of textures
    fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(nrows=2, ncols=2,
                                                        figsize=(9, 6))
    plt.gray()

    ax1.imshow(agr)
    ax1.axis('off')
    hist(ax4, refs['agriculture'])
    ax4.set_ylabel('Percentage')

    ax2.imshow(non)
    ax2.axis('off')
    hist(ax5, refs['nonagriculture'])
    ax5.set_xlabel('Uniform LBP values')

    plt.show()


if __name__ == "__main__":
    print("Possible features to extract: HOG, LBP, Blob, Gabor")
    print("Images are named with numbers between 0 and 199. 0-99 are of agriculture, 100-199 are not.")
    print("Enter 'q' to quit.")
    user_input = input("Enter desired feature: ")

    while user_input != 'q':
        if user_input == "HOG":
            image_name = input("Enter number corresponding to an image: ")

            if int(image_name) < 0 or int(image_name) > 199:
                print("Invalid number.")
            else:
                image = io.imread("Images\\" + str(image_name) + ".png")
                get_HOG(image)
        elif user_input == "LBP":
            agr_name = input("Enter number between 0-99 corresponding to an agriculture image: ")
            non_name = input("Enter number between 100-199 corresponding to a non-agriculture image: ")

            if int(agr_name) < 0 or int(non_name) < 100 or int(agr_name) > 99 or int(non_name) > 199:
                print("Invalid number.")
            else:
                agr_image = io.imread("Images\\" + str(agr_name) + ".png", as_gray=True)
                non_image = io.imread("Images\\" + str(non_name) + ".png", as_gray=True)
                get_lbp(agr_image, non_image)

        elif user_input == "Blob":
            image_name = input("Enter number corresponding to an image: ")

            if int(image_name) < 0 or int(image_name) > 199:
                print("Invalid number.")
            else:
                image = io.imread("Images\\" + str(image_name) + ".png")
                get_blobs(image)
        elif user_input == "Gabor":
            agr_name = input("Enter number between 0-99 corresponding to an agriculture image: ")
            non_name = input("Enter number between 100-199 corresponding to a non-agriculture image: ")

            if int(agr_name) < 0 or int(non_name) < 100 or int(agr_name) > 99 or int(non_name) > 199:
                print("Invalid number.")
            else:
                agr_image = io.imread("Images\\" + str(agr_name) + ".png", as_gray=True)
                non_image = io.imread("Images\\" + str(non_name) + ".png", as_gray=True)
                get_gabor(agr_image, non_image)
        else: 
            print("Invalid feature.")

        user_input = input("Enter desired feature or enter 'q' to quit: ")
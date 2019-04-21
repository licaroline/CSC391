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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.size'] = 9
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

def get_feats(file_name):
    img_feats = []
    img = io.imread("Images\\" + str(file_name) + ".png", as_gray=True)
        
    # Calculate LBP features
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    img_feats.append(hist.mean())
    img_feats.append(hist.var())

    # Calculate Gabor features
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    shrink = (slice(0, None, 3), slice(0, None, 3))
    img_shrunk = img_as_float(img)[shrink]
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(img, kernel, mode='wrap')
        img_feats.append(filtered.mean())
        img_feats.append(filtered.var())

    return img_feats

if __name__ == "__main__":
    # validation contains 10 agriculture 10 non
    val = [11, 15, 17, 22, 23, 36, 51, 62, 86, 99, 103, 108, 114, 118, 132, 136, 142, 168, 187, 191]
    train = list(range(200))
    train = list(set(train) - set(val))
    X = []
    y = [1] * 90 + [0] * 90
    X_val = []
    y_val = [1] * 10 + [0] * 10

    print("Getting image features...")
    print("Getting training image features.")
    for x in train:
        X.append(get_feats(x))
    print("Getting validation image features.")
    for x in val:
        X_val.append(get_feats(x))

    print("Done.")
    print("Training classifiers.")
    models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), svm.SVC(kernel='linear'), svm.LinearSVC(), svm.SVC(kernel='rbf', gamma=0.7), svm.SVC(kernel='poly', degree=3)]
        # [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
        # [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
        # [1 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 0 0 0 1]
        # [1 1 1 1 0 1 1 0 0 1 1 1 0 0 1 0 1 0 0 1]
        # [1 1 1 1 0 0 1 0 0 1 1 1 0 0 1 0 0 0 0 1]
        # [1 1 1 1 0 1 1 0 0 1 1 1 0 0 1 0 1 0 0 1]
        # [1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
    models = [clf.fit(X, y) for clf in models]

    predictions = [clf.predict(X_val) for clf in models]
    print("Validation set should be 10 agriculture (1) followed by 10 non-agriculture (0).")
    print(y_val)
    print("-----")
    for prediction in predictions:
        print(prediction)
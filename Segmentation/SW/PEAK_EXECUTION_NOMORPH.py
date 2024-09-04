import numpy as np
from numpy import linalg
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.path import Path

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def onclick(event):

    ind = event.ind
    # code to remove scatter point
    offset = line.get_offsets()
    xy = np.delete(offset, ind, axis=0)
    line.set_offsets(xy)
    plt.gca().alist = ind
    plt.draw()
    print(len(offset.data))


def onpick(event):

    if event.xdata and event.ydata > 1.0:
        x1, y1 = event.xdata, event.ydata

        if event.button == 1:
            offset = line.get_offsets()
            xy = np.append(offset, np.array([[x1], [y1]]).T, axis=0)
            line.set_offsets(xy)
            plt.gca().alist = xy
            plt.draw()
            print(len(offset.data))

    # code to remove scatter point

        if event.button == 2:
            offset = line.get_offsets()
            print(len(offset.data))
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            plt.show(block=False)

        if event.button == 3:
            fig.canvas.mpl_connect('pick_event', onclick)
    else:
        pass


def mask_ellipse(rad):
    ctr = np.floor(np.divide([20, 20], 2) + 1)
    msk = np.zeros([20, 20])
    prm = np.sqrt(2) * np.pi * np.linalg.norm(rad, ord=2)
    for theta in np.arange(0, 2 * np.pi, np.pi / prm):
        x = rad[0] * np.cos(theta)
        y = rad[1] * np.sin(theta)
        msk[np.int(ctr[0] + np.ceil(x)), np.int(ctr[1] + np.ceil(y))] = 1
        msk = ndimage.morphology.binary_fill_holes(msk) * 1
    return msk


def update(val):
    value = Qt_slider.val
    img.set_clim(value)
    plt.draw()


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)



# @@ main @@

# 1. img 불러오기
img_dir = 'C:/Users/dlatj/Anaconda3/floodmap/test'
flood_map = cv2.imread('C:/Users/dlatj/Anaconda3/floodmap/test/Image_Sample.jpeg')
img_size = np.shape(flood_map)[0] # 정사각형이니까 하나만 가져온듯
R = flood_map[:, :, 0]
G = flood_map[:, :, 1]
B = flood_map[:, :, 2]
flood_map = (R * 299. / 1000 + G * 587. / 1000 + B * 114. / 1000)/255   # grayscale 전환?


# 2. Parameters
background_param = '0.15'
X_slide = '4'
Y_slide = '4'
patch_size = '4'  # if patch size is 4, the size of the patch will be 8x8


# 3. plot 
fig, ax = plt.subplots()
fig.canvas.set_window_title('PROCESS') # plot 제목 설정
plt.subplots_adjust(bottom=0.4) # subplot 간의 간격
plt.imshow(flood_map)
axbox1 = plt.axes([0.2, 0.05, 0.6, 0.05])
axbox2 = plt.axes([0.2, 0.1, 0.6, 0.05])
axbox3 = plt.axes([0.2, 0.15, 0.6, 0.05])
axbox4 = plt.axes([0.2, 0.2, 0.6, 0.05])

text_box1 = TextBox(axbox1, 'Thres', initial=background_param)
text_box2 = TextBox(axbox2, 'X_Slide', initial=X_slide)
text_box3 = TextBox(axbox3, 'Y_Slide', initial=Y_slide)
text_box4 = TextBox(axbox4, 'Patch_size', initial=patch_size)

background_param = float(text_box1.text)
X_slide = int(text_box2.text)
Y_slide = int(text_box3.text)
patch_size = int(text_box4.text)
plt.show()



# 4. load model
model2 = load_model('Normal_CNN.h5')
peak_map = np.zeros((img_size, img_size), dtype='int64')
index_map = np.zeros((img_size, img_size), dtype='int64')
width = range(0, img_size-20, X_slide)
length = range(0, img_size-20, Y_slide)


# 5. Scanning and Peak detection
for i in width:
    for j in length:
        sub_img = flood_map[i:i + 20, j:j + 20]
        sub_img = sub_img.reshape(1, 20, 20, 1)
        idx1 = model2.predict(sub_img)
        th = 15

        if idx1[0, 1] >= 0.90 and np.sum(peak_map[i + 10 - th:i + 10 + th, j + 10 - th:j + 10 + th]) == 0:  # if its 99% true and there are no patches
            sub_img4 = sub_img.reshape(20, 20)
            A = np.where(sub_img4 == 1)

            # l1, and l2 is the mid_point in x&y direction of the morphological image
            l1 = np.int(np.floor((np.max(A[0]) + np.min(A[0])) / 2))
            l2 = np.int(np.floor((np.max(A[1]) + np.min(A[1])) / 2))
            # patch for preventing the superposition of the peak
            peak_map[l1 + i - patch_size:l1 + i + patch_size, l2 + j - patch_size:l2 + j + patch_size] = 255
            # peak index
            index_map[l1+i, l2+j] = 255
        else:
            continue
        
        
# scatter plot 전처리
xy = np.where(index_map == 255)
x = xy[1]
y = xy[0]
x = x.tolist()
y = y.tolist()

# scatter plot & interactive plotting
fig, ax = plt.subplots()
img = ax.imshow(flood_map)
fig.canvas.set_window_title('DETECTED PEAK')
line = ax.scatter(xy[1], xy[0], c='red', s=3, picker=5)
cid = fig.canvas.mpl_connect('button_press_event', onpick)

axcolor = 'lightgoldenrodyellow'
f0 = 0
delta_f = 0.01
Qt_ax = plt.axes([0.18, 0.02, 0.65, 0.03], facecolor=axcolor)
Qt_slider = Slider(Qt_ax, 'Quantization', 0.0, 1.0, valinit=f0, valstep=delta_f)
Qt_slider.on_changed(update)
plt.show()

points = line.get_offsets().data
sort_points = points[np.argsort(points[:, 0])]





#%% mytest
import tensorflow as tf

model2 = tf.keras.models.load_model('Normal_CNN.h5')
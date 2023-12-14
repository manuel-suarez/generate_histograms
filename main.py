import os
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

home_dir = os.path.expanduser('~')
base_dir = os.path.join(home_dir, 'data/cimat/oil-spill-dataset/tarso/')

sar_dir = os.path.join(base_dir, 'sar')
tiff_dir = os.path.join(base_dir, 'geo/tiff')
norm_dir = os.path.join(base_dir, 'geo/tiff_norm')

os.listdir(norm_dir)
for file in os.listdir(tiff_dir):
    img = rasterio.open(os.path.join(tiff_dir,file))
    print(f"File: {file}, max: {np.max(img.read(1))}, min: {np.min(img.read(1))}, nan: {np.count_nonzero(np.isnan(img.read(1)))}, 1e-25: {np.count_nonzero(np.abs(img.read(1))<1e-25)}")
for file in os.listdir(norm_dir):
    img = rasterio.open(os.path.join(norm_dir,file))
    print(f"File: {file}, max: {np.max(img.read(1))}, min: {np.min(img.read(1))}, nan: {np.count_nonzero(np.isnan(img.read(1)))}, 1e-25: {np.count_nonzero(np.abs(img.read(1))<1e-25)}")

imgs = [rasterio.open(os.path.join(tiff_dir,file)) for file in os.listdir(tiff_dir)]
num_files = len(imgs)

plt.imshow(imgs[0].read(1), cmap='gray')
imgs_norm = [rasterio.open(os.path.join(norm_dir,file)) for file in os.listdir(norm_dir)]

img0 = imgs[0].read(1)
img0[(np.abs(img0) < 1e-25).astype(bool)] = np.nan

cmap = matplotlib.colormaps.get_cmap('viridis')
cmap.set_bad(color='red')

plt.imshow(img0, cmap=cmap)
plt.colorbar()

img0 = imgs_norm[0].read(1)
img0[(np.abs(img0) < 1e-25).astype(bool)] = np.nan

cmap = matplotlib.colormaps.get_cmap('viridis')
cmap.set_bad(color='red')

plt.imshow(img0, cmap=cmap)
plt.colorbar()

plt.hist(imgs[0].read(1))
plt.show()

plt.hist(imgs_norm[0].read(1))
plt.title(os.path.basename(imgs_norm[0].name))
plt.show()

def show_hists(img, img_norm, index):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].set_title('Imagen original')
    axs[0].hist(img.read(1))
    axs[1].set_title('Imagen normalizada')
    axs[1].hist(img_norm.read(1))
    # Normalización max-min
    x = img.read(1)
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    axs[2].set_title('Normalización max-min')
    axs[2].hist(x_norm)
    plt.savefig(f"histogram_{index}.png")
    plt.close()

def show_hist(index):
    show_hists(imgs[index], imgs_norm[index], index)

for i in range(len(imgs)):
    show_hist(i)
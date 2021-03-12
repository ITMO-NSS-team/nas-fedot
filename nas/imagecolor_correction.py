from __future__ import print_function
import sys
import os

gdal_existed = rasterio_existed = georasters_existed = False
try:
    import gdal
    gdal_existed = True
    print('INFO: GDAL package will be used to read GeoTIFF files')
    import cv2
    import numpy as np
    from os.path import join
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import minmax_scale
    import pandas as pd
    import json
    from sklearn.model_selection import train_test_split
    import png
    from PIL import Image

except ImportError:
    try:
        import rasterio
        rasterio_existed = True
        print('INFO: rasterio package will be used to read GeoTIFF files')
    except ImportError:
        print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
        exit()

def adjust_gamma(image, gamma=4.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def gamma_convertion(image, gamma=4.0):

    # original = cv2.imread(image_name, 1)
    # cv2.imshow('original', original)
    #   # change the value here to get different result
    adjusted = adjust_gamma(image, gamma=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
    cv2.imshow("gammam image 1", adjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_images(folder_name):

    images_dict = {'blue': None, 'green': None, 'red': None}
    blue_channel = "B02.tif"
    green_channel = "B03.tif"
    red_channel = "B04.tif"
    json_file = "data.json"
    labels_name = None
    # list_of_channels = [blue_channel, green_channel, red_channel]
    for file in os.listdir(folder_name):

        if file[-7::] == blue_channel:
            images_dict['blue'] = join(folder_name, file)
        elif file[-7::] == green_channel:
            images_dict['green'] = join(folder_name, file)
        elif file[-7::] == red_channel:
            images_dict['red'] = join(folder_name, file)
        elif file[-9::] == json_file:
            labels_name = join(folder_name, file)
    return images_dict, labels_name

def combine_channels(images_dict):
    final_arr = [[], [], []]
    for ch, image in enumerate(images_dict):
        final_arr[ch] = cv2.imread(images_dict[image], 0)
    final_arr = np.array(final_arr)
    # image_res = np.resize(final_arr, (120, 120, 3))
    image_res = final_arr.transpose(2,1,0)
    return image_res


def gdal_combine_channels(images_dict):
    final_arr = [[], [], []]
    for ch, image in enumerate(images_dict):
        band_path = images_dict[image]
        band_ds = gdal.Open(band_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = raster_band.ReadAsArray()
        # band_data = normalize(band_data, norm='max', axis=1)
        # band_data = minmax_scale(band_data)
        band_data = band_data/band_data.max()
        # print(band_data.mean())

        final_arr[ch] = band_data

    final_arr = np.array(final_arr)
    # image_res = np.resize(final_arr, (120, 120, 3))
    image_res = final_arr.transpose(2,1,0)
    return image_res


def get_labels(filename):
    data = json.load(open(filename))
    # label = pd.DataFrame(data["labels"])
    labels = data['labels']
    print(labels)
    return labels


def draw_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[0]
    cv2.imshow('ss', img2)
    cv2.waitKey(0)


def preprocess_save_images():
    folder_path = "dataset/Earth/"
    save_dir = "Generated_dataset/"
    file_format = ".png"
    labels_dict = {}
    for file_name in os.listdir(folder_path):
        folder_name = folder_path + file_name
        images_dict, labels_name = get_images(folder_name)
        labels = get_labels(labels_name)
        # print(folder_name)
        image = gdal_combine_channels(images_dict)
        save_path = save_dir + file_name + file_format
        # cv2.imwrite(save_path, image)
        # png.from_array(image).save(save_path)
        # Rescale to 0-255 and convert to uint8
        rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint8)

        im = Image.fromarray(rescaled)
        im.save(save_path)
        labels_dict[file_name] = labels
    with open('labels.json', 'w') as fp:
        json.dump(labels_dict, fp)

    return labels_dict

if __name__ == "__main__":
    dict = preprocess_save_images()
    # Xarr.append(image)
    # Yarr.append(labels)
    # Xarr = np.array(Xarr)
    # Yarr = np.array(Yarr)
    # Xtrain, Xval, Ytrain, Yval = train_test_split(Xarr, Yarr, random_state=1, train_size=0.8)
    # print(Ytrain.shape)
    # train_input_data = InputData(idx=np.arange(0, len(Xtrain)), features=Xtrain, target=np.array(Ytrain),
    #                              task_type=task_type)
    # val_input_data = InputData(idx=np.arange(0, len(Xval)), features=Xval, target=np.array(Yval),
    #                            task_type=task_type)
    # test_input_data = InputData(idx=np.arange(0, len(Xtest)), features=Xtest, target=np.array(Ytest),
    #                             task_type=task_type)

    # return train_input_data, val_input_data, test_input_data



# gamma = 1
# gamma_convertion(image, gamma)



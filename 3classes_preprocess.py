import numpy as np
import cv2
import json
import os
from tqdm import tqdm
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

from matplotlib import pyplot as plt
from PIL import Image
import sys

sys.path.append("C:/Users/aliev/AppData/Local/Programs/Python/Python37/Lib/site-packages/gdal.py")
try:
    import gdal

    print('INFO: All packages are installed. GDAL will be used to read GeoTIFF files')
except ImportError:
    print('ERROR: Some of packages are not installed. Please install GDAL to read GeoTIFF files')


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
    # folder_name = "dataset/Earth/S2A_MSIL2A_20170613T101031_11_38"
    images_dict = {'blue': None, 'green': None, 'red': None}
    blue_channel = "B02.tif"
    green_channel = "B03.tif"
    red_channel = "B04.tif"
    json_file = "data.json"
    labels_name = None
    # list_of_channels = [blue_channel, green_channel, red_channel]
    for file in os.listdir(folder_name):

        if file[-7::] == blue_channel:
            images_dict['blue'] = (join(folder_name, file), 2)
        elif file[-7::] == green_channel:
            images_dict['green'] = (join(folder_name, file), 1)
        elif file[-7::] == red_channel:
            images_dict['red'] = (join(folder_name, file), 0)
        elif file[-9::] == json_file:
            labels_name = join(folder_name, file)
    return images_dict, labels_name


def gdal_combine_channels(images_dict):
    final_arr = [[], [], []]
    for tup_im in images_dict.values():
        band_path = tup_im[0]
        ch = tup_im[1]
        # band_path = images_dict[image]
        band_ds = gdal.Open(band_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = raster_band.ReadAsArray()
        # band_data = normalize(band_data, norm='max', axis=1)
        # band_data = minmax_scale(band_data)

        # band_data = (band_data - band_data.min()) / (band_data.max() - band_data.min()) * 200.0

        final_arr[ch] = band_data

    final_arr = np.array(final_arr)
    # image_res = np.resize(final_arr, (120, 120, 3))
    image_res = final_arr.transpose(1, 2, 0)
    return image_res


def get_labels(filename):
    data = json.load(open(filename))
    labels = data['labels']
    # print(labels)
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


def count_classes(class_dict, patch_class):
    if class_dict.get(patch_class):
        class_dict[patch_class] += 1
    else:
        class_dict[patch_class] = 1
    return class_dict


def preprocess_save_images():
    folder_path = "dataset/Earth/"
    save_dir = "3classes/"
    file_format = ".png"
    labels_dict = {}
    encoded_labels = {}
    all_labels = set()
    not_needed = []
    needed_labels = ['Mixed forest', 'Non-irrigated arable land']
    # for file_name in os.listdir(folder_path):
    classes_counter = dict()

    for nomer_iter, file_name in tqdm(enumerate(os.listdir(folder_path))):
        skip = False
        folder_name = folder_path + file_name
        images_dict, labels_name = get_images(folder_name)
        labels = get_labels(labels_name)

        image = gdal_combine_channels(images_dict)



        for label in labels:
            if label not in needed_labels:
                skip = True
                break
            elif len(labels) > 1:
                skip = True
                break
            elif classes_counter.get(label) is not None:
                if classes_counter.get(label) >= 20:
                    skip = True
                    break
                else:
                    continue
            else:
                print('shet')
                continue
        #
        if skip:
            continue

        labels_dict[file_name] = labels
        save_path = save_dir + file_name + file_format

        # Rescale to 0-255 and convert to uint8
        rescaled = ((image - image.min()) / (image.max() - image.min()) * 255.0).astype(np.uint8)
        im = Image.fromarray(rescaled, mode="RGB")
        im.save(save_path)


        for label in labels:
            if not skip:
                all_labels.add(label)
                encoded_number = encoded_labels.get(label)
                if not encoded_number:
                    rand_num = np.random.randint(0, 3)
                    # if rand_num not in encoded_labels.values():
                    encoded_labels[label] = rand_num
                classes_counter = count_classes(classes_counter, label)
    with open('labels.json', 'w') as fp:
        json.dump(labels_dict, fp)
    with open('encoded_labels.json', 'w') as fp:
        json.dump(encoded_labels, fp)
    with open('counted_classes.json', 'w') as cc:
        json.dump(classes_counter, cc)

    write_labels = open('all_possible_labels.txt', 'w')
    all_labels = map(lambda x: x + '\n', all_labels)
    write_labels.writelines(all_labels)
    write_labels.close()
    print(classes_counter)

    return labels_dict, encoded_labels



if __name__ == "__main__":
    p = preprocess_save_images()
    # from_images("Generated_dataset")

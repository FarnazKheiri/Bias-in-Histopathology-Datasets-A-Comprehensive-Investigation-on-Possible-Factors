import os
import pathlib
import cv2
import pandas as pd

def spilit_image_method(source_path, slide_name, images, center_label, cancer_label, filename, center_labels,
                        cancer_labels, slidenames, filenames):
    number_of_points = 5
    dim = 224
    orig_dim = 200
    img = cv2.imread(source_path)
    spilited_file_counter = 0
    for i in range(number_of_points):
        for j in range(number_of_points):
            if j == 4 and i != 4:
                crop_img = img[orig_dim * i: orig_dim * i + dim, 775: 999]
            elif i == 4 and j != 4:
                crop_img = img[775: 999, orig_dim * j: orig_dim * j + dim]
            elif i == 4 and j == 4:
                crop_img = img[775: 999, 775: 999]
            else:
                crop_img = img[orig_dim * i: orig_dim * i + dim, orig_dim * j: orig_dim * j + dim]
            images.append(crop_img)
            center_labels.append(center_label.values)
            cancer_labels.append(cancer_label.values)
            slidenames.append(slide_name)
            filenames.append(filename)
    return images, center_labels, cancer_labels, slidenames, filenames


def get_labels(root):
    # get the main path
    data_dir = pathlib.Path(root)
    # produces a sequence of file paths by recursively searching for all files in all subdirectories of data_dir.
    image_paths = list(data_dir.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    labels_file = pd.read_csv("patch_info.csv", index_col=0)
    images = []
    center_labels = []
    cancer_labels = []
    filenames = []
    slidenames = []

    for path in image_paths:
        print(path)
        slide_name = os.path.basename(os.path.normpath(path)) + '.svs'
        center_label = labels_file.loc[labels_file["slide_name"] == slide_name]["medical_center"]
        cancer_label = labels_file.loc[labels_file["slide_name"] == slide_name]["disease_type"]
        #     pdb.set_trace()
        for filename in os.listdir(path):

            try:
                images, center_labels, cancer_labels, slidenames, filenames = spilit_image_method(
                    os.path.join(path, filename), slide_name, images, center_label, cancer_label, filename,
                    center_labels, cancer_labels, slidenames, filenames)
            except:
                continue
    return images, cancer_labels, center_labels, slidenames, filenames

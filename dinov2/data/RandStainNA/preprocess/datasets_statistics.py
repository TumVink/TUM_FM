import os
import cv2
import numpy as np
import time
import argparse
import yaml
import json
import random
import copy
from skimage import color
from fitter import Fitter
import h5py

parser = argparse.ArgumentParser(description="norm&jitter dataset lab statistics")
parser.add_argument(
    "--data-dir", type=str,default='/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/TUM_patches/', metavar="DIR", help="path to dataset(ImageFolder form)"
)
parser.add_argument("--save-dir", type=str,default='/dss/dsshome1/05/ge54xof2/Foundation-Model-for-Pathology/RandStainNA/preprocess/', metavar="DIR", help="path to save dataset")
parser.add_argument(
    "--dataset-name", type=str, default="TUM_slides", metavar="DIR", help="dataset output_name"
)
parser.add_argument("--methods", type=str, default="Reinhard", help="colornorm_methods")
parser.add_argument(
    "--color-space",
    type=str,
    default="HED",
    choices=["LAB", "HED", "HSV"],
    help="dataset statistics color space",
)
parser.add_argument(
    "--random", action="store_true", default=False, help="random shuffle sample"
)
parser.add_argument(
    "--n",
    type=int,
    default=0,
    metavar="DIR",
    help="datasets statistics sample n image each class(0:all)",
)


def _parse_args():
    args = parser.parse_args()

    return args


def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)


if __name__ == "__main__":

    args = _parse_args()

    path_dataset = args.data_dir

    labL_avg_List = []
    labA_avg_List = []
    labB_avg_List = []
    labL_std_List = []
    labA_std_List = []
    labB_std_List = []

    t1 = time.time()
    i = 0
    j = 0

    for h5_path in os.listdir(path_dataset):
        h5_file = os.path.join(path_dataset, h5_path)
        #print(h5_file)
        j+=1
        if j%100 == 0:
            print(j)
        #path_class_list = os.listdir(path_class)
        # if args.random:
        #     random.shuffle(path_class_list)
        if not h5_file.endswith('.h5'):
            continue
        # read h5 imgs
        with h5py.File(h5_file, "r") as f:
            imgs = f['imgs'][:] #(500,512,512,3)
        #print(imgs.shape)
        for i in range(imgs.shape[0]):
            img = imgs[i] #in RGB order
            #print(img.shape)
            try:  # debug
                if args.color_space == "LAB":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                elif args.color_space == "HED":
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = color.rgb2hed(img)
                elif args.color_space == "HSV":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                else:
                    print("wrong color space: {}!!".format(args.color_space))
                img_avg, img_std = getavgstd(img)
            except:
                continue
                print(path_img)
            labL_avg_List.append(img_avg[0])
            labA_avg_List.append(img_avg[1])
            labB_avg_List.append(img_avg[2])
            labL_std_List.append(img_std[0])
            labA_std_List.append(img_std[1])
            labB_std_List.append(img_std[2])
            #break
        #break
    t2 = time.time()
    print(t2 - t1)
    l_avg_mean = np.mean(labL_avg_List).item()
    l_avg_std = np.std(labL_avg_List).item()
    l_std_mean = np.mean(labL_std_List).item()
    l_std_std = np.std(labL_std_List).item()
    a_avg_mean = np.mean(labA_avg_List).item()
    a_avg_std = np.std(labA_avg_List).item()
    a_std_mean = np.mean(labA_std_List).item()
    a_std_std = np.std(labA_std_List).item()
    b_avg_mean = np.mean(labB_avg_List).item()
    b_avg_std = np.std(labB_avg_List).item()
    b_std_mean = np.mean(labB_std_List).item()
    b_std_std = np.std(labB_std_List).item()

    std_avg_list = [
        labL_avg_List,
        labL_std_List,
        labA_avg_List,
        labA_std_List,
        labB_avg_List,
        labB_std_List,
    ]
    distribution = []
    for std_avg in std_avg_list:
        f = Fitter(std_avg, distributions=["norm", "laplace"])
        f.fit()
        distribution.append(list(f.get_best(method="sumsquare_error").keys())[0])

    yaml_dict_lab = {
        "random": args.random,
        "n_each_class": args.n,
        "color_space": args.color_space,
        "methods": args.methods,
        "{}".format(args.color_space[0]): {  # lab-L/hed-H
            "avg": {
                "mean": round(l_avg_mean, 3),
                "std": round(l_avg_std, 3),
                "distribution": distribution[0],
            },
            "std": {
                "mean": round(l_std_mean, 3),
                "std": round(l_std_std, 3),
                "distribution": distribution[1],
            },
        },
        "{}".format(args.color_space[1]): {  # lab-A/hed-E
            "avg": {
                "mean": round(a_avg_mean, 3),
                "std": round(a_avg_std, 3),
                "distribution": distribution[2],
            },
            "std": {
                "mean": round(a_std_mean, 3),
                "std": round(a_std_std, 3),
                "distribution": distribution[3],
            },
        },
        "{}".format(args.color_space[2]): {  # lab-B/hed-D
            "avg": {
                "mean": round(b_avg_mean, 3),
                "std": round(b_avg_std, 3),
                "distribution": distribution[4],
            },
            "std": {
                "mean": round(b_std_mean, 3),
                "std": round(b_std_std, 3),
                "distribution": distribution[5],
            },
        },
    }
    yaml_save_path = "{}/{}_{}.yaml".format(
        args.save_dir,
        args.dataset_name,
        args.color_space
        if args.dataset_name != ""
        else "dataset_{}_random{}_n{}".format(args.color_space, args.random, args.n),
    )
    with open(yaml_save_path, "w") as f:
        yaml.dump(yaml_dict_lab, f)
        print("The dataset lab statistics has been saved in {}".format(yaml_save_path))

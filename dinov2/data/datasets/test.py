# test the h5 file patches /mnt/nfs02-R6/TUM_slides/clam/patches/h2022013653t7-a-1_011418.h5
import h5py
import numpy as np
import openslide
import os
import shutil

def test_h5_file(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Print all root level object names (aka keys)
        #print(f.keys())
        #print f['coords'] attributes
        print(f['img'].shape)

def check_mag_svs(file):
    if file.endswith('.tif'):
        slide = openslide.OpenSlide(file)
        #print the properties of the slide
        print(slide.properties)

def generate_cam16(global_path='/mnt/nfs03-R6/CAMELYON16/',sub_folder='images/',slide_format='.tif'):
    import pandas as pd
    slide_list = []
    coor_list = []
    wdir = global_path + sub_folder
    files = os.listdir(wdir)
    for file in files:
        if file.endswith(slide_format):
            file_path = os.path.join(wdir, file)
            coor_path = os.path.join(global_path, 'clam_20', 'coor', file.replace(slide_format, '.h5'))
            if os.path.exists(coor_path):
                coor_list.append(coor_path)
                slide_list.append(file_path)
            else:
                print(coor_path)
                print('No coor file found for:', file)
    print('Total files:', len(slide_list))
    print('Total coor files:', len(coor_list))
    df = pd.DataFrame({'slide': slide_list, 'coor': coor_list})
    df.to_csv(os.path.join(global_path, 'clam_20','dataset.csv'), index=False)

def generate_symbolic_hest():
    #for each image ends with .tif in /mnt/nfs03-R6/HEST-1k/hest_data/wsis/,
    # generate a symbolic link to the image in /mnt/nfs03-R6/HEST_jingsong/wsis/
    import os
    import glob
    import shutil

    src_dir = '/mnt/nfs03-R6/HEST-1k/hest_data/wsis/'
    dst_dir = '/mnt/nfs03-R6/HEST_jingsong/wsis/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    files = glob.glob(os.path.join(src_dir, '*.tif'))
    for file in files:
        file_name = os.path.basename(file)
        dst_file = os.path.join(dst_dir, file_name)
        if not os.path.exists(dst_file):
            #generate symbolic link
            os.symlink(file, dst_file)
        else:
            print('File already exists:', dst_file)

def move_svs(subset_ls=['CRS_Test/slides/','CRS1/slides/']):
    global_path = '/mnt/nfs03-R6/IMP_CRC/'
    import pandas as pd
    for subset in subset_ls:
        wdir = global_path+subset
        files = os.listdir(wdir)
        for file in files:
            if file.endswith('.svs'):
                shutil.move(os.path.join(wdir, file), os.path.join(global_path, 'images', file))
                os.symlink(os.path.join(global_path, 'images', file), os.path.join(wdir, file))
                print('Moved File:', file)

def merge_dataset_csv():
    import pandas as pd
    TCGA_path = '/mnt/nfs03-R6/TCGA/clam_20/'
    CPTAC_path = '/mnt/nfs03-R6/CPTAC/clam_20/'
    Hest_path = '/mnt/nfs03-R6/HEST_jingsong/clam_20/'
    CAM17_path = '/mnt/nfs03-R6/CAMELYON17/clam_20/'
    CAM16_path = '/mnt/nfs03-R6/CAMELYON16/clam_20/'
    IMP_path = '/mnt/nfs03-R6/IMP_CRC/clam_20/'
    SLN_path = '/mnt/nfs03-R6/SLN_cli/clam_20/'
    #read all csv files
    TCGA_df = pd.read_csv(os.path.join(TCGA_path, 'dataset.csv'))
    CPTAC_df = pd.read_csv(os.path.join(CPTAC_path, 'dataset.csv'))
    Hest_df = pd.read_csv(os.path.join(Hest_path, 'dataset.csv'))
    CAM17_df = pd.read_csv(os.path.join(CAM17_path, 'dataset.csv'))
    CAM16_df = pd.read_csv(os.path.join(CAM16_path, 'dataset.csv'))
    IMP_df = pd.read_csv(os.path.join(IMP_path, 'dataset.csv'))
    SLN_df = pd.read_csv(os.path.join(SLN_path, 'dataset.csv'))
    #merge all dataframes
    df = pd.concat([TCGA_df, CPTAC_df, Hest_df, CAM17_df, CAM16_df, IMP_df, SLN_df])
    #save into "public_dataset.csv"
    global_path = '/home/ge24juj/dino-tum/dinov2/data/datasets/'
    df.to_csv(os.path.join(global_path, 'public_dataset.csv'), index=False)


#check_mag_svs('/mnt/nfs03-R6/HEST-1k/hest_data/wsis/TENX120.tif')
# generate_cam16(global_path='/mnt/nfs03-R6/HEST_jingsong/',
#                 sub_folder='wsis/',
#                 slide_format='.tif')
#move_svs()
merge_dataset_csv()
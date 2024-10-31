# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from itertools import repeat
import h5py
import concurrent.futures
#from select_slides import select_slides
import psutil
import subprocess
import pipes
import shutil
import cv2
import sys


def check_local_storage(dir='/mnt/data/TUM_Slides_Patches/',threshold=50):
    disk_usage = psutil.disk_usage(dir)
    if disk_usage.free / (2 ** 30) < threshold:
        print("Patching stop! Free space is lower than threshold.")
        print("Free space is only "+str(disk_usage.free / (2 ** 30))+ 'GB now.')
        return False
    else:
        return True

def download_slide_local(local_folder = '/media/research/data_slow/TUM_slides/',slide_dir=None):
    """
    download the slides on Qnap (/media/scanner/Scans01/slide_id...) into local storage (/mnt/data/TUM_Slides_Patches/slide_buffer/slide_id)
    """
    #local_folder = '/mnt/data/TUM_Slides_Patches/slide_buffer/'
    slide_id = os.path.basename(slide_dir)
    slide_local_dir = os.path.join(local_folder,slide_id)
    print('copying from Qnap to local')
    start_time = time.time()
    if ('data_slow' in local_folder) and (os.path.exists(slide_local_dir)):
        print('slide exists in local dir, skip copying')
        return slide_local_dir
    shutil.copy(slide_dir, slide_local_dir)
    copy_time = time.time() - start_time
    #print('copying slides took '+str(copy_time)+' seconds')
    print(f'copying slides took {copy_time:0.3f} seconds')

    return slide_local_dir

def mark_processed_slide(df):
    """
    mark already existed slide as 'processed'
    """
    DST_patches_dir = "/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/TUM_patches/"
    cmd_lrz = "ls " + DST_patches_dir + " |grep '.h5'"
    stream = os.popen('ssh ge54xof2@login.ai.lrz.de ' + cmd_lrz)
    output = stream.read()
    h5_list_lrz = output.split()
    # print(len(usage))

    mask = df['status'] == "tbs"
    process_stack = df[mask]
    total = len(process_stack)

    patch_save_dir = "/mnt/data/TUM_Slides_Patches/patches"

    for i in range(total):
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        slide_id = os.path.basename(slide).split('.')[0]
        file_dir = slide_id + '.h5'

        if file_dir in h5_list_lrz or os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            df.loc[idx, 'status'] = 'tb trained'
            df.loc[idx,'process'] = 0

    print("finish marking processed slide")
    return df


def check_lrz_file_exist(dir="/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/TUM_patches/",file_dir='.h5'):
    """
    check if a patches file exits in lrz
    """
    path = dir+file_dir
    status = subprocess.call(
        ['ssh', 'ge54xof2@login.ai.lrz.de', 'test -f {}'.format(pipes.quote(path))])
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception('SSH failed')

def check_patches_exist_in_lrz(lrz_ls,file_dir):
    """
    check if a patches file exists in lrz dir.
    This is a fast version of previous check_lrz_file_exist() by checking patches files with only accessing lrz once, but not everytime.
    """
    if file_dir in lrz_ls:
        return True
    else:
        return False


def check_lrz_storage(lrz_dir="/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/", threshold=100):
    quota = 10000 #GB, this is the number of storage that LRZ assigned
    cmd_lrz = "du -s "+lrz_dir
    stream = os.popen('ssh ge54xof2@login.ai.lrz.de '+cmd_lrz)
    output = stream.read()
    usage = output.split()[0] # G or M or
    usage = usage.split('G')[0]
    usage_GB = int(usage)/(2**20) #usage in GB
    free_GB = quota - usage_GB

    if free_GB < threshold:
        print("Stop transferring!")
        print(str(round(usage_GB,2)) + "G is used, while " + str(round(free_GB,2)) + "G is free")
        return False
    else:
        return True

def select_random_pairs(data, num_pairs=1000):
    indices = np.random.choice(len(data['coords']), num_pairs, replace=False)
    selected_coors = data['coords'][indices]
    return selected_coors

def write_to_h5(file_path, selected_coors, selected_imgs):
    with h5py.File(file_path, 'w') as h5file:
        # Overwrite the existing 'coors' and 'imgs' keys
        del h5file['coords']

        # Create new datasets for the selected pairs
        h5file.create_dataset('coords', data=selected_coors)
        h5file.create_dataset('imgs', data=selected_imgs)
    print('finish writing')

def create_img_h5(file_path, selected_coors, selected_imgs):
    with h5py.File(file_path, 'a') as h5file:
        # Overwrite the existing 'coors' and 'imgs' keys
        # del h5file['coords']

        # Create new datasets for the selected pairs
        #h5file.create_dataset('coords', data=selected_coors,compression="gzip")
        h5file.create_dataset('imgs', data=selected_imgs,compression='lzf')
    print('finish writing')


def read_imgs_parallel(WSI_object,coords,patch_level,patch_size,img_patch_path,**kwargs):
    #print(patch_level)
    num_patch = len(coords)
    print(num_patch)
    # coords_40 = coords[:2, :] # 0:300
    # coords_20 = coords[:2, :] # 300:400
    # coords_10 = coords[:2, :] # 400:500
    [coords_40,coords_20,coords_10]=np.split(coords,[int(num_patch*0.6),int(num_patch*0.8)])
    with concurrent.futures.ThreadPoolExecutor(16) as executor:
        results_40 = list(executor.map(WSI_object.wsi.read_region, coords_40,repeat(patch_level),repeat((patch_size, patch_size))))

    with concurrent.futures.ThreadPoolExecutor(16) as executor:
        results_20 = list(
            executor.map(WSI_object.wsi.read_region, coords_20, repeat(patch_level+1), repeat((patch_size, patch_size))))
    #print(results_20)
    with concurrent.futures.ThreadPoolExecutor(16) as executor:
        results_10 = list(
            executor.map(WSI_object.wsi.read_region, coords_10, repeat(patch_level+2), repeat((patch_size, patch_size))))
    #print(results_10)

    results_40.extend(results_20)
    results_40.extend(results_10)
    #print(results_40)
    imgs = np.array([np.array(result.convert('RGB')) for result in results_40 if result is not None])

    #create slide_id/ folder if not exists
    # patch_dir_path = img_patch_path.split('.h5')[0]
    # if os.path.isdir(patch_dir_path): #/mnt/nfs03-R6/CAMELYON16/clam/patches/test_020
    #     pass
    # else:
    #     os.mkdir(patch_dir_path)
    #compress_level = 5 #1-> 256MB,3->251

    # i = 0
    # t1 = time.time()
    # for result in results_40:
    #     if result is not None:
    #         img_path = os.path.join(patch_dir_path,str(i)+".png")
    #         result.convert('RGB').save(img_path, compress_level=compress_level)
    #         i +=1
    #
    # t2 = time.time()
    # print(f"Total time for PIL: {t2 - t1}s ")
    #
    #
    # patch_dir_path = img_patch_path.split('.h5')[0]+'_cv2'
    # if os.path.isdir(patch_dir_path): #/mnt/nfs03-R6/CAMELYON16/clam/patches/test_020
    #     pass
    # else:
    #     os.mkdir(patch_dir_path)

    # t1 = time.time()
    # compression_level = [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    # i = 0
    # for result in results_40:
    #     if result is not None:
    #         image_array = np.array(result)
    #         image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    #         img_path = os.path.join(patch_dir_path, str(i) + ".png")
    #         cv2.imwrite(img_path, image_array, compression_level)
    #         i +=1

    #t2 = time.time()
    # print(f"Total time for cv2: {t2 - t1}s ")

    return imgs
    #WSI_object.wsi.patch_PIL = WSI_object.wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object,num_selected_patches, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs) #h5 file
    start_time = time.time()

    if file_path:
        # Select K patches
        with h5py.File(file_path, 'r+') as h5file:
            coords = h5file['coords'][:]
        # todo: record the patch_level as well as selected or not
            if len(coords) >= args.num_selected_patches:
                selected_coors = select_random_pairs({'coords': coords,},
                                                                    num_selected_patches_40)
                select_imgs = read_imgs_parallel(WSI_object,selected_coors,patch_level,patch_size,**kwargs)
                write_to_h5(file_path, selected_coors, select_imgs)
            else:
                # Remove the h5 file if it contains fewer than 1000 'coors' and 'imgs'
                print(f"Removing {file_path} as it contains less than 1000 'coors' and 'imgs'")
                #os.remove(file_path)
                return None, 0


        ### Stop Patch Timer
        patch_time_elapsed = time.time() - start_time
        return file_path, patch_time_elapsed
    else:
        return None, 0


def fp_patching(WSI_object,num_selected_patches,img_path,patch_level,patch_size, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path,n_patches = WSI_object.process_contours(**kwargs) #h5 file
    print('Coor h5 saved!')

    if file_path:
        patching_status = extract_imgs_from_coor(WSI_object, num_selected_patches, file_path,img_path,patch_level,patch_size, **kwargs)

    patch_time_elapsed = time.time() - start_time

    return n_patches,file_path, patch_time_elapsed



def extract_imgs_from_coor(WSI_object,num_selected_patches,coor_path,img_patch_path,patch_level,patch_size, **kwargs):
    ### Start extract Timer

    with h5py.File(coor_path, 'r+') as h5file:
        coords = h5file['coords'][:]
    # todo: record the patch_level as well as selected or not
        if len(coords) >= args.num_selected_patches:
            selected_coors = select_random_pairs({'coords': coords,},
                                                                num_selected_patches)
            patching_status = num_selected_patches
        else:
            selected_coors = coords
            patching_status = len(coords)
        select_imgs = read_imgs_parallel(WSI_object,selected_coors,patch_level,patch_size,img_patch_path,**kwargs)
        create_img_h5(img_patch_path, selected_coors, select_imgs)
    print('Extract images and saved!')

    return patching_status

def extract_imgs_as_h5_from_coor(WSI_object,num_selected_patches,coor_path,img_path,patch_level,patch_size, **kwargs):
    ### Start extract Timer

    with h5py.File(coor_path, 'r+') as h5file:
        coords = h5file['coords'][:]
    # todo: record the patch_level as well as selected or not
        if len(coords) >= args.num_selected_patches:
            selected_coors = select_random_pairs({'coords': coords,},
                                                                num_selected_patches)
            patching_status = num_selected_patches
        else:
            selected_coors = coords
            patching_status = len(coords)

        select_imgs = read_imgs_parallel(WSI_object,selected_coors,patch_level,patch_size,**kwargs)
        create_img_h5(img_path, selected_coors, select_imgs)
    print('Extract images and saved!')

    return patching_status


def patch_transfer_single(i,process_stack,df,save_dir='/mnt/data/TUM_Slides_Patches/',
                      patch_size=512, step_size=512, custom_downsample=1,
                      seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                                  'keep_ids': 'none', 'exclude_ids': 'none'},
                      filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8},
                      vis_params={'vis_level': -1, 'line_thickness': 500},
                      patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                      patch_level=0,
                      use_default_params=False,
                      seg=True, save_mask=True,save_h5=True,
                      stitch=True,
                      patch=True, auto_skip=True, process_list=None,num_selected_patches=500,coor_dir="/mnt/data/TUM_Slides_Patches/coor_h5/",local_patch=True,local_slide_dir=None,repatch=False):


    #df.to_csv(process_list, index=False)
    idx = process_stack.index[i]
    slide = process_stack.loc[idx, 'slide_id']
    print(slide)
    print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
    print('processing {}'.format(slide))

    df.loc[idx, 'process'] = 0
    slide_id = os.path.basename(slide).split('.')[0]
    print(slide_id)  # 037b8f47-1776-5d98-8684-12d56aca8be9_114046


    if slide_id in broken_ls:
        print('This file is broken')
        #os.remove(slide)
        return None,0,df

    if save_h5:
    # check local storage and lrz_storage
        while (not check_local_storage(save_dir)):
            time.sleep(1)

    # Inialize WSI
    # full_path = os.path.join(source, slide)
    if local_slide_dir:
        WSI_object = WholeSlideImage(local_slide_dir)
    else:
        WSI_object = WholeSlideImage(slide)

    if use_default_params:
        current_vis_params = vis_params.copy()
        current_filter_params = filter_params.copy()
        current_seg_params = seg_params.copy()
        current_patch_params = patch_params.copy()

    else:
        current_vis_params = {}
        current_filter_params = {}
        current_seg_params = {}
        current_patch_params = {}

        for key in vis_params.keys():
            if legacy_support and key == 'vis_level':
                df.loc[idx, key] = -1
            current_vis_params.update({key: df.loc[idx, key]})

        for key in filter_params.keys():
            if legacy_support and key == 'a_t':
                old_area = df.loc[idx, 'a']
                seg_level = df.loc[idx, 'seg_level']
                scale = WSI_object.level_downsamples[seg_level]
                adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                current_filter_params.update({key: adjusted_area})
                df.loc[idx, key] = adjusted_area
            current_filter_params.update({key: df.loc[idx, key]})

        for key in seg_params.keys():
            if legacy_support and key == 'seg_level':
                df.loc[idx, key] = -1
            current_seg_params.update({key: df.loc[idx, key]})

        for key in patch_params.keys():
            current_patch_params.update({key: df.loc[idx, key]})

    if current_vis_params['vis_level'] < 0:
        if len(WSI_object.level_dim) == 1:
            current_vis_params['vis_level'] = 0

        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            current_vis_params['vis_level'] = best_level

    if current_seg_params['seg_level'] < 0:
        if len(WSI_object.level_dim) == 1:
            current_seg_params['seg_level'] = 0

        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            current_seg_params['seg_level'] = best_level

    keep_ids = str(current_seg_params['keep_ids'])
    if keep_ids != 'none' and len(keep_ids) > 0:
        str_ids = current_seg_params['keep_ids']
        current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
    else:
        current_seg_params['keep_ids'] = []

    exclude_ids = str(current_seg_params['exclude_ids'])
    if exclude_ids != 'none' and len(exclude_ids) > 0:
        str_ids = current_seg_params['exclude_ids']
        current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
    else:
        current_seg_params['exclude_ids'] = []

    w, h = WSI_object.level_dim[current_seg_params['seg_level']]
    if w * h > 1e8:
        print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
        df.loc[idx, 'status'] = 'failed_seg'
        df.to_csv(process_list, index=False)
        #os.remove(slide)
        return None,0,df

    df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
    df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


    extract_time_elapsed = -1
    coor_h5_path = os.path.join(coor_dir, slide_id+ '.h5')
    print(coor_h5_path)
    if os.path.isfile(coor_h5_path): # already exists such coord h5 files
        #print()
        #directly exract images from h5 files
        print('coord file already exists.')
        img_patch_path = os.path.join(patch_save_dir, slide_id+ '.h5')
        start_time = time.time()
        patching_status = extract_imgs_from_coor(WSI_object=WSI_object,num_selected_patches=num_selected_patches,
                                                 coor_path=coor_h5_path,img_patch_path = img_patch_path,patch_level=patch_level,patch_size=patch_size,**current_patch_params, )
        extract_time_elapsed = time.time() - start_time
        df.loc[idx, 'status'] = 'tb trained'
        #df.loc[idx, 'process'] = patching_status
        df.to_csv(process_list, index=False)
        #os.remove(slide)
        return img_patch_path,extract_time_elapsed,df

    else:
        seg_time_elapsed = -1
        #print(seg)
        if seg:
            try:
                WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
                print("segmentation took {} seconds".format(seg_time_elapsed))
            except:
                df.loc[idx, 'status'] = 'failed_seg'
                df.to_csv(process_list, index=False)
                return None, 0, df

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            img_patch_path = os.path.join(patch_save_dir, slide_id+ '.h5')
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': coor_dir})
            # if save_h5:
            #     h5_file_path, patch_time_elapsed = patching(WSI_object=WSI_object,num_selected_patches=num_selected_patches, **current_patch_params, )
            # else:
            #     h5_file_path = None
            #     patching_status,patch_time_elapsed = patching_no_save(WSI_object=WSI_object,num_selected_patches=num_selected_patches, **current_patch_params, )

            patching_status,h5_file_path, patch_time_elapsed = fp_patching(WSI_object=WSI_object, num_selected_patches=num_selected_patches,img_path = img_patch_path, **current_patch_params, )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print("segmentation + patching took {} seconds".format(seg_time_elapsed+patch_time_elapsed))
        # print("patching took {} seconds".format(patch_time_elapsed))
        # print("stitching took {} seconds".format(stitch_time_elapsed))
        # if save_h5:
        #     if h5_file_path is None: #when no contours
        #         df.loc[idx, 'status'] = 'skipped'
        #     else:
        #         df.loc[idx, 'status'] = 'processed'
        #         df.loc[idx, 'process'] = patching_status
        # else:
        if h5_file_path is None: #when no contours
            df.loc[idx, 'status'] = 'tb trained'
        else:
            if patching_status == 0:# no countours
                df.loc[idx, 'status'] = 'tb trained'
                df.loc[idx, 'process'] = patching_status
            else:
                df.loc[idx, 'process'] = patching_status
                df.loc[idx, 'status'] = 'tb trained' #before patching -> tbp, after patching but not saved -> tbs, patching and saved -> already_exist
        df.to_csv(process_list, index=False)
        #if 'data_slow' not in local_slide_dir:
        #print('to remove the slide '+slide)
        #os.remove(slide)

        return h5_file_path,seg_time_elapsed+patch_time_elapsed,df


parser = argparse.ArgumentParser(description='seg and patch parallel ')
parser.add_argument('--csv_file', type=str,default='',
                    help='path to folder containing raw wsi image files')
parser.add_argument('--save_patch', default=True,
                    help='whether save the patch or not')
parser.add_argument('--patch_locally', default=False,
                    help='whether patch the slides locally or on Qnap')
# parser.add_argument('--save_coor', default=False,
#                     help='whether save the coor_h5 or not')
parser.add_argument('--coor_dir', default="",
                    help='where the coor_h5 files are saved')
parser.add_argument('--save_mask', default=False,
                    help='whether save mask or not')
parser.add_argument('--step_size', type=int, default=512,
                    help='step_size')
parser.add_argument('--patch_size', type=int, default=512,
                    help='patch_size')
parser.add_argument('--num_selected_patches', type=int, default=500,
                    help='num of each selected slides')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True)
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type=str,default='/mnt/nfs02-R6/TUM_slides/',
                    help='directory to save processed data')
parser.add_argument('--preset', default='/home/ge54xof/Foundation-Model-for-Pathology/data/presets/bwh_biopsy.csv', type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
                    help='downsample level at which to patch')
parser.add_argument('--custom_downsample', type=int, choices=[1, 2], default=1,
                    help='custom downscale when native downsample is not available (only tested w/ 2x downscale)')
parser.add_argument('--process_list', type=str, default="process_list2.csv",
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir,'clam','patches')
    mask_save_dir = os.path.join(args.save_dir,'clam', 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'clam','stitches')
    coor_dir = '/mnt/nfs01-R0/coor_h5' #os.path.join(args.save_dir,'clam', 'coor')

    if args.process_list:
        process_list = os.path.join(args.save_dir,'clam', args.process_list)

    else:
        process_list = None


    directories = {'save_dir': args.save_dir,
                   'coor_dir':coor_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': True,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 10, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(args.preset)
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    # define slides list
    if 'CAMELYON' in args.save_dir:
        source = args.save_dir+'images/'
        slides = sorted(os.listdir(source))
        slides = [os.path.join(source, slide) for slide in slides if os.path.isfile(os.path.join(source, slide)) and slide.endswith('.tif')]

    # define slides list for TCGA slides: /TCGA/slide_id/slide_name.svs
    elif 'TUM' in args.save_dir:
        slides = sorted(os.listdir(args.save_dir))
        slides = [os.path.join(args.save_dir, slide) for slide in slides if os.path.isfile(os.path.join(args.save_dir, slide)) and slide.endswith('.svs')]
        #print(len(slides))
    else:
        source = args.save_dir
        slide_folder_list = sorted(os.listdir(source))
        slides = []
        for slide_folder in slide_folder_list:
            if os.path.isfile(os.path.join(source, slide_folder)): #only consider folders
                continue
            sub_file_list = os.listdir(os.path.join(source, slide_folder))
            for file in sub_file_list:
                #print(file)
                if file.endswith('svs'):
                    print('yes')
                    slide_path = os.path.join(source,slide_folder, file)
                    slides.append(slide_path)
    #slides = [os.path.join(source, slide) for slide in slides if os.listdir(source) and slide.endswith('.tif')]


    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, save_patches=True)

    else:
        #df = pd.read_csv(process_list)
        if not os.path.exists(process_list):
            df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, save_patches=True)
        else:
            df = pd.read_csv(process_list)

    #df = mark_processed_slide(df)

    # mask = df['status'] == "tbp"
    # process_stack = df[mask]
    process_stack = df
    total = len(process_stack)
    print('total number of slides: {}'.format(total))
    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.
    broken_ls = ['h2022012526t1-c-1_195902']
    total_time = 0.0
    df.to_csv(process_list, index=False)


    # interate every slide
    for i in range(total):
        repatch = False
        #print(args.save_patch)
        idx = process_stack.index[i]
        if process_stack.loc[idx, 'status'] == 'trained':
            slide = process_stack.loc[idx, 'slide_id']
            #print(slide+" already trained, skipped")
            continue
        elif process_stack.loc[idx, 'status'] == 'tb trained' or process_stack.loc[idx, 'status'] == 'already_exist':
            slide = process_stack.loc[idx, 'slide_id']
            #print(slide+" was patched and waits for training, skipped")
            continue
        elif process_stack.loc[idx, 'status'] == 'failed_seg':
            slide = process_stack.loc[idx, 'slide_id']
            #print(slide + " too huge or failed with segmentation, skipped")
            continue
        elif process_stack.loc[idx, 'status'] == 'skipped' or (process_stack.loc[idx, 'process'] >1 and process_stack.loc[idx, 'process'] <args.num_selected_patches):
            repatch = True
            slide = process_stack.loc[idx, 'slide_id']
            print(slide + " needs to be repatched")
            #continue

        if args.patch_locally:
            slide = process_stack.loc[idx, 'slide_id']  # /media/scanner/Scans01/...
            if os.path.exists(os.path.join('/media/research/data_slow/TUM_slides/',os.path.basename(slide))): #This slide already in local
                local_slide_dir = os.path.join('/media/research/data_slow/TUM_slides/',os.path.basename(slide))
            else:
                if check_local_storage('/media/research/data_slow/TUM_slides/',5000): #5T free space
                    local_folder = '/media/research/data_slow/TUM_slides/'
                else:
                    local_folder = '/mnt/data/TUM_Slides_Patches/slide_buffer/'
            #local_folder = '/media/research/data_slow/TUM_slides/'

                local_slide_dir = download_slide_local(local_folder,slide)
        else:
            local_slide_dir = None
        try:
            h5_file_path,t,df = patch_transfer_single(i,process_stack,df=df, save_dir=args.save_dir,
                          patch_size=args.patch_size, step_size=args.step_size, custom_downsample=1,
                          seg_params=seg_params,
                          filter_params=filter_params,
                          vis_params=vis_params,
                          patch_params=patch_params,
                          seg=args.seg, use_default_params=False, save_mask=args.save_mask,save_h5=args.save_patch,
                          stitch=args.stitch,patch_level=args.patch_level,
                          patch=args.patch, auto_skip=args.no_auto_skip, process_list=process_list,num_selected_patches=args.num_selected_patches,coor_dir=coor_dir,local_patch=args.patch_locally,local_slide_dir=local_slide_dir,repatch=repatch)
            total_time +=t
            slide = process_stack.loc[idx, 'slide_id']
            os.remove(slide)
            print(slide + " is removed")
            df.to_csv(process_list, index=False)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(130)
        except Exception as e:
            print(e)
            slide = process_stack.loc[idx, 'slide_id']
            os.remove(slide)
            print(slide + " is removed")
            df.loc[idx, 'status'] = 'skipped'
            df.to_csv(process_list, index=False)
            continue


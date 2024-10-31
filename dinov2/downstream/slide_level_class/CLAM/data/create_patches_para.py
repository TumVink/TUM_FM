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
import psutil
import subprocess
import pipes
# def transfer_globus_batch(batch_size=50,lrz_ep=,dlb_ep=):
#     '''
#     func to transfer batches (50) of patches to LRZ
#     :return:
#     '''


def check_local_storage(dir='/mnt/data/TUM_Slides_Patches/',threshold=500):
    disk_usage = psutil.disk_usage(dir)
    if disk_usage.free / (2 ** 30) < threshold:
        print("Patching stop! Free space is lower than threshold.")
        print("Free space is only "+str(disk_usage.free / (2 ** 30))+ 'GB now.')
        return False
    else:
        return True

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

    mask = df['status'] == "processed"
    process_stack = df[mask]
    total = len(process_stack)

    patch_save_dir = "/mnt/data/TUM_Slides_Patches/patches"

    for i in range(total):
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        slide_id = os.path.basename(slide).split('.')[0]
        file_dir = slide_id + '.h5'

        if file_dir in h5_list_lrz or os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            df.loc[idx, 'status'] = 'already_exist'
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
    with h5py.File(file_path, 'a') as h5file:
        # Overwrite the existing 'coors' and 'imgs' keys
        #del h5file['coords']

        # Create new datasets for the selected pairs
        h5file.create_dataset('coords', data=selected_coors)
        h5file.create_dataset('imgs', data=selected_imgs)
    print('finish writing')

def read_imgs_parallel(WSI_object,coords,patch_level,patch_size,**kwargs):
    # print(patch_level)
    # with concurrent.futures.ThreadPoolExecutor(8) as executor:
    #     results = list(executor.map(WSI_object.wsi.read_region, coords,repeat(patch_level),repeat((patch_size, patch_size))))
    # imgs = np.array([result.convert('RGB') for result in results if result is not None])

    with concurrent.futures.ThreadPoolExecutor(16) as executor:
        results = list(executor.map(WSI_object.wsi.read_region, coords,repeat(patch_level),repeat((patch_size, patch_size))))
    imgs = np.array([result.convert('RGB') for result in results if result is not None])
    # print(results_40)
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
                                                                    num_selected_patches)
                select_imgs = read_imgs_parallel(WSI_object,selected_coors,**kwargs)
                write_to_h5(file_path, selected_coors, select_imgs)
            else:
                # Remove the h5 file if it contains fewer than 1000 'coors' and 'imgs'
                print(f"Removing {file_path} as it contains less than 1000 'coors' and 'imgs'")
                os.remove(file_path)
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
    # print('patch_level:' + str(patch_level))
    # print('patch_size:' + str(patch_size))
    file_path,n_patches = WSI_object.process_contours(patch_level=patch_level,patch_size=patch_size,**kwargs) #h5 file

    if file_path:

        patching_status = extract_imgs_from_coor(WSI_object, num_selected_patches, file_path,img_path,patch_level,patch_size, **kwargs)
    patch_time_elapsed = time.time() - start_time
    # if n_patches > num_selected_patches:
    #     n_patches = num_selected_patches

    return n_patches,file_path, patch_time_elapsed



def extract_imgs_from_coor(WSI_object,num_selected_patches,coor_path,img_path,patch_level,patch_size, **kwargs):
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
        write_to_h5(img_path, selected_coors, select_imgs)

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
                      patch=True, auto_skip=True, process_list=None,num_selected_patches=500,coor_dir="/mnt/data/TUM_Slides_Patches/coor_h5/"):


    df.to_csv(process_list, index=False)
    idx = process_stack.index[i]
    slide = process_stack.loc[idx, 'slide_id']
    print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
    print('processing {}'.format(slide))

    df.loc[idx, 'process'] = 0
    slide_id = os.path.basename(slide).split('.')[0]
    print(slide_id)  # 037b8f47-1776-5d98-8684-12d56aca8be9_114046

    #access the lrz target dir

    # DST_patches_dir = "/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/TUM_patches/"
    #
    # cmd_lrz = "ls " + DST_patches_dir + " |grep '.h5'"
    # stream = os.popen('ssh ge54xof2@login.ai.lrz.de ' + cmd_lrz)
    # output = stream.read()
    # h5_list_lrz = output.split()
    # # print(len(usage))
    # file_dir = slide_id+'.h5'

    if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
        #print(os.path.join(patch_save_dir, slide_id + '.h5'))
        print('{} already exist in dlb location, skipped'.format(slide_id))
        df.loc[idx, 'status'] = 'already_exist'
        return None,0,df
    # elif check_patches_exist_in_lrz(h5_list_lrz,file_dir):
    #     print('{} already exist in lrz location, skipped'.format(slide_id))
    #     df.loc[idx, 'status'] = 'already_exist'
    #     return None,0
    broken_ls = ['patient_103_node_1']
    if slide_id in broken_ls:
        print('This file is broken')
        df.loc[idx, 'status'] = 'skip'
        return None,0,df

    if save_h5:
    # check local storage and lrz_storage
        while (not check_local_storage(save_dir)) or (not check_lrz_storage()):
            time.sleep(1)

    # Inialize WSI
    # full_path = os.path.join(source, slide)
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
        return None,0,df

    df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
    df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


    extract_time_elapsed = -1
    coor_h5_path = os.path.join(coor_dir, slide_id+'.h5')
    #print(coor_h5_path)
    if os.path.isfile(coor_h5_path): # already exists such coord h5 files
        print('{} coors exists.'.format(slide_id))
        #directly exract images from h5 files
        img_patch_path = os.path.join(save_dir, slide_id+ '.h5')
        start_time = time.time()
        patching_status = extract_imgs_from_coor(WSI_object=WSI_object,num_selected_patches=num_selected_patches,
                                                 coor_path=coor_h5_path,img_path = img_patch_path,patch_level=patch_level,patch_size=patch_size,**current_patch_params, )
        extract_time_elapsed = time.time() - start_time
        df.loc[idx, 'status'] = 'processed'
        df.loc[idx, 'process'] = patching_status
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        return img_patch_path,extract_time_elapsed,df

    else:
        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
            print("segmentation  took {} seconds".format(seg_time_elapsed))

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            img_patch_path = os.path.join(patch_save_dir, slide_id + '.h5')
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
            df.loc[idx, 'status'] = 'skipped'
        else:
            if patching_status == 0:# no countours
                df.loc[idx, 'status'] = 'skipped'
                df.loc[idx, 'process'] = patching_status
            else:
                df.loc[idx, 'process'] = patching_status
                df.loc[idx, 'status'] = 'tbs' #before patching -> tbp, after patching but not saved -> tbs, patching and saved -> already_exist
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

        return h5_file_path,seg_time_elapsed+patch_time_elapsed,df


parser = argparse.ArgumentParser(description='seg and patch parallel ')
parser.add_argument('--csv_file', type=str,default='',
                    help='path to folder containing raw wsi image files')
parser.add_argument('--save_patch', default=False,
                    help='whether save the patch or not')
# parser.add_argument('--save_coor', default=False,
#                     help='whether save the coor_h5 or not')
parser.add_argument('--coor_dir', default="/mnt/nfs03-R6/SLN_cli/clam_all_patches/coor/",
                    help='where the coor_h5 files are saved')
parser.add_argument('--save_mask', default=False,
                    help='whether save mask or not')
parser.add_argument('--step_size', type=int, default=512,
                    help='step_size')
parser.add_argument('--patch_size', type=int, default=512,
                    help='patch_size')
parser.add_argument('--num_selected_patches', type=int, default=9999999,
                    help='num of each selected slides')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True,action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type=str,default='/mnt/nfs03-R6/CAMELYON17/',
                    help='directory to save processed data')
parser.add_argument('--preset', default='/home/ge54xof/Foundation-Model-for-Pathology/data/presets/bwh_biopsy.csv', type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
                    help='downsample level at which to patch')
parser.add_argument('--custom_downsample', type=int, choices=[1, 2], default=1,
                    help='custom downscale when native downsample is not available (only tested w/ 2x downscale)')
parser.add_argument('--process_list', type=str, default="",
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'clam_20','patches')
    mask_save_dir = os.path.join(args.save_dir,'clam_20', 'masks')
    stitch_save_dir = os.path.join(args.save_dir,'clam_20', 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir,'clam_20', args.process_list)

    else:
        process_list = None

    # print('csv_file: ', args.csv_file)
    # print('patch_save_dir: ', patch_save_dir)
    # print('mask_save_dir: ', mask_save_dir)
    # print('stitch_save_dir: ', stitch_save_dir)

    directories = {'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
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
    # slides_ls = select_slides()
    # slides = ["/media/scanner/Scans"+ slide.replace('\\', '/') for slide in slides_ls if slide.startswith('0')]
    # slides = sorted(slides)
    if 'SLN' in args.save_dir:
        slides = sorted(os.listdir(args.save_dir+'SLN-Breast/'))
        #print(slides)
        slides = [os.path.join(args.save_dir+'SLN-Breast/', slide) for slide in slides if
                  os.path.isfile(os.path.join(args.save_dir+'SLN-Breast/', slide)) and slide.endswith('.svs')]
    elif 'CAM' in args.save_dir:
        slides = sorted(os.listdir(args.save_dir+'images/'))
        #print(slides)
        slides = [os.path.join(args.save_dir+'images/', slide) for slide in slides if
                  os.path.isfile(os.path.join(args.save_dir+'images/', slide)) and slide.endswith('.tif')]
    #print(slides)


    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, save_patches=True)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params, save_patches=True)

    #df = mark_processed_slide(df)

    mask = df['status'] == "tbp"
    process_stack = df[mask]
    total = len(process_stack)
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
    broken_ls = ['patient_103_node_0']
    total_time = 0.0

    # interate every slide
    for i in range(total):
    #for i in range(int(total / 5),int(total *2/ 5)):
    #for i in range(int(total / 5)):
    # for i in range(int(total / 5)):

        #print(args.save_patch)
        if args.save_patch:
            while not check_local_storage():
                time.sleep(1)
        # try:
        start_time = time.time()
        h5_file_path,t,df = patch_transfer_single(i,process_stack,df=df, save_dir=patch_save_dir,
                      patch_size=args.patch_size, step_size=args.step_size, custom_downsample=1,
                      seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                                  'keep_ids': 'none', 'exclude_ids': 'none'},
                      filter_params={'a_t': 10, 'a_h': 16, 'max_n_holes': 8},
                      vis_params={'vis_level': -1, 'line_thickness': 500},
                      patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                      seg=args.seg, use_default_params=False, save_mask=args.save_mask,save_h5=args.save_patch,
                      stitch=args.stitch,patch_level=args.patch_level,
                      patch=args.patch, auto_skip=args.no_auto_skip, process_list=process_list,num_selected_patches=500,coor_dir=args.coor_dir)
        df.to_csv(os.path.join(args.save_dir, 'clam_20', 'process_list_autogen.csv'), index=False)
        total_time +=t
        duration = time.time() - start_time
        print("totally it  took {} seconds".format(duration))
        # except:
        #     pass
        #break

    # seg_times, patch_times = seg_and_patch(**directories, **parameters,
    #                                        patch_size=args.patch_size, step_size=args.step_size,
    #                                        seg=args.seg, use_default_params=False, save_mask=True,
    #                                        stitch=args.stitch,
    #                                        patch_level=args.patch_level, patch=args.patch,
    #                                        process_list=process_list, auto_skip=args.no_auto_skip,num_selected_patches=1000)
    #start_time = time.time()
    # Select K patches
    # file_path = '/home/ge54xof/TUM_slides/patches/007c63d6-08d8-3011-b4b2-f57bd6c0aa3f_205039.h5'
    # WSI_object = WholeSlideImage('/home/ge54xof/TUM_slides/007c63d6-08d8-3011-b4b2-f57bd6c0aa3f_205039.svs')
    # with h5py.File(file_path, 'r+') as h5file:
    #     coords = h5file['coords'][:]
    #
    #     if len(coords) >= args.num_selected_patches:
    #         selected_coors = select_random_pairs({'coords': coords,},
    #                                                             args.num_selected_patches)
    #         select_imgs = read_imgs_parallel(WSI_object,coords,args.patch_level,args.patch)
    #         write_to_h5(file_path, selected_coors, select_imgs)
    #     else:
    #         # Remove the h5 file if it contains fewer than 1000 'coors' and 'imgs'
    #         print(f"Removing {file_path} as it contains less than 1000 'coors' and 'imgs'")
    #         os.remove(file_path)
    #
    # ### Stop Patch Timer
    # patch_time_elapsed = time.time() - start_time
    # print(patch_time_elapsed)

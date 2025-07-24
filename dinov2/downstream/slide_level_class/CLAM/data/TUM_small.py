#import h5py
import glob, os

import pandas as pd
# from PIL import Image
# import psutil


# h5_dir = '/home/ge24juj/Foundation-Model-for-Pathology/data/camelyon/patches/normal_014.h5'
# out_dir = '/home/ge24juj/Foundation-Model-for-Pathology/data/camelyon/imgs_normal/'
#
# with h5py.File(h5_dir, 'r+') as h5file:
#     imgs = h5file['imgs']
#     #print(h5file['coords'].shape)
#
#     for i in [0,100,200,300,499]:
#         img = Image.fromarray(imgs[i,:],'RGB')
#         img.save(out_dir+str(i)+".png")

# with h5py.File(h5_dir, 'r+') as h5file:
#     imgs_4 = h5file['imgs'][:4]
#     print(imgs_4[0].shape)
#
#     for i in range(4):
#         img = Image.fromarray(imgs_4[i],'RGB')
#         img.save(out_dir+str(i)+".png")





# import psutil
# dir='/mnt/data/TUM_Slides_Patches/'
# disk_usage = psutil.disk_usage(dir)
# threshold = 5000
# if disk_usage.free / (2 ** 30) < threshold:
#     print("Patching stop! Free space is lower than threshold.")
#     print("Free space is only " + str(round(disk_usage.free / (2 ** 30),2)) + 'GB now.')

# def check_lrz_storage(lrz_dir="/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/", threshold=10000):
#     quota = 10000 #GB
#     cmd_lrz = "du -s "+lrz_dir
#     stream = os.popen('ssh ge24juj2@login.ai.lrz.de '+cmd_lrz)
#     output = stream.read()
#     usage = output.split()[0] # G or M or
#     usage = usage.split('G')[0]
#     usage_GB = int(usage)/(2**20) #usage in GB
#     free_GB = quota - usage_GB
#
#     if free_GB < threshold:
#         print("Stop transferring!")
#         print(str(round(usage_GB,2)) + "G is used, while " + str(round(free_GB,2)) + "G is free")
#         return False
#     else:
#         return True
#     #return round(usage_GB,2), round(free_GB,2)
# #check_lrz_storage()
#
#
# def check_local_storage(dir='/mnt/data/TUM_Slides_Patches/',threshold=500):
#     disk_usage = psutil.disk_usage(dir)
#     if disk_usage.free / (2 ** 30) < threshold:
#         print("Patching stop! Free space is lower than threshold.")
#         print("Free space is only "+str(round(disk_usage.free / (2 ** 30),2))+ 'GB now.')
#         return False
#     else:
#         return True
#
# check_local_storage()
# def mark_processed_slide(df):
#     """
#     mark already existed slide as 'processed'
#     """
#
#
#     DST_patches_dir = "/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/TUM_patches/"
#     cmd_lrz = "ls " + DST_patches_dir + " |grep '.h5'"
#     stream = os.popen('ssh ge24juj2@login.ai.lrz.de ' + cmd_lrz)
#     output = stream.read()
#     h5_list_lrz = output.split()
#     # print(len(usage))
#
#     mask = df['status'] == "tbp"
#     process_stack = df[mask]
#     total = len(process_stack)
#
#     patch_save_dir = "/mnt/data/TUM_Slides_Patches/patches"
#
#
#
#     for i in range(total):
#         idx = process_stack.index[i]
#         slide = process_stack.loc[idx, 'slide_id']
#         slide_id = os.path.basename(slide).split('.')[0]
#         file_dir = slide_id + '.h5'
#
#         if file_dir in h5_list_lrz or os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
#             df.loc[idx, 'status'] = 'already_exist'
#             df.loc[idx,'process'] = 1000
#
#     print("finish marking processed slide")
#     return df


# process_list_merged = "/mnt/data/TUM_Slides_Patches/process_list_update.csv"
# process_list_gh = "/mnt/data/TUM_Slides_Patches/process_list_gh.csv"
# #process_list_new = "~/Foundation_model/moco-v3/data/process_list_autogen.csv"
# df1 = pd.read_csv(process_list_merged)
# df_gh = pd.read_csv(process_list_gh)
#
# n = 112584
#
# print('before')
# #print(df1.iloc[100])
# key_ls = df1.keys()[1:]
# key_backup = ['process', 'status', 'seg_level', 'sthresh', 'mthresh', 'close',
#        'use_otsu', 'keep_ids', 'exclude_ids', 'a_t', 'a_h', 'max_n_holes',
#        'vis_level', 'line_thickness', 'use_padding', 'contour_fn']
# print(df1.iloc[n-2:n+1,[1,2]])
# #print(df.keys()[1:] == df1.keys()[1:])
# #df1 = df1[:n]
# #print(len(df1))
# #tmp = df_backup[key_ls]
# #print(df_backup.slide_id.isin(df1.slide_id))
# df1.loc[n-1:, key_backup] = df_gh.loc[n-1:,key_backup]
#= df_backup[key_backup]

#df.iloc[:n,df.keys()[1:]] = df1[:n,df.keys()[1:]]
#df1 = df1.merge(df_backup,on='slide_id',how="right")
# print('after')
# print(df1.iloc[202460:202465,[1,2]])
#
# print(len(df1))
#
# # mask = df['status'] == "already_exist"
# # df.loc[mask,'process'] = 1000
# # df.loc[df['status'] == "processed",'status'] = "tbp"
# process_list_update = "/mnt/data/TUM_Slides_Patches/process_list_merged.csv"
# df1.to_csv(process_list_update, index=False)


# ls = ['/mnt/data/b9705b89-f57c-cf24-3a7a-e4eb62bfe1c9_191033.h5', #no
#       '/mnt/data/c7ea3137-bce6-24c1-1323-869c289005e4_042312.h5',
#       '/mnt/data/a65a0847-853b-29f7-f500-16a21a6b05a1_204703.h5',
#       '/mnt/data/aa6e71a7-d041-f935-2184-a6a5f9f81b57_023828.h5',
#        '/mnt/data/c7ea3137-bce6-24c1-1323-869c289005e4_042312.h5',
#        '/mnt/data/h2021000281t1-e-1_052647.h5',
#       '/mnt/data/h2021000281t1-h-1_051738.h5']

# dir_path = '/mnt/data'
# i =0
# j=0
# num_pathces_ls = []
# for file in os.listdir(dir_path):
#     # check only text files
#     if file.endswith('.h5'):
#         j+=1
#         try:
#                with h5py.File(os.path.join(dir_path,file), 'r') as h5file:
#                    n = h5file['imgs'].shape[0]
#                    num_pathces_ls.append(n)
#                    pass
#         except:
#             #os.remove
#             i +=1
#             #os.remove(os.path.join(dir_path,file))
#             print(file)
#     if i+j >5000:
#         break
# print(i)
# print(j)
# #print(num_pathces_ls)
# print(len(num_pathces_ls))



# import pandas as pd
# import numpy as np
#
# slides_each_epoch = 1000
# data_dir = '/mnt/data'
#
# slides_valid_ls = sorted(
#     [os.path.splitext(filename)[0] for filename in os.listdir(data_dir) if filename.endswith('.h5')])
#
# if len(slides_valid_ls) >= slides_each_epoch:
#     slides_id_ls = slides_valid_ls[:slides_each_epoch]  # normal case: selects only the first N slides
#     #last_batch = last_slide_in_local_ls(slides_id_ls)  # True if it is the last batch
#
# status_ls = ['tb trained'] * len(slides_valid_ls)
#
# batch_slides_data = {
#     'slide_id': slides_valid_ls,
#     'status': status_ls  # 'deleted', 'training', 'tb trained'
# }
#
# local_slides_df = pd.DataFrame(batch_slides_data)
# df = pd.read_csv('/mnt/data/scripts/process_list_merged.csv')
#
# df['slide_id'] = df['slide_id'].str.split('/').str[-1].str.replace('.svs', '')
# num_ls = pd.merge(local_slides_df, df[['slide_id', 'process']], on='slide_id', how='left')[
#     'process'].values
#
# num_ls[num_ls > 500] = 500
# length = num_ls.sum()
# accumulate_ls = np.add.accumulate(num_ls)
#
# def get_bag_given_id(id,accumulate_ls):
#     bag_candidate_idx = np.argwhere(accumulate_ls > id).min()  # ]
#
#     return bag_candidate_idx
# #print(accumulate_ls)
# bag_candidate_idx = get_bag_given_id(100,accumulate_ls)
# slide_id = local_slides_df.slide_id.tolist()
# print(slide_id[bag_candidate_idx])



# import numpy as np
#
# dir = "local_slides_df_bt_8.csv"
# pd1 = pd.read_csv(dir)
# slide_ls = pd1.slide_id.tolist()
# for i in range(len(slide_ls)):
#     slide = slide_ls[i]
#
#     h5_dir = os.path.join('/mnt/data',slide+'.h5')
#
#     if not os.path.exists(h5_dir):
#         print(slide)
import os
def sample_patches(TUM_data_dir,slides_each_epoch):
    # sample patches in public/private dataset in a ratio of 1:9
    # 1. list public dataset patches in TCGA/CAMELYON16/CAMELYON17/SLN/others in one ls []
    TCGA_dir = '/mnt/nfs03-R6/TCGA/clam/patches/'
    CAMELYON16_dir = '/mnt/nfs03-R6/CAMELYON16/clam/patches/'
    CAMELYON17_dir = '/mnt/nfs03-R6/CAMELYON17/clam/patches/'
    SLN_dir = '/mnt/nfs03-R6/SLN_cli/clam/patches/'
    TCGA_list = sorted([os.path.join(TCGA_dir,filename) for filename in os.listdir(TCGA_dir) if filename.endswith('.h5')])
    CAMELYON16_list = sorted([os.path.join(CAMELYON16_dir,filename) for filename in os.listdir(CAMELYON16_dir) if filename.endswith('.h5')])
    CAMELYON17_list = sorted([os.path.join(CAMELYON17_dir,filename) for filename in os.listdir(CAMELYON17_dir) if filename.endswith('.h5')])
    SLN_list = sorted([os.path.join(SLN_dir,filename) for filename in os.listdir(SLN_dir) if filename.endswith('.h5')])

    public_list = TCGA_list + CAMELYON16_list + CAMELYON17_list + SLN_list
    #print(len(public_list))
    # 2. list all available TUM dataset patches in one ls []
    # TUM_list = []
    # for TUM_dir in TUM_data_dir:
    #     #TUM_dir = '/mnt/nfs02-R6/TUM_slides/clam/patches/'
    #     TUM_list = sorted([os.path.join(TUM_dir,filename) for filename in os.listdir(TUM_dir) if filename.endswith('.h5')])
    #     TUM_list += TUM_list
    # print('load TUM data finished')

    # 3. list all all patches appeared in all local df 'data/local_slides_df_bt_*.csv' with header 'slide_id'

    local_slides_dfs = ['/home/ge24juj/dino-tum/dinov2/data/datasets/tum_small.csv']
    print(local_slides_dfs)
    local_slides_df = pd.concat([pd.read_csv(df) for df in local_slides_dfs])
    local_slides_paths_ls = local_slides_df['slide_id'].tolist()
    print('load data df finished')
    # 4. filter out the already trained patches from public and TUM dataset
    public_list = [path for path in public_list if path not in local_slides_paths_ls]
    #TUM_list = [path for path in TUM_list if path not in local_slides_paths_ls]
    print(len(public_list))
    #print(len(TUM_list))
    # 5. sample patches from public and TUM dataset in a ratio 1:9, the total number of patches is 5000
    # public_num = slides_each_epoch*0.001
    # TUM_num = slides_each_epoch*0.999
    # random.seed(123)
    # #public_sample = random.sample(public_list,public_num)
    # public_sample = public_list[:int(public_num)]
    # #TUM_sample = random.sample(TUM_list,TUM_num)
    # TUM_sample = TUM_list[:int(TUM_num)]
    # final_sample_ls = public_sample + TUM_sample
    # print(len(final_sample_ls))

    return public_list

public_list = sample_patches(None,None)



#compare if different models output different patch features

import os
import torch
from extract_patch_features import get_dino_finetuned_downloaded

def feature_compare(feats_dir, model1, model2, dataset,split):
    feats1 = dataset + '_' + model1 + '_' + split + '.pth'
    feats2 = dataset + '_' + model2 + '_' + split + '.pth'

    feats1 = torch.load(os.path.join(feats_dir, feats1),map_location=torch.device("cpu"))
    feats2 = torch.load(os.path.join(feats_dir, feats2),map_location=torch.device("cpu"))
    print(feats1.size())
    print(feats1.size())

    #compare if they are the same
    result = torch.equal(feats1, feats2)
    print(str(result)+ ' for ' + model1 + ' and ' + model2 + ' on ' + dataset + ' ' + split)

    return None

def feats_level_compare(feats_dir, model1, model2, dataset,split):
    for data in dataset:
        for s in split:
            try:
                feature_compare(feats_dir, model1, model2, data, s)
            except Exception as e:
                print(e)
                continue
    return None

def model_para_compare(model1, model2):
    #compare if two models have the same parameters
    model_dir = '/home/ge24juj/dino-tum/eval/'

    model1 = get_dino_finetuned_downloaded(DINO_PATH_FINETUNED_DOWNLOADED=os.path.join(model_dir,model1,'teacher_checkpoint.pth'))
    model2 = get_dino_finetuned_downloaded(DINO_PATH_FINETUNED_DOWNLOADED=os.path.join(model_dir,model2,'teacher_checkpoint.pth'))

    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            print('Models match on', key_item_1[0])
            #pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                pass
                #print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

    return None

if __name__ == '__main__':
    #150K_final, 150K_before_spike
    model1 = 'one_slide'
    model2 = 'Dino_manual_74340'
    dataset = ['CCRCC','CRC_norm','CRC_unnorm','MHIST']
    split = ['train', 'test']
    feats_dir = '/home/ge24juj/dino-tum/dinov2/downstream/feats'

    #model_para_compare(model1, model2)
    feats_level_compare(feats_dir, model1, model2, dataset, split)


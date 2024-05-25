import os
import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(srcFolder)

from methods.utils.metrics import *
from methods.utils.util import *

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Evaluate predicted saliency map')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--fixation_folder', type=str, default='')
parser.add_argument('--point_folder', type=str, default='')
parser.add_argument('--salmap_folder', type=str, default='')
parser.add_argument('--split_file', type=str, default='')
parser.add_argument('--fxt_loc_name', type=str, default='fixationPts')# 'fixationPts''fixLocs'
parser.add_argument('--fxt_size', type=str, default=(480, 640), help='fixation resolution: (600, 800) | (480, 640) | (320, 640) (768, 1366)')
parser.add_argument('--appendix', type=str, default='')
parser.add_argument('--file_extension', type=str, default='jpg')
args = parser.parse_args()

fixation_folder = args.fixation_folder
salmap_folder = args.salmap_folder
point_folder = args.point_folder

fxtimg_type = detect_images_type(fixation_folder)
split_file = args.split_file
if split_file != '' and os.path.isfile(split_file):
    print('in to here 0')
    npzfile = np.load(split_file)
    salmap_names = [os.path.join(salmap_folder, x) for x in npzfile['val_imgs']]
    gtsal_names = [os.path.join(fixation_folder, x[:x.find('.')+1]+fxtimg_type) for x in npzfile['val_imgs']]
    fxtpts_names = [os.path.join(fixation_folder, '{}mat'.format(x[:x.find('.')+1])) for x in npzfile['val_imgs']]
else:
    print('in to here 1')
    print('Calculating......')
    salmap_names = load_allimages_list(salmap_folder)
    gtsal_names = []
    fxtpts_names = []
    pts_names = []
    for sn in salmap_names:
        file_name = sn.split('/')[-1]
        gtsal_names.append(os.path.join(fixation_folder, '{}{}'.format(file_name[:file_name.find('.')+1], fxtimg_type)))
        fxtpts_names.append(os.path.join(fixation_folder, '{}mat'.format(file_name[:file_name.find('.')+1])))
        pts_names.append(os.path.join(point_folder, '{}mat'.format(file_name[:file_name.find('.')+1])))
    print('out form here and :')
    print('salmap_names :', salmap_names[0])
    print('gtsal_names :', gtsal_names[0])
    print('fxtpts_names :', fxtpts_names[0])
    print('pts_names :', pts_names[0])
    print('fxt_loc_name :', args.fxt_loc_name)
    print(salmap_names[0].split('/')[-3])

if salmap_names[0].split('/')[-3] == 'salicon':
    nss_score, _ = nss(salmap_names, pts_names, image_size=args.fxt_size, sigma=-1.0, fxt_field_in_mat=args.fxt_loc_name, dataset='salicon')
    emd_score, _ = emd(salmap_names, gtsal_names, image_size=args.fxt_size)
    sim_score, _ = sim(salmap_names, gtsal_names, image_size=args.fxt_size)
    kld_score, _ = kld(salmap_names, gtsal_names, image_size=args.fxt_size)
    cc_score, _ = cc(salmap_names, gtsal_names, image_size=args.fxt_size)
    auc_score, _ = auc_salicon(salmap_names, pts_names, image_size=args.fxt_size, sigma=-1.0, fxt_field_in_mat=args.fxt_loc_name)
    aucj_score, _ = auc_judd(salmap_names, pts_names, image_size=args.fxt_size, fxt_field_in_mat=args.fxt_loc_name, dataset='salicon')
    aucb_score, _ = auc_borji(salmap_names, pts_names, image_size=args.fxt_size, fxt_field_in_mat=args.fxt_loc_name, dataset='salicon')
    # aucs_score, _ = auc_borji.compute_score_shuffled(salmap_names, pts_names, gtsal_names, image_size=args.fxt_size, sigma=-1.0, fxt_field_in_mat=args.fxt_loc_name)
else:
    nss_score, _ = nss(salmap_names, pts_names, image_size=args.fxt_size, sigma=-1.0, fxt_field_in_mat=args.fxt_loc_name)
    emd_score, _ = emd(salmap_names, gtsal_names, image_size=args.fxt_size)
    sim_score, _ = sim(salmap_names, gtsal_names, image_size=args.fxt_size)
    kld_score, _ = kld(salmap_names, gtsal_names, image_size=args.fxt_size)
    cc_score, _ = cc(salmap_names, gtsal_names, image_size=args.fxt_size)
    auc_score, _ = auc(salmap_names, pts_names, image_size=args.fxt_size, sigma=-1.0, fxt_field_in_mat=args.fxt_loc_name)
    aucj_score, _ = auc_judd(salmap_names, pts_names, image_size=args.fxt_size, fxt_field_in_mat=args.fxt_loc_name)
    aucb_score, _ = auc_borji(salmap_names, pts_names, image_size=args.fxt_size, fxt_field_in_mat=args.fxt_loc_name)
    # aucs_score, _ = auc_borji.compute_score_shuffled(salmap_names, pts_names, gtsal_names, image_size=args.fxt_size, sigma=-1.0, fxt_field_in_mat=args.fxt_loc_name)
with open(args.output, 'a') as f:
    f.write('{}: emd:{:0.4f}, sim:{:0.4f}, kld:{:0.4f}, cc:{:0.4f}, nss:{:0.4f}, auc:{:0.4f}, auc_judd:{:0.4f}, auc_borji:{:0.4f}{}\n'.format(
            salmap_names[0].split('/')[-3],
            emd_score, sim_score, kld_score, cc_score, nss_score, auc_score, aucj_score, aucb_score, args.appendix))
print('End of Calculation!')
import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        d.pop(key)
        print('key: {} is removed'.format(key))
    return d


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="e2e_mask_rcnn_X_101_32x8d_FPN_1x_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
#_d = load_c2_format(cfg, DETECTRON_PATH)
_d=torch.load(DETECTRON_PATH)
newdict = _d['model']
print(_d['model'].keys())
#exit()
newdict['model'] = removekey(_d['model'],['module.roi_heads.box.predictor.cls_score.weight',
    'module.roi_heads.box.predictor.cls_score.bias','module.roi_heads.box.predictor.bbox_pred.weight',
    'module.roi_heads.box.predictor.bbox_pred.bias','module.roi_heads.mask.predictor.mask_fcn_logits.weight',
    'module.roi_heads.mask.predictor.mask_fcn_logits.bias'])

torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))

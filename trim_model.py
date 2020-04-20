import os
import torch
import argparse
#from maskrcnn_benchmark.config import cfg
#from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        d.pop(key)
        print('key: {} is removed'.format(key))
    return d


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="pretrained_model/e2e_faster_rcnn_R_50_C4_1x.pth",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="pretrained_model/e2e_faster_rcnn_R_50_C4_1x_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_faster_rcnn_R_50_C4_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

#cfg.merge_from_file(args.cfg)
#_d = load_c2_format(cfg, DETECTRON_PATH)
model=torch.load(DETECTRON_PATH)
#newdict = _d['model']
for k in (model['model'].keys()):
    print('before ',k)

del model["model"]["module.roi_heads.box.predictor.cls_score.weight"]
del model["model"]["module.roi_heads.box.predictor.cls_score.bias"]
del model["model"]["module.roi_heads.box.predictor.bbox_pred.weight"]
del model["model"]["module.roi_heads.box.predictor.bbox_pred.bias"]
#mask prediction
try:
    del model["model"]["module.roi_heads.mask.predictor.mask_fcn_logits.weight"]
    del model["model"]["module.roi_heads.mask.predictor.mask_fcn_logits.bias"]
except:
    pass
# RPN
del model["model"]["module.rpn.head.cls_logits.weight"]
del model["model"]["module.rpn.head.cls_logits.bias"]
del model["model"]["module.rpn.head.bbox_pred.weight"]
del model["model"]["module.rpn.head.bbox_pred.bias"]

del model['iteration']
del model['scheduler']
del model['optimizer']

for k in (model['model'].keys()):
    print('after ',k)

torch.save(model, args.save_path)
print('saved to {}.'.format(args.save_path))

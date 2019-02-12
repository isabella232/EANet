import cv2
import numpy as np
import torchvision.transforms.functional as F

from package.model.model import Model
from package.utils.misc import import_file
from PIL import Image


# def create_model(self):
#     if hasattr(self, "train_loader"):
#         reid_loader = (
#             self.train_loader.loaders[0]
#             if self.cfg.cd_ps_loss.use
#             else self.train_loader
#         )
#         self.cfg.model.num_classes = reid_loader.dataset.num_ids
#     self.model = Model(deepcopy(self.cfg.model))
#     self.model = may_data_parallel(self.model)
#     self.model.to(self.device)
# 
# 
# def load_items(self, model=False, optimizer=False, lr_scheduler=False):
#     """To allow flexible multi-stage training."""
#     cfg = self.cfg.log
#     objects = {}
#     if model:
#         objects["model"] = self.model
#     if optimizer:
#         objects["optimizer"] = self.optimizer
#     if lr_scheduler:
#         objects["lr_scheduler"] = self.lr_scheduler
#     load_ckpt(objects, cfg.ckpt_file, strict=False)
# 
# 
# # Objects: model, optimizer, lr_scheduler
# def load_ckpt(objects, ckpt_file, strict=True):
#     """Load state_dict's of modules/optimizers/lr_schedulers from file.
#     Args:
#         objects: A dict, which values are either
#             torch.nn.optimizer
#             or torch.nn.Module
#             or torch.optim.lr_scheduler._LRScheduler
#             or None
#         ckpt_file: The file path.
#     """
#     assert osp.exists(ckpt_file), "ckpt_file {} does not exist!".format(ckpt_file)
#     assert osp.isfile(ckpt_file), "ckpt_file {} is not file!".format(ckpt_file)
#     ckpt = torch.load(ckpt_file, map_location=(lambda storage, loc: storage))
#     for name, obj in objects.items():
#         if obj is not None:
#             # Only nn.Module.load_state_dict has this keyword argument
#             if not isinstance(obj, torch.nn.Module) or strict:
#                 obj.load_state_dict(ckpt["state_dicts"][name])
#             else:
#                 load_state_dict(obj, ckpt["state_dicts"][name])
#     objects_str = ", ".join(objects.keys())
#     msg = "=> Loaded [{}] from {}, epoch {}, score:\n{}".format(
#         objects_str, ckpt_file, ckpt["epoch"], ckpt["score"]
#     )
#     print(msg)
#     return ckpt["epoch"], ckpt["score"]
# 
# 
# # Delete me?
# def load_state_dict(model, src_state_dict, fold_bnt=True):
#     """Copy parameters and buffers from `src_state_dict` into `model` and its
#     descendants. The `src_state_dict.keys()` NEED NOT exactly match
#     `model.state_dict().keys()`. For dict key mismatch, just
#     skip it; for copying error, just output warnings and proceed.
# 
#     Arguments:
#         model: A torch.nn.Module object.
#         src_state_dict (dict): A dict containing parameters and persistent buffers.
#     Note:
#         This is modified from torch.nn.modules.module.load_state_dict(), to make
#         the warnings and errors more detailed.
#     """
#     from torch.nn import Parameter
# 
#     dest_state_dict = model.state_dict()
#     for name, param in src_state_dict.items():
#         if name not in dest_state_dict:
#             continue
#         if isinstance(param, Parameter):
#             # backwards compatibility for serialized parameters
#             param = param.data
#         try:
#             dest_state_dict[name].copy_(param)
#         except Exception, msg:
#             print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))
# 
#     # New version of BN has buffer `num_batches_tracked`, which is not used
#     # for normal BN, so we fold all these missing keys into one line
#     def _fold_nbt(keys):
#         nbt_keys = [s for s in keys if s.endswith('.num_batches_tracked')]
#         if len(nbt_keys) > 0:
#             keys = [s for s in keys if not s.endswith('.num_batches_tracked')] + ['num_batches_tracked  x{}'.format(len(nbt_keys))]
#         return keys
# 
#     src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
#     if len(src_missing) > 0:
#         print("Keys not found in source state_dict: ")
#         if fold_bnt:
#             src_missing = _fold_nbt(src_missing)
#         for n in src_missing:
#             print('\t', n)
# 
#     dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
#     if len(dest_missing) > 0:
#         print("Keys not found in destination state_dict: ")
#         if fold_bnt:
#             dest_missing = _fold_nbt(dest_missing)
#         for n in dest_missing:
#             print('\t', n)
# 
# 
# 
# 
# 
# def extract_batch_feat(model, in_dict, cfg):
#     model.eval()
#     with torch.no_grad():
#         in_dict = recursive_to_device(in_dict, cfg.device)
#         import ipdb; ipdb.set_trace()
#         out_dict = model(in_dict, forward_type=cfg.forward_type)
#         import ipdb; ipdb.set_trace()
#         out_dict['feat_list'] = [normalize(f) for f in out_dict['feat_list']]
#         import ipdb; ipdb.set_trace()
#         feat = torch.cat(out_dict['feat_list'], 1)
#         import ipdb; ipdb.set_trace()
#         feat = feat.cpu().numpy()
#         import ipdb; ipdb.set_trace()
#         ret_dict = {
#             'im_path': in_dict['im_path'],
#             'feat': feat,
#         }
#         if 'label' in in_dict:
#             ret_dict['label'] = in_dict['label'].cpu().numpy()
#         if 'cam' in in_dict:
#             ret_dict['cam'] = in_dict['cam'].cpu().numpy()
#         if 'visible' in out_dict:
#             ret_dict['visible'] = out_dict['visible'].cpu().numpy()
#     return ret_dict
# 
# 
# def extract_dataloader_feat(model, loader, cfg):
#     dict_list = []
#     for batch in tqdm(loader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
#         feat_dict = extract_batch_feat(model, batch, cfg)
#         dict_list.append(feat_dict)
#     ret_dict = concat_dict_list(dict_list)
#     return ret_dict


def resize(image, cfg):
    return Image.fromarray(
        cv2.resize(
            np.array(image),
            tuple(cfg.im.h_w[::-1]),
            interpolation=cv2.INTER_LINEAR
        )
    )


def to_tensor(image, cfg):
    image = F.to_tensor(image)
    return F.normalize(image, cfg.im.mean, cfg.im.std)


# Setup
image_path = "dataset/market1501/Market-1501-v15.09.15/query/0054_c4s1_006926_00.jpg"
cfg = import_file("exp/eanet/test_paper_models/GlobalPool/market1501/default.py").cfg
cfg.im = {
    'std': [0.229, 0.224, 0.225],
    'h_w': [256, 128],
    'mean': [0.486, 0.459, 0.408]
}
model = Model(cfg.model)
model.eval()

# Process input image
image = Image.open(image_path).convert("RGB")
image = resize(image, cfg)
image = to_tensor(image, cfg)
image = image[None, :]  # Fake stack

# Run this bad boy
input_dict = {'im': image}
output_dict = model(input_dict)
feature = output_dict['feat_list'][0]

import ipdb; ipdb.set_trace()

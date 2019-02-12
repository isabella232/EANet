import torch.nn as nn


class GlobalPool(object):
    def __init__(self, cfg):
        self.pool = nn.AdaptiveAvgPool2d(1) if cfg.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d(1)

    def __call__(self, in_dict):
        feat = self.pool(in_dict['feat'])
        feat = feat.view(feat.size(0), -1)
        out_dict = {'feat_list': [feat]}
        print(in_dict['feat'].shape)
        print(in_dict['feat'].mean())
        print(feat.shape)
        print(feat.mean())
        return out_dict

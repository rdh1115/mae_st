import unittest
from functools import partial
import torch.nn as nn
import models_mae
from util.kinetics import Kinetics
import torch

class ModelTest(unittest.TestCase):
    def testMAEVIT(self):
        model = models_mae.MaskedAutoencoderViT(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        print(model.patch_embed.patch_size)
        dataset_train = Kinetics(
            mode="pretrain",
            path_to_data_dir='/kaggle/input/kinetics-test/data',
        )
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=3,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
        print(next(iter(data_loader_train)))



if __name__ == '__main__':
    unittest.main()
import unittest

import torch
import torch.nn as nn

from train.create_trainer import GradualBackboneUnfreezer


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1), nn.BatchNorm2d(2))
        self.fc1 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc1(x.flatten(1))


class GradualBackboneUnfreezerTest(unittest.TestCase):
    def test_unfreezes_complete_blocks_from_back_to_front(self):
        model = _ToyModel()
        unfreezer = GradualBackboneUnfreezer(model)

        self.assertEqual(unfreezer.frozen_block_names, ["layer2", "layer1"])
        self.assertFalse(any(parameter.requires_grad for parameter in model.layer1.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.layer2.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.fc1.parameters()))

        self.assertEqual(unfreezer.unfreeze_next(), ["layer2"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.layer2.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.layer1.parameters()))

        self.assertEqual(unfreezer.unfreeze_next(), ["layer1"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.layer1.parameters()))
        self.assertEqual(unfreezer.frozen_block_names, [])

    def test_frozen_batchnorm_stays_in_eval_until_block_is_unfrozen(self):
        model = _ToyModel()
        unfreezer = GradualBackboneUnfreezer(model)
        model.train()

        model(torch.ones(2, 1, 2, 2))
        self.assertFalse(model.layer1.training)
        self.assertFalse(model.layer2.training)

        unfreezer.unfreeze_next()
        model.train()
        model(torch.ones(2, 1, 2, 2))
        self.assertFalse(model.layer1.training)
        self.assertTrue(model.layer2.training)

    def test_data_parallel_uses_the_same_block_names(self):
        model = nn.DataParallel(_ToyModel())
        unfreezer = GradualBackboneUnfreezer(model)
        self.assertEqual(unfreezer.frozen_block_names, ["layer2", "layer1"])


if __name__ == "__main__":
    unittest.main()

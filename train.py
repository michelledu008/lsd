import funlib.learn.torch as ft
import gunpowder as gp
import gunpowder.torch.Train
import numpy as np
import torch
from radam import RAdam

import logging
logging.basicConfig(level=logging.INFO)

class BaselineUNet(torch.nn.Module):

    def __init__(self):

        super(BaselineUNet, self).__init__()

        unet = ft.models.UNet(
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factor=5,
            kernel_size_down=[
                [(3, 3), (3, 3)],
                [(3, 3), (3, 3)],
                [(3, 3), (3, 3)],
                [(3, 3), (3, 3)]
            ],
            kernel_size_up=[
                [(3, 3), (3, 3)],
                [(3, 3), (3, 3)],
                [(3, 3), (3, 3)]
            ],
            downsample_factors=[(2, 2), (2, 2), (2, 2)],
            constant_upsample=True)
        self.sequence = torch.nn.Sequential(
            unet,
            torch.nn.Conv2d(12, 2, (1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequence(x)


class Squash(gp.BatchFilter):
    '''Remove a dimension of arrays in batches passing through this node.
    Assumes that the shape of this dimension is 1.

    Args:

        dim (int, optional):

              The dimension to remove (defaults to 0, i.e., the first
              dimension).
    '''

    def __init__(self, dim=0):
        self.dim = dim

    def setup(self):

        # remove selected dim from provided specs
        for key, upstream_spec in self.get_upstream_provider().spec.items():
            spec = upstream_spec.copy()
            spec.roi = gp.Roi(
                self.__remove_dim(spec.roi.get_begin()),
                self.__remove_dim(spec.roi.get_shape()))
            spec.voxel_size = self.__remove_dim(spec.voxel_size)
            self.spec[key] = spec

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        # add a new dim
        for key, spec in request.items():
            upstream_voxel_size = upstream_spec[key].voxel_size
            v = upstream_voxel_size[self.dim]
            spec.roi = gp.Roi(
                self.__insert_dim(spec.roi.get_begin(), 0),
                self.__insert_dim(spec.roi.get_shape(), v))
            if spec.voxel_size is not None:
                spec.voxel_size = self.__insert_dim(spec.voxel_size, v)

    def process(self, batch, request):

        for key, array in batch.arrays.items():

            # remove first dim
            array.spec.roi = gp.Roi(
                self.__remove_dim(array.spec.roi.get_begin()),
                self.__remove_dim(array.spec.roi.get_shape()))
            array.spec.voxel_size = self.__remove_dim(array.spec.voxel_size)
            assert array.data.shape[self.dim] == 1, \
                "Squash for dim %d requires that the array %s has size 1 in " \
                "that dim, but array shape is %s" % (
                    self.dim,
                    key,
                    array.data.shape)
            array.data = array.data.reshape(
                self.__remove_dim(array.data.shape))

    def __remove_dim(self, a):
        return a[:self.dim] + a[self.dim + 1:]

    def __insert_dim(self, a, s):
        return a[:self.dim] + (s,) + a[self.dim:]


class AddChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[np.newaxis]


class TransposeDims(gp.BatchFilter):

    def __init__(self, permutation):
        self.permutation = permutation

    def process(self, batch, request):

        for key, array in batch.arrays.items():
            array.data = array.data.transpose(self.permutation)


class RemoveChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[0]


if __name__ == "__main__":

    model = BaselineUNet()
    print(model)
    loss = torch.nn.MSELoss()
    optimizer = RAdam(model.parameters(), lr=1e-5)

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    affs = gp.ArrayKey('AFFS')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

    input_size = (100, 100)
    output_size = (8, 8)

    batch_size = 10

    pipeline = (
        gp.ZarrSource(
            '../../01_data/ctc/train_Fluo-N2DH-SIM+.zarr',
            {
                raw: 't01/raw',
                labels: 't01/gt'
            }) +
        # raw: (d=1, h, w)
        # labels: (d=1, h, w)
        gp.RandomLocation() +
        # raw: (d=1, h, w)
        # labels: (d=1, h, w)
        gp.AddAffinities(
            affinity_neighborhood=[(0, 1, 0), (0, 0, 1)],
            labels=labels,
            affinities=affs) +
        gp.Normalize(affs, factor=1.0) +
        # raw: (d=1, h, w)
        # affs: (c=2, d=1, h, w)
        Squash(dim=-3) +
        # get rid of z dim
        # raw: (h, w)
        # affs: (c=2, h, w)
        AddChannelDim(raw) +
        # raw: (c=1, h, w)
        # affs: (c=2, h, w)
        gp.PreCache() +
        gp.Stack(batch_size) +
        # raw: (b=10, c=1, h, w)
        # affs: (b=10, c=2, h, w)
        Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={'x': raw},
            target=affs,
            output=affs_predicted) +
        # raw: (b=10, c=1, h, w)
        # affs: (b=10, c=2, h, w)
        # affs_predicted: (b=10, c=2, h, w)
        TransposeDims((1, 0, 2, 3)) +
        # raw: (c=1, b=10, h, w)
        # affs: (c=2, b=10, h, w)
        # affs_predicted: (c=2, b=10, h, w)
        RemoveChannelDim(raw) +
        # raw: (b=10, h, w)
        # affs: (c=2, b=10, h, w)
        # affs_predicted: (c=2, b=10, h, w)
        gp.Snapshot(
            dataset_names={
                raw: 'raw',
                labels: 'labels',
                affs: 'affs',
                affs_predicted: 'affs_predicted'
            },
            every=100) +
        gp.PrintProfilingStats(every=100)
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(affs, output_size)
    request.add(affs_predicted, output_size)

    print("Starting training...")
    with gp.build(pipeline):
        for i in range(100000):
            pipeline.request_batch(request)

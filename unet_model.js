<docs lang="markdown">
[TODO: write documentation for this plugin.]
</docs>

<config lang="json">
{
  "name": "UNET",
  "type": "native-python",
  "version": "0.1.0",
  "description": "[TODO: describe this plugin with one sentence.]",
  "tags": [],
  "ui": "",
  "cover": "",
  "inputs": null,
  "outputs": null,
  "flags": [],
  "icon": "extension",
  "api_version": "0.1.7",
  "env": "conda create -n unet python=3.7.6",
  "permissions": [],
  "requirements": ["pip:-e git+https://github.com/TuragaLab/malis.git@572ef0420107eee3c721bdafb58775a8a0fc467a#egg=malis",
                    "pip: -e git+https://github.com/funkey/gunpowder.git@fdadd2dbfd836f45495b9c48ec98687e032e701d#egg=gunpowder",
                    "pip: torch",
                    "pip: zarr",
                    "pip: -e git+https://github.com/funkey/gunpowder.git@fdadd2dbfd836f45495b9c48ec98687e032e701d#egg=gunpowder",
                    "pip: -e git+https://github.com/funkelab/funlib.learn.torch.git@a38f67eb877097f4cdd1c2a1bf814e36a5ce3e22#egg=funlib.learn.torch", 
                    "pip: git+https://github.com/michelledu008/unet_sample.git", 
                    "repo:https://github.com/michelledu008/unet_sample.git",
                    "pip: keras"],
  "dependencies": ["oeway/ImJoy-Plugins:Im2Im-Dashboard", "oeway/ImJoy-Plugins:launchpad"]
}
</config>

<script lang="python">
from imjoy import api
import os
import gunpowder as gp
os.chdir('unet_sample')
from gunpowder.torch import Train
import radam
import zarr
import torch
import train
api.log("imports done")

class ImJoyPlugin():
    def setup(self):
        self.window = None
        api.log('initialized')
     
    async def start_train(self, data_dir):

        ret = await api.showDialog( name="Training Configurations", ui= "<br>".join([
            "UNet Model Name {id: 'model_name', type: 'string', placeholder: 'my_model'}",
            "Epochs { id: 'epochs', type:'number', placeholder: 30}",
            "Source folder name { id: 'source_dir', type:'string', placeholder: 'raw'}",
            "Target folder name { id: 'target_dir', type:'string', placeholder: 'gt'}",
            ]))
        
        model_name = ret.model_name
        epochs = int(ret.epochs)
        source_dir = ret.source_dir
        target_dir = ret.target_dir

        raw = gp.ArrayKey('RAW')
        labels = gp.ArrayKey('LABELS')
        affs = gp.ArrayKey('AFFS')
        affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

        model = train.BaselineUNet()
        api.log(str(model))

        loss = torch.nn.MSELoss()
        optimizer = radam.RAdam(model.parameters(), lr=1e-5)

        dataset_shape = zarr.open(str(data_dir))['train/raw'].shape
        num_samples = dataset_shape[0]
        sample_size = dataset_shape[1:]

        batch_size = 10
        input_size = (100,100)
        output_size = (8,8)

        pipeline = (
            gp.ZarrSource(
                data_dir,
                {
                    raw: 'train/{}'.format(source_dir),
                    labels: 'train/{}'.format(target_dir)
                },
                array_specs={
                    raw: gp.ArraySpec(
                        roi=gp.Roi((0, 0, 0), (num_samples,) + sample_size),
                        voxel_size=(1, 1, 1)),
                    labels: gp.ArraySpec(
                        roi=gp.Roi((0, 0, 0), (num_samples,) + sample_size),
                        voxel_size=(1, 1, 1))
                }) +
            # raw: (d=1, h, w)
            # labels: (d=1, fmap_inc_factors=5h, w)
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
            train.Squash(dim=-3) +
            # get rid of z dim
            # raw: (h, w)
            # affs: (c=2, h, w)
            train.AddChannelDim(raw) +
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
                output=affs_predicted,
                save_every=1000,
                log_dir='log') +
            # raw: (b=10, c=1, h, w)
            # affs: (b=10, c=2, h, w)
            # affs_predicted: (b=10, c=2, h, w)
            train.TransposeDims((1, 0, 2, 3)) +
            # raw: (c=1, b=10, h, w)
            # affs: (c=2, b=10, h, w)
            # affs_predicted: (c=2, b=10, h, w)
            train.RemoveChannelDim(raw) +
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


        api.log("Starting training...")
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="UNet Training", w=25, h=20, data={"display_mode": "all", 'metrics': ['loss'], 'callbacks': ['onStep']})

        with gp.build(pipeline):
            for i in range(epochs):
                batch = pipeline.request_batch(request)
                await self.dash.updateCallback('onStep', batch.iteration, {'loss': float(str(batch.loss))})
                api.log("Iteration {} {}".format(str(batch.iteration), str(batch.loss)))

    async def train_folder(self):
        self.dialog.close()
        await self.start_train("/groups/funke/home/dum/Projects/Fluo-N2DH-SIM+.zarr")

    async def predict_folder(self):
        self.dialog.close()
        ret = await api.showFileDialog(type="file", title="please select a model", engine=api.ENGINE_URL)
        checkpoint_file = ret.path

        


    async def run(self, ctx):
        self.dialog = await api.showDialog(type='launchpad', data= [
                {'name': 'Train', 'description': 'Trains unet with data from Fluo-N2DH-SIM+.zarr.', 'callback': self.train_folder, 'img': 'https://img.icons8.com/color/96/000000/opened-folder.png'},
                {'name': 'Predict', 'description': 'Predicts using existing model', 'callback': self.predict_folder, 'img': 'https://img.icons8.com/color/96/000000/double-right.png'}
            ]
        )


api.export(ImJoyPlugin())
</script>


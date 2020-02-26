<docs lang="markdown">
Local Shape Descriptors
-----------------------
-----------------------

Training
--------
To train using your own data, select a zarr container from your local directory (must be formatted to have raw and gt in train and test).

Specify training configuration or use the default ones that are preset in the window.
* Epochs is how many iterations you want to train the unet (normally around 50,000)
* Learning rate is the parameter which determines how much to adjust each step during each iteration (normally around 10x-5)
* asd

Once training starts, a real-time graph of the loss function will be displayed. Every 50 iterations, an image of the local shape descriptors will be displayed using Neuroglancer. 

Inferences
----------
Select a checkpoint file that you want to use for inferences. 


</docs>

<config lang="json">
{
  "name": "lsd",
  "type": "native-python",
  "version": "0.1.0",
  "description": "Using a Unet to predict local shape descriptors",
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
  "requirements": [],
  "dependencies": ["oeway/ImJoy-Plugins:Im2Im-Dashboard", "oeway/ImJoy-Plugins:launchpad"]
}
</config>

<script lang="python">

#requirements: 
#"pip:-e git+https://github.com/TuragaLab/malis.git@572ef0420107eee3c721bdafb58775a8a0fc467a#egg=malis",
#"pip: -e git+https://github.com/funkey/gunpowder.git@fdadd2dbfd836f45495b9c48ec98687e032e701d#egg=gunpowder",
#"pip: torch",
#"pip: zarr",
#"pip: -e git+https://github.com/funkey/gunpowder.git@fdadd2dbfd836f45495b9c48ec98687e032e701d#egg=gunpowder",
#"pip: -e git+https://github.com/funkelab/funlib.learn.torch.git@a38f67eb877097f4cdd1c2a1bf814e36a5ce3e22#egg=funlib.learn.torch", 
#"pip: git+https://github.com/michelledu008/unet_sample.git", 
#"repo:https://github.com/michelledu008/unet_sample.git"

from imjoy import api
import os
import gunpowder as gp
#os.chdir('unet_sample')
api.log(os.getcwd())
from gunpowder.torch import Train
import radam
import zarr
import torch
import train
import predict
api.log("imports done")

class ImJoyPlugin():
    def setup(self):
        self.window = None
        api.log('initialized')
     
    async def start_train(self):

        epochs = 1000
    
        data_dir = "/groups/funke/home/dum/Projects/Fluo-N2DH-SIM+.zarr"

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
        save_every = 1000

        api.log("starting to build pipeline")
        pipeline = train.build_pipeline(
                                data_dir,
                                model,
                                save_every,
                                batch_size,
                                input_size,
                                output_size,
                                raw,
                                labels,
                                affs,
                                affs_predicted)
        api.log("finished building pipeline")

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
        await self.start_train()

    async def predict_folder(self):
        self.dialog.close()
        ret = await api.showFileDialog(type="file", title="please select a model", engine=api.ENGINE_URL)
        checkpoint_file = os.getcwd() + ret.path
        api.log(ret.path)
        api.log(os.getcwd())
        api.log(checkpoint_file)

        data_dir = "/groups/funke/home/dum/Projects/Fluo-N2DH-SIM+.zarr/"
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = train.BaselineUNet()
        model.to(device)

        input_size = (100, 100)
        output_size = (8, 8)

        raw = gp.ArrayKey('RAW')
        labels = gp.ArrayKey('LABELS')
        affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

        dataset_shape = zarr.open(str(data_dir))['validate/raw'].shape
        num_samples = dataset_shape[0]
        sample_size = dataset_shape[1:]

        total_request = gp.BatchRequest()
        total_request.add(raw, sample_size)
        total_request.add(affs_predicted, sample_size)
        total_request.add(labels, sample_size)

        pipeline = predict.build_pipeline(
                data_dir,
                model,
                checkpoint_file,
                input_size,
                output_size,
                raw,
                labels,
                affs_predicted,
                dataset_shape,
                num_samples,
                sample_size)
        
        api.log("starting predictions")
        with gp.build(pipeline):
            batch = pipeline.request_batch(total_request)



    async def run(self, ctx):
        self.dialog = await api.showDialog(type='launchpad', data= [
                {'name': 'Train', 'description': 'Trains a UNet.', 'callback': self.train_folder, 'img': 'https://img.icons8.com/color/96/000000/opened-folder.png'},
                {'name': 'Inferences', 'description': 'Validates using existing model', 'callback': self.predict_folder, 'img': 'https://img.icons8.com/color/96/000000/double-right.png'}
            ]
        )


api.export(ImJoyPlugin())
</script>



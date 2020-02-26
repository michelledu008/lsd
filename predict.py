import torch
import train 
import zarr 
import gunpowder as gp
from gunpowder.torch.nodes.predict import * 

def build_pipeline(
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
        sample_size
        ):
    
    checkpoint = torch.load(checkpoint_file) 
    model.load_state_dict(checkpoint['model_state_dict'])

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(affs_predicted, output_size)
    scan_request.add(labels, output_size)

    pipeline = (
        gp.ZarrSource(
            str(data_dir),
            {
                raw: 'validate/raw',
                labels: 'validate/gt'
            }) +
        gp.Pad(raw, size=None) +
        gp.Normalize(raw) +
        # raw: (s, h, w)
        # labels: (s, h, w)
        train.AddChannelDim(raw) +
        # raw: (c=1, s, h, w)
        # labels: (s, h, w)
        train.TransposeDims(raw, (1, 0, 2, 3)) +
        # raw: (s, c=1, h, w)
        # labels: (s, h, w)
        Predict(
            model=model,
            inputs={'x': raw},
            outputs={0: affs_predicted}) +
        # raw: (s, c=1, h, w)
        # affs_predicted: (s, c=2, h, w)
        # labels: (s, h, w)
        train.TransposeDims(raw, (1, 0, 2, 3)) +
        train.RemoveChannelDim(raw) +
        # raw: (s, h, w)
        # affs_predicted: (s, c=2, h, w)
        # labels: (s, h, w)
        gp.PrintProfilingStats(every=100) + 
        gp.Scan(scan_request)
    )
    
    return pipeline 


if __name__ == "__main__":
    data_dir = "/groups/funke/home/dum/Projects/Fluo-N2DH-SIM+.zarr/"
    checkpoint_file = "/groups/funke/home/dum/ImJoyWorkspace/default/unet_sample/model_checkpoint_9000"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train.BaselineUNet()
    model.to(device) 
    print(model)

    output_size = (8, 8)
    input_size = (100, 100)
    
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

    pipeline = build_pipeline(
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
    with gp.build(pipeline): 
        batch = pipeline.request_batch(total_request)
        #evaluate


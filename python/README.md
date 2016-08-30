# Usage

## Forward
### Prepare network spec
- Predefined models can be found in [kaffe/models](https://github.com/torms3/kaffe/tree/master/models). 
- Running `python net.py z y x` will generate three `prototxt` files, i.e., `train.prototxt`, `val.prototxt`, and `deploy.prototxt`.
- Use `deploy.prototxt` for inference.

### Prepare configuration file
Example configuration file for inference: [forward.cfg.example](https://github.com/torms3/kaffe/blob/master/python/forward.cfg.example).

### Run forward pass

    python forward.py GPU_ID path/to/forward.cfg

`GPU_ID` is different from what is shown from `nvidia-smi`. `caffe device_query` will let you know the precise `GPU_ID` information. Basically faster GPU gets assigned lower number.

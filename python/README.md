## Usage

### Forward

    python forward.py 0 path/of/forward.cfg

0 is GPU ID, but this is different from the ID you can know from `nvidia-smi`. `caffe device_query` will give you the IDs. Basically faster GPU gets assigned lower number.

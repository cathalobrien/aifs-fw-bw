# aifs-fw-bw
A lightweight benchmarking code for AIFS which does a single FW and BW pass. 

# setup
```bash
pip install anemoi-training anemoi-models


#need a link to config/ in aifs-fw-bw/
ln -s anemoi-core/training/src/anemoi-training/config .
#need graphs/ file in aifs-fw-bw/
tar xvzf $SCRATCH/graphs.tar.gz .

#might need to edit datasets path in 'get_dataset()'
```

# example usage
```bash
#run with 'config/aifs-fw-bw.yaml', overwrites to 256 o1280 channels
srun -n 4 python main.py -c aifs-fw-bw -C 1024 -r n320
srun -n 4 python main.py -c aifs-fw-bw -C 256 -r o1280
srun -n 4 python main.py -c aifs-fw-bw -C 64 -r o2560

#checking correctness
CUBLAS_WORKSPACE_CONFIG=:16:8 srun -n 4 python main.py -C 64 -r o1280 --verify
```

# Checking correctness
You can use 'aifs-fw-bw' to check for equality across models. See the 'example usage' section above for an example of running with correctness checks enabled. Please note, to force determinism from 'cuBLAS' you must export an env var before running with correctness checks.

Be warned, this mode increases memory usage and heavily decreases performance. These degradations come from
* Suboptimal output recording, via the 'Output' class. inefficiencies include allocating pinned mem buffers and syncronously copying to CPU during the BM path. The output class also increases device memory usage.
* use of 'torch.use_deterministic_algorithms(True)' and 'CUBLAS_WORKSPACE_CONFIG=:16:8'

# example NSYS usage
```bash
nsys profile -o nsys/aifs-fw-bw-o1280-256c.%q{SLURM_PROCID} -f true --gpu-metrics-devices=all --cuda-memory true --python-backtrace=cuda python main.py -C 64 -r o1280
```
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
```

# example NSYS usage
```bash
nsys profile -o nsys/aifs-fw-bw-o1280-256c.%q{SLURM_PROCID} -f true --gpu-metrics-devices=all --cuda-memory true --python-backtrace=cuda python main.py -C 64 -r o1280
```
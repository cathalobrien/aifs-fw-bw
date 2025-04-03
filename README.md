# aifs-fw-bw
A lightweight benchmarking code for AIFS which does a single FW and BW pass. 

# setup
```bash
pip install anemoi-training anemoi-models

#need a link to config/ in aifs-fw-bw/
ln -s anemoi-core/training/src/anemoi-training/config .
#inputs/ contains  dummy datasets
# The required graphs will be built on the fly
ln -s $SCRATCH/path/to/aifs-fw-bw/inputs .

```

# example usage
```bash
#get a gpu node
salloc --mem=0 --qos=ng -N 1 --ntasks-per-node=4 --cpus-per-task=32 --gpus-per-task=1 --time=2:00:00

#run with 'config/aifs-fw-bw.yaml', overwrites to 256 o1280 channels
srun -n 4 python main.py -c aifs-fw-bw -C 1024 -r n320 --slurm
srun -n 4 python main.py -c aifs-fw-bw -C 256 -r o1280 --slurm
srun -n 4 python main.py -c aifs-fw-bw -C 64 -r o2560 --slurm

#checking correctness
srun -n 4 python main.py -C 64 -r o1280 --slurm -c default-config,new-config --verify 

#example torchrun command, benchmarking two different configs
torchrun --nproc-per-node 4 main.py -r o1280 -C 512 -c edge,head

#Mem snapshots - produces 2 snapshots 'aifs-fw-bw.head.pickle' and 'aifs-fw-bw.edge.pickle'
torchrun --nproc-per-node 4 main.py -r o1280 -C 512 -c head,edge --mem-snapshot
```

# Checking correctness
You can use 'aifs-fw-bw' to check for equality across models. See the 'example usage' section above for an example of running with correctness checks enabled. Please note, to force determinism from 'cuBLAS' you must export an env var before running with correctness checks.

Be warned, this mode increases memory usage and heavily decreases performance. These degradations come from
* Suboptimal output recording, via the 'Output' class. inefficiencies include allocating pinned mem buffers and syncronously copying to CPU during the BM path. The output class also increases device memory usage.
* use of 'torch.use_deterministic_algorithms(True)' and 'CUBLAS_WORKSPACE_CONFIG=:16:8'
* Aggresive clearing of the cache during correctness checking due to increased memory pressure => ~33% higher runtime

# example NSYS usage
```bash
srun -np 4 nsys profile -o nsys/aifs-fw-bw-o1280-256c.%q{SLURM_PROCID} -f true --gpu-metrics-device=all --cuda-memory true --python-backtrace=cuda --python-sampling=true python main.py -C 64 -r o1280
#torchrun example
torchrun --no-python --nproc-per-node 4 nsys profile -o aifs-9km-1024c-8mc.%q{RANK} -f true --gpu-metrics-device=all --cuda-memory true --python-backtrace=cuda python main.py -r o1280 -C 1024 -c hackathon
```

# TODO
## Improvements
* Remove requirement for dummy dataset and graph inputs
## bugs
* Correctness checking with a cloned compiled model passes. With a different model from the same config or an uncompiled clone it fails.
*~~Increased mem pressure when making two models from the same config than when I clone a model~~ mercurial OOMs while checking correctness
* Increased mem pressure while checking correctness after I merged correctness checking and torchrun/multi-config commits

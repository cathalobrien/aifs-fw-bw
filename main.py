import argparse
import torch
import torch.distributed as dist
import time
import os
import logging
import sys
import numpy as np
from contextlib import contextmanager
import subprocess

#for building models
from anemoi.models.interface import AnemoiModelInterface


#for bulding from config :(
from hydra import compose, initialize
from omegaconf import OmegaConf
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.schemas.base_schema import BaseSchema, UnvalidatedBaseSchema
from hydra.utils import instantiate

#loss
from anemoi.training.train.forecaster import GraphForecaster

log_level=logging.INFO
if (os.getenv("SLURM_PROCID", "0") != "0"):
    log_level=logging.WARNING
logging.basicConfig(format='%(message)s',stream=sys.stdout, level=log_level)
LOG = logging.getLogger(__name__) 
logging.getLogger("anemoi").setLevel(logging.WARNING) #suppress spammy Anemoi logging
logging.getLogger("hydra_plugins").setLevel(logging.WARNING) #and hydra

LOG.info("Imports completed")

def get_grid_points(res):
    if res == "o2560":
        return 26306560
    elif res == "o1280":
        return 6599680
    elif res == "n320":
        return 542080
    else:
        return 0

def parse_inputs(args, device):
    model=None
    
    if args.checkpoint != "":
        #TODO check if args.checkpoint path is valid
        LOG.info(f"Loading {args.checkpoint}...")
        model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device)
        LOG.info(f"Checkpoint loaded.")
        
    return model

#(1,2,1,'n320',99) gave an error in precproc, so changed to 100 vars
#       return F.linear(input, self.weight, self.bias)
#   RuntimeError: mat1 and mat2 shapes cannot be multiplied (542080x309 and 212x1024)
#n320 has to be 100
#o1280 has to 99
def generate_inputs(res,device,shape=None,vars=100,batch=1,time=2,ensemble=1,grad=True,dtype=torch.float32,world_size=1):
    #  x = batch[:, 0 : self.multi_step, None, ...]  #from predict_step
    # Preparing input tensor with shape (2, 99, 6599680)
    #batch time ensemble grid vars
    if res == "o1280":
        vars=99
    gridpoints=get_grid_points(res)//world_size
    if shape is None:
        shape=(batch,time,ensemble,gridpoints,vars)
    input=torch.randn(shape,dtype=dtype, device=device, requires_grad=grad)
    return input

#TODO replace this
def get_dataset(res):
    path="/home/mlx/ai-ml/datasets/"
    if res == "n320":
        return path, "aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v4.zarr"
    elif res == "o1280":
        return path, "aifs-ea-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"
    elif res == "o2560":
        return path, "aifs-rd-an-lwda-ifc3-mars-o2560-2023-2023-6h-v3-1week.zarr"
    else:
        raise ValueError(f"Error. {res=} unsupported")
    

def build_config(setup):
    with initialize(version_base=None, config_path=setup.config_path, job_name="debug"):
        hardware_paths_data, hardware_files_dataset=get_dataset(setup.res)
        hardware_list=[f"hardware.paths.data='{hardware_paths_data}'", f"hardware.files.dataset='{hardware_files_dataset}'"]
        parallel_list=[f"hardware.num_gpus_per_node={setup.procs_per_node}", f"hardware.num_nodes={setup.num_nodes}"]
        ignore_list=["diagnostics.log.wandb.entity=''", "diagnostics.log.mlflow.tracking_uri=''", "hardware.paths.output=''", "hardware.files.graph=''"]
        overrides= hardware_list + parallel_list + ignore_list
        config = compose(config_name=setup.config_name, overrides=overrides)
        
    #config = OmegaConf.to_object(config)
    LOG.debug(f"{config=}")
    #config=DotDict(**config) #has to be baseschema bc DataModule calls model_dump
    config=UnvalidatedBaseSchema(**config) #using Baseschema instantiaes all the objects early for some reason
    
    #change the setup slightly for o1280
    if setup.res == "o1280":
        config.data.forcing = list(config.data.forcing).remove("insolation")
        config.data.normalizer.none = list(config.data.normalizer.none).remove("insolation")
        #config.model.num_channels=256 #can run 128 on 1 40GB A100, or 256 on 4
    
    if setup.channels != config.model.num_channels:
        LOG.debug(f"Overwriting configs num_channels ({config.model.num_channels}) with the command line arg ({setup.channels})")
        config.model.num_channels = setup.channels    

    return config

def get_graph_data(res="n320"):
    graph=torch.load(f"graphs/{res}.graph", weights_only=False)
    LOG.debug(f"{graph=}")
    return graph

def get_loss(config, data_indices,device):
    variable_scaling = GraphForecaster.get_variable_scaling(
            config.model_dump(by_alias=True).training.variable_loss_scaling,
            config.model_dump(by_alias=True).training.pressure_level_scaler,
            data_indices,
        )

    loss = GraphForecaster.get_loss_function(
        config.training.training_loss,
        node_weights=torch.ones(1),
        scalars={"variable": (-1, variable_scaling), "loss_weights_mask": ((-2, -1), torch.ones((1, 1)))},
    ).to(device)
    return loss

def dummy_loss(_out):
    loss = 0
    if isinstance(_out, torch.Tensor):
        loss += _out.sum()
    else:
        for o in _out:
            loss += o.sum()
    return loss

def reset_grad(*inputs):
    for i in inputs:
        if isinstance(i, torch.Tensor) and i.requires_grad:
            i.grad = None
        elif isinstance(i, tuple):
            reset_grad(*i)

def build_model(setup):
    res=setup.res
    device=setup.device
    start_time=time.time()
    LOG.info(f"Building model based on '{setup.config_path}{setup.config_name}.yaml'...")
    config=build_config(setup)
    
    graph_data=get_graph_data(res)
    datamodule = AnemoiDatasetsDataModule(config, graph_data) #need training just for this
    
    model=AnemoiModelInterface(config=config, graph_data=graph_data, statistics=datamodule.statistics, data_indices=datamodule.data_indices, metadata=datamodule.metadata).to(device)
   
    if setup.model_comm_group is not None: 
        dist.barrier(setup.model_comm_group)
        
    LOG.info(f"Model built in {time.time()-start_time:.2f}s.")
    
    model.loss = get_loss(config, datamodule.data_indices, device)
    
    #needed for sharded batch
    model.grid_indices = instantiate(
            config.model_dump(by_alias=True).dataloader.grid_indices,
            reader_group_size=setup.world_size,)
    model.grid_indices.setup(graph_data) # need a loaded graph here
    #model.grid_indices.grid_size = get_grid_points(res) #instead we do the FullGrid setup here
    
    return model
    
def iter(model,setup, verbose=False):

    with profiler_wrapper(setup.device, "Generate inputs"):
        x = generate_inputs(res=setup.res,device=setup.device, grad=setup.bw, dtype=setup.dtype, world_size=setup.world_size)
    grid_shard_shapes = model.grid_indices.shard_shapes
    grid_shard_slice = model.grid_indices.get_shard_indices(setup.global_rank)
    
    #without torch.autocast(dtype=torch.float16) I got an error in FW pass
    #     File "/perm/naco/venvs/aifs-fw-bw/lib/python3.11/site-packages/flash_attn/flash_attn_interface.py", line 96, in _flash_attn_forward
    #       out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
                                                   #^^^^^^^^^^^^^^^^^^^
    #       RuntimeError: FlashAttention only support fp16 and bf16 data type
    with torch.autocast(device_type=setup.device, dtype=setup.dtype):
        with profiler_wrapper(setup.device, "Forward"):
            y_pred=model.model.forward(x, model_comm_group=setup.model_comm_group, grid_shard_slice=grid_shard_slice, grid_shard_shapes=grid_shard_shapes)
            #y_pred=model.model.forward(x, setup.model_comm_group)
        #x=None
        #del x
        #torch.cuda.empty_cache()
            
        if setup.bw:
            #y=torch.rand_like(y_pred)
            with profiler_wrapper(setup.device,"Loss"):
                #loss = model.loss(y_pred, y)
                loss = dummy_loss(y_pred)
                #loss = self.loss(y_pred_full,y_full,grid_shard_slice=grid_shard_slice,group=self.model_comm_group,)
                #LOG.info(f"{y_pred.shape=}")
                #target = torch.randn(get_grid_points(setup.res), setup.channels)
                #target = torch.randn(y_pred.shape[-1], setup.channels)
                #loss_fn = torch.nn.MSELoss()
                #loss = loss_fn(y_pred, target, )
            
            with profiler_wrapper(setup.device,"Backward"):
                loss.backward()
                    
            reset_grad(x)
            
            #return (forward_time,loss_time,backward_time)
            return (0,0,0)
        else:
            return (0,0,0)
            
#nvtx wrapper function
#if a marker is given, push it
#otherwise pop it
#TODO replace with autograd
@contextmanager
def profiler_wrapper(device, marker, record_mem=False, verbose=False, torch_profiler=False):
    if verbose:
        LOG.info(marker)
    if device.startswith("cuda"):
        if record_mem:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        torch.cuda.nvtx.range_push(marker)
    start_time=time.time()
    if torch_profiler:
        record_shapes=True
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],record_shapes=record_shapes) as p:
            yield
        #LOG.info(p.key_averages(group_by_input_shape=record_shapes).table(sort_by="self_cuda_time_total", row_limit=10))
        #LOG.info(p.key_averages(group_by_input_shape=record_shapes).table(row_limit=10))
        LOG.info(p.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=30))
    else:
        yield
    end_time=time.time()
    if device.startswith("cuda"):
        torch.cuda.nvtx.range_pop()
        if record_mem:
            torch.cuda.memory._dump_snapshot(f"aifs-fw-bw.pickle")
            LOG.debug(f"Memory snapshot saved to ./aifs-fw-bw.pickle")
            #torch.cuda.memory_summary(device=device)
    #end_time-start_time
            
def compute_times(lst):
    length = len(lst)
    times=tuple(f"{sum(x) / length:.4f}" for x in zip(*lst))
    LOG.info(times)

def benchmark(models, setup, count=10, warmup=5):
    
    for model_index in range(len(models)):
        model = models[model_index]
        if setup.compile:
            torch.compile(model, dynamic=False)
        LOG.info(f"Benchmarking model {model_index}...")
    
        #Do warmup iters
        start_time=time.time()
        with profiler_wrapper(setup.device, "Warmup"):
            for _ in range(0,warmup):
                iter(model, setup)
        torch.cuda.empty_cache()
        warmup_finish_time=time.time()
        if warmup > 0:
            LOG.info(f"{warmup} warmup iterations completed in {warmup_finish_time-start_time:.2f}s")
            
        #Do the main iters
        times=list(range(count))
        with profiler_wrapper(setup.device, f"Benchmark", record_mem=setup.mem_snapshot, torch_profiler=setup.torch_profiler):
            for i in range(0,count):
                with profiler_wrapper(setup.device, f"iter {i}"):
                    times[i]=iter(model,setup)
        bm_finish_time=time.time()
        LOG.info(f"{count} iterations completed in {bm_finish_time - warmup_finish_time:.2f}s")
        
        compute_times(times)
            #LOG.info(torch.cuda.memory_summary(device=setup.device))
        
    
class Setup:
    def __init__(self, res, dtype=torch.float16, device="cuda:0", bw=True, mem_snapshot=False, config_path="config/", config_name="fw-bw", channels=128, torch_profiler=True, compile=True) -> None:
        self.res = res
        self.dtype = dtype
        self.device = device
        self.bw=bw
        self.mem_snapshot=mem_snapshot #has a slight perf impact (4.79s vs 5.29s for 10 n320 FW passes)
        self.config_path=config_path
        self.config_name=config_name
        self.channels=channels
        self.torch_profiler=torch_profiler
        self.compile=compile

        #init parallel
        if self.device != "cuda":
            raise ValueError("device=Cuda hardcoded in init_parallel")
        self.model_comm_group, self.global_rank, self.world_size, self.procs_per_node, self.num_nodes, self.local_rank = init_parallel()

        self.mem_snapshot = self.mem_snapshot and (self.global_rank == 0)

        #set device properly if running on cuda
        if self.device == "cuda":
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(torch.device(self.device))
            
    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group(group=dist.group.WORLD) #prevent warning about proc group not being destroyed
    
    def __str__(self) -> str:
        return f"Benchmarking setup:\n\t{self.res=}\n\t{self.dtype=}\n\t{self.device=}\n\t{self.bw=}\n\t{self.mem_snapshot=}\n\t{self.procs_per_node=}\n\t{self.num_nodes=}\n\t{self.channels=}\n\t{self.torch_profiler=}\n\t{self.compile}"
        
#Assumes each GPU is in a model comm group
def init_parallel():
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    #WARNING the below syntax for SLURM NTASKS isnt constant, on JEDI it was '4,'
    procs_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", '1').split('(')[0]) #in the form "NTASKS(xNNODES),"
    num_nodes= world_size//procs_per_node
    
    if world_size > 1:
        master_port="11221"
        try:
            slurm_nodelist = os.environ.get("SLURM_NODELIST", "")
            result = subprocess.run(
                    ["scontrol", "show", "hostname", slurm_nodelist], stdout=subprocess.PIPE, text=True, check=True
                )
            master_addr = result.stdout.splitlines()[0]
        except subprocess.CalledProcessError as err:
            master_addr="localhost"
        #world_size=world_size, rank=global_rank, device_id=local_rank
        dist.init_process_group(backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", world_size=world_size, rank=global_rank, device_id=torch.device(f"cuda:{local_rank}"))
        model_comm_group_ranks = np.arange(world_size, dtype=int)
        model_comm_group = dist.new_group(model_comm_group_ranks)
    else:
        model_comm_group=None
    
    return model_comm_group, global_rank, world_size, procs_per_node, num_nodes, local_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default="")
    parser.add_argument('-r', '--res', default="o1280")
    parser.add_argument('-C', '--channels', default=128, type=int)
    parser.add_argument('-f','--forward', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    setup=Setup(res="o1280", dtype=torch.float16, device="cuda", bw=(not args.forward), mem_snapshot=False, channels=args.channels, torch_profiler=False)
    LOG.info(str(setup))
    
    model=parse_inputs(args, device=setup.device) #optionally load model from checkpoint if given
    if model is None:
        model = build_model(setup)
    
    benchmark([model],setup)

    
if __name__ == "__main__":
    main()
    
#TODO
#   fix this error running o1280 over 4 procs:
#       'RuntimeError: mat1 and mat2 shapes cannot be multiplied (6599680x212 and 210x256)'
#   remove dependancy on datasets and graphs
#   num_vars changes depending on the dataset, find a way to pass this info

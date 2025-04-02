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
from pathlib import Path

#for building models
from anemoi.models.interface import AnemoiModelInterface


#for bulding from config :(
from hydra import compose, initialize
from omegaconf import OmegaConf
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.schemas.base_schema import BaseSchema, UnvalidatedBaseSchema, convert_to_omegaconf
from hydra.utils import instantiate

#loss
from anemoi.training.train.forecaster import GraphForecaster

log_level=logging.INFO
if (int(os.getenv("RANK", "0")) != 0) or (int(os.getenv("SLURM_PROCID", "0")) != 0):
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

#n320 has to be 100
#o1280 has to 99
#TODO get the number of vars from the dataset
def generate_inputs(res,device,shape=None,vars=100,batch=1,time=2,ensemble=1,grad=True,dtype=torch.float32,world_size=1, generator=None):
    #  x = batch[:, 0 : self.multi_step, None, ...]  #from predict_step
    # Preparing input tensor with shape (2, 99, 6599680)
    #batch time ensemble grid vars
    if res == "o1280":
        vars=99
    gridpoints=get_grid_points(res)//world_size
    if shape is None:
        shape=(batch,time,ensemble,gridpoints,vars)
    #if generator is not None:
    input=torch.randn(shape,dtype=dtype, device=device, requires_grad=grad, generator=generator)
    return input

#TODO replace this
def get_dataset(res):
    path="inputs/"
    if res == "n320":
        return path, "dummy-n320.zarr"
    elif res == "o1280":
        return path, "dummy-o1280.zarr"
    elif res == "o2560":
        return path, "dummy-o2560.zarr"
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
        
    LOG.debug(f"{config=}")
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

def get_graph_data(config, input_res="n320"):
    try:
        #graphtransformer
        hidden_res=config.graph.nodes.hidden.node_builder.resolution
    except:
        hidden_res=config.graph.nodes.hidden.node_builder.grid
    graph_filename = Path(f"inputs/{input_res}_{hidden_res}.graph")
    if graph_filename.exists():
        graph=torch.load(graph_filename, weights_only=False)
        return graph
    else:
        LOG.info(f"'{graph_filename}' not found. Building it...")
        from anemoi.graphs.create import GraphCreator

        graph_config = convert_to_omegaconf(config).graph
        #TODO could have race con if running this in torch distributed
        return GraphCreator(config=graph_config).create(
                save_path=graph_filename,
                overwrite=False,
            )

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
    if setup.check_correctness: #in case there's some rng in the model init phase
        setup.reset_rng()
    LOG.info(f"Building model based on '{setup.config_path}{setup.config_name}.yaml'...")
    config=build_config(setup)
    
    graph_data=get_graph_data(config, input_res=res)
    datamodule = AnemoiDatasetsDataModule(config, graph_data) #need training just for this
    
    #keep cpu on cpu until we need it on device
    model=AnemoiModelInterface(config=config, graph_data=graph_data, statistics=datamodule.statistics, data_indices=datamodule.data_indices, metadata=datamodule.metadata).to("cpu")
   
    if setup.model_comm_group is not None: 
        dist.barrier(setup.model_comm_group)
        
    LOG.info(f"Model built in {time.time()-start_time:.2f}s.")
    
    model.loss = get_loss(config, datamodule.data_indices, device)
    
    #needed for sharded batch
    model.grid_indices = instantiate(
            config.model_dump(by_alias=True).dataloader.grid_indices,
            reader_group_size=setup.world_size,)
    model.grid_indices.setup(graph_data) # need a loaded graph here
    model.name = f"{setup.config_name}.yaml"
    
    return model
    
def iter(model,setup, verbose=False, generator=None):

    with profiler_wrapper(setup.device, "Generate inputs", model_name=model.name):
        x = generate_inputs(res=setup.res,device=setup.device, grad=setup.bw, dtype=setup.dtype, world_size=setup.world_size, generator=generator)
    grid_shard_shapes = model.grid_indices.shard_shapes
    grid_shard_slice = model.grid_indices.get_shard_indices(setup.global_rank)
    
    #without torch.autocast(dtype=torch.float16) I got an error in FW pass
    #     File "/perm/naco/venvs/aifs-fw-bw/lib/python3.11/site-packages/flash_attn/flash_attn_interface.py", line 96, in _flash_attn_forward
    #       out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
                                                   #^^^^^^^^^^^^^^^^^^^
    #       RuntimeError: FlashAttention only support fp16 and bf16 data type
    with torch.autocast(device_type=setup.device, dtype=setup.dtype):
        with profiler_wrapper(setup.device, "Forward", model_name=model.name):
            y_pred=model.model.forward(x, model_comm_group=setup.model_comm_group, grid_shard_slice=grid_shard_slice, grid_shard_shapes=grid_shard_shapes)
            
        if setup.bw:
            with profiler_wrapper(setup.device,"Loss", model_name=model.name):
                loss = dummy_loss(y_pred)
            
            with profiler_wrapper(setup.device,"Backward", model_name=model.name):
                loss.backward()
            
            grad=x.grad
            
            return (y_pred,loss,grad)
        else:
            return (y_pred,0,0)
            
#nvtx wrapper function
#if a marker is given, push it
@contextmanager
def profiler_wrapper(device, marker, record_mem=False, verbose=False, torch_profiler=False, mem_summary=False, model_name=""):
    if model_name != "":
        marker = f"{model_name} - {marker}"
    if verbose:
        LOG.info(marker)
    if device.startswith("cuda"):
        if record_mem:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        torch.cuda.nvtx.range_push(marker)
    if torch_profiler:
        record_shapes=True
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],record_shapes=record_shapes) as p:
            yield
        LOG.info(p.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=30))
    else:
        yield
    if mem_summary:
        LOG.info(torch.cuda.memory_summary(device=device, abbreviated=True))
    if device.startswith("cuda"):
        torch.cuda.nvtx.range_pop()
        if record_mem:
            output_file=f"aifs-fw-bw.pickle"
            if model_name != "":
                output_file=f"aifs-fw-bw.{model_name}.pickle"
            if os.path.exists(output_file):
                os.remove(output_file)
            torch.cuda.memory._dump_snapshot(output_file)
            LOG.info(f"Memory snapshot saved to ./{output_file}")
            torch.cuda.memory._record_memory_history(enabled=None) #disable recording memory history
            
    #TODO make checking correctness not awful
    #check correctness has a performance penalty bc of sync copying the output tensors to cpu, and allocating pinned mem cpu tensors during BM iter 0
    #also increases memory usage (not clear from where, results should be on CPU)
    #further reduction in performance and increase in mem usage from deterministic algorithms
    
def benchmark(models, setup, count=10, warmup=5):
    outputs=[]
    #generator=torch.Generator(device=setup.device)
    for model_index in range(len(models)):
        
        if setup.check_correctness:
            setup.reset_rng()
            output = Output(count,setup.dtype) #Storing output reduces performance and increases memory pressure
            if len(models) != 2:
                raise ValueError("Error! correctness checking requested. please rerun with 2 configs provided")
            
        model = models[model_index].to(setup.device)
        if setup.compile:
            model=torch.compile(model, dynamic=False)
        LOG.info(f"Benchmarking model {model_index}: '{model.name}'...")
    
        #Do warmup iters
        start_time=time.time()
        with profiler_wrapper(setup.device, "Warmup", model_name=model.name):
            for _ in range(0,warmup):
                iter(model, setup)
        torch.cuda.empty_cache()
        warmup_finish_time=time.time()
        if warmup > 0:
            LOG.info(f"{warmup} warmup iterations completed in {warmup_finish_time-start_time:.2f}s")
            
        #Do the main iters
        with profiler_wrapper(setup.device, f"Benchmark", record_mem=setup.mem_snapshot, torch_profiler=setup.torch_profiler, mem_summary=True, model_name=model.name):
            for i in range(0,count):
                with profiler_wrapper(setup.device, f"iter {i}", model_name=model.name):
                    results = iter(model,setup)
                    
                if setup.check_correctness:
                    output.record(i, *results)
                    #runtime goes from 18s to 25s with emptying the cache each iter
                    torch.cuda.empty_cache() #big perf hit but for some reason my mem pressure has increased when checking correctness
                    #but it gets it through the correctness checks
                else:
                    results = None

        bm_finish_time=time.time()
        LOG.info(f"{count} iterations completed in {bm_finish_time - warmup_finish_time:.2f}s")
        
        model=model.to("cpu", non_blocking=True) #need this to ensure no mem leak from multiple models
        torch.cuda.empty_cache()
        #TODO ensure I dont have a memory leak after benchmarking a model
        #There is a mem leak...
        if setup.check_correctness:
            outputs.append(output)
    
    if setup.check_correctness: 
        #LOG.info(f"{outputs[0] == outputs[1]=}")
        LOG.info(f"Checking correctness... (on the CPU, so relatively slow)")
        outputs[0].compare(outputs[1])
        
class Output:
    def __init__(self, len,dtype):
        self.len=len
        self.y_pred=None
        self.loss=None
        self.grad=None
        
    def __eq__(self, other):
        y_pred_matches=torch.allclose(self.y_pred, other.y_pred)
        loss_matches=torch.allclose(self.loss, other.loss)
        grad_matches=torch.allclose(self.grad, other.grad)
        return y_pred_matches and loss_matches and grad_matches
    
    def compare(self, other):
        y_pred_matches=torch.allclose(self.y_pred, other.y_pred)
        loss_matches=torch.allclose(self.loss, other.loss)
        grad_matches=torch.allclose(self.grad, other.grad)
        if not grad_matches:
            grad_absdiff = torch.abs(self.grad) - torch.abs(other.grad)
            #vmap_allclose = torch.vmap(torch.allclose)
            #grad_matches_per_step = vmap_allclose(self.grad, other.grad)
            LOG.info(f"{torch.max(grad_absdiff)=}")
        LOG.info(f"{y_pred_matches=}, {loss_matches=}, {grad_matches=}")
    
    def record(self, i, y_pred,loss,grad):
        #TODO remove pinned memory allocation from the benchmark path
        if self.y_pred is None:
            self.y_pred = torch.empty((self.len,*y_pred.shape), dtype=y_pred.dtype, device="cpu", pin_memory=True)
        if self.loss is None:
            self.loss = torch.empty((self.len,*loss.shape), dtype=loss.dtype, device="cpu", pin_memory=True)
        if self.grad is None:
            self.grad = torch.empty((self.len,*grad.shape), dtype=grad.dtype, device="cpu", pin_memory=True)
        self.y_pred[i]=y_pred
        self.loss[i]=loss
        self.grad[i]=grad

class Setup:
    def __init__(self, res, dtype=torch.float16, device="cuda:0", bw=True, mem_snapshot=False, config_path="config/", configs="hackathon", channels=128, torch_profiler=True, compile=False, slurm=False, check_correctness=False, seed=None) -> None:
        self.res = res
        self.dtype = dtype
        self.device = device
        self.bw=bw
        self.mem_snapshot=mem_snapshot #has a slight perf impact (4.79s vs 5.29s for 10 n320 FW passes)
        self.config_path=config_path
        self.configs=configs.split(",")
        self.channels=channels
        self.torch_profiler=torch_profiler
        self.compile=compile
        self.check_correctness=check_correctness
        self.slurm=slurm
        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed=seed
        
        if self.check_correctness:
            #:16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB).
            #I OOM when i run with torch.use_deterministic_algorithms(True) and :4096:8
            #You should always use :16:8, perf is bad anyway but you dont OOM
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True) 
            self.reset_rng()

        else:
            if os.getenv("CUBLAS_WORKSPACE_CONFIG", "0") == ":16:8":
                raise ValueError("Error! You are running with correctness checks disabled, but 'CUBLAS_WORKSPACE_CONFIG=:16:8'. This will decrease performance. Please unset and rerun.")

        #init parallel
        if self.device != "cuda":
            raise ValueError("device=Cuda hardcoded in init_parallel")
        self.model_comm_group, self.global_rank, self.world_size, self.procs_per_node, self.num_nodes, self.local_rank = init_parallel(self.slurm)

        self.mem_snapshot = self.mem_snapshot and (self.global_rank == 0)

        #set device properly if running on cuda
        if self.device == "cuda":
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(torch.device(self.device))
            
    def reset_rng(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
            
    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group(group=dist.group.WORLD) #prevent warning about proc group not being destroyed
    
    def __str__(self) -> str:
        correctness_warning="Correctness checking enabled! performance will suffer and memory usage will increase because of this"
        setup_str=f"Benchmarking setup:\n\t{self.res=}\n\t{self.dtype=}\n\t{self.device=}\n\t{self.bw=}\n\t{self.mem_snapshot=}\n\t{self.procs_per_node=}\n\t{self.num_nodes=}\n\t{self.channels=}\n\t{self.torch_profiler=}\n\t{self.compile=}\n\t{self.configs=}\n\t{self.check_correctness=}\n\t{self.slurm=}\n\t{self.seed=}"
        if self.check_correctness:
            return f"{setup_str}\n{correctness_warning}"
        else:
            return setup_str
        
#Assumes each GPU is in a model comm group
def init_parallel(use_slurm=False):
    if use_slurm:
        global_rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        procs_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", '1').split('(')[0]) #in the form "NTASKS(xNNODES),"
    else:
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        procs_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", '1'))
    num_nodes= world_size//procs_per_node
    
    if world_size > 1:
        if use_slurm:
            master_port="11221"
            try:
                slurm_nodelist = os.environ.get("SLURM_NODELIST", "")
                result = subprocess.run(
                        ["scontrol", "show", "hostname", slurm_nodelist], stdout=subprocess.PIPE, text=True, check=True
                    )
                master_addr = result.stdout.splitlines()[0]
            except subprocess.CalledProcessError as err:
                master_addr="localhost"

            #TODO remove hardcoded cuda here
            dist.init_process_group(backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", world_size=world_size, rank=global_rank, device_id=torch.device(f"cuda:{local_rank}"))
        else:
            dist.init_process_group(backend="nccl",device_id=torch.device(f"cuda:{local_rank}"))
        model_comm_group_ranks = np.arange(world_size, dtype=int)
        model_comm_group = dist.new_group(model_comm_group_ranks)
    else:
        model_comm_group=None
    
    return model_comm_group, global_rank, world_size, procs_per_node, num_nodes, local_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="")
    parser.add_argument('--slurm',action=argparse.BooleanOptionalAction)
    parser.add_argument('-r', '--res', default="o1280")
    parser.add_argument('-C', '--channels', default=128, type=int)
    parser.add_argument('-f','--forward', action=argparse.BooleanOptionalAction)
    parser.add_argument('-m','--mem-snapshot', action=argparse.BooleanOptionalAction)
    parser.add_argument('-c', '--configs', default="aifs-fw-bw", type=str)
    parser.add_argument('-v', '--verify', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    setup=Setup(res=args.res, dtype=torch.float16, device="cuda", bw=(not args.forward), mem_snapshot=args.mem_snapshot, channels=args.channels, torch_profiler=False, configs=args.configs, check_correctness=args.verify, slurm=args.slurm)
    LOG.info(str(setup))
    
    models=[]
    for config in setup.configs:
        setup.config_name=config #need this in some places
    
        model=parse_inputs(args, device=setup.device) #optionally load model from checkpoint if given
        if model is None:
            model = build_model(setup)
        models.append(model)
    
    #model1=model
    #models.append(model1)
    
    benchmark(models,setup)
    
if __name__ == "__main__":
    main()
    
#TODO
#   remove dependancy on datasets and graphs
#   num_vars changes depending on the dataset, find a way to pass this info
#   should I make 1 mem snapshow with all models on it?

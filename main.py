import argparse
import torch
import pprint
import time

#for building models
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict


#for bulding from config :(
from hydra import compose, initialize
from omegaconf import OmegaConf
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.schemas.base_schema import BaseSchema, UnvalidatedBaseSchema
from hydra.utils import instantiate

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
        print(f"Loading {args.checkpoint}...")
        model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device)
        print(f"Checkpoint loaded.")
        
    return model

#(1,2,1,'n320',99) gave an error in precproc, so changed to 100 vars
#       return F.linear(input, self.weight, self.bias)
#   RuntimeError: mat1 and mat2 shapes cannot be multiplied (542080x309 and 212x1024)
def generate_inputs(res,device,shape=None,vars=100,batch=1,time=2,ensemble=1,grad=True,dtype=torch.float32):
    #  x = batch[:, 0 : self.multi_step, None, ...]  #from predict_step
    # Preparing input tensor with shape (2, 99, 6599680)
    #batch time ensemble grid vars
    gridpoints=get_grid_points(res)
    if shape is None:
        shape=(batch,time,ensemble,gridpoints,vars)
    input=torch.randn(shape,dtype=dtype, device=device, requires_grad=grad)
    return input

def build_config(config_path="config", config_name="fw-bw"):
    with initialize(version_base=None, config_path=config_path, job_name="debug"):
        #ignore lots of missing values in the config we dont care about
        #TODO would be nicer if hydra could chill and be lazy about MissingMandatoryValues
        hardware_paths_data="/home/mlx/ai-ml/datasets/"
        hardware_files_dataset="aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v4.zarr"
        ignore_list=[f"hardware.paths.data='{hardware_paths_data}'", f"hardware.files.dataset='{hardware_files_dataset}'", "diagnostics.log.wandb.entity=''", "diagnostics.log.mlflow.tracking_uri=''", "hardware.paths.output=''", "hardware.files.graph=''"]
        config = compose(config_name=config_name, overrides=ignore_list)
    #config = OmegaConf.to_object(config)
    #pprint.pp(config)
    #config=DotDict(**config) #has to be baseschema bc DataModule calls model_dump
    config=UnvalidatedBaseSchema(**config) #using Baseschema instantiaes all the objects early for some reason
    

    return config

def get_graph_data(res="n320"):
    graph=torch.load(f"graphs/{res}.graph", weights_only=False)
    #pprint.pp(graph)
    return graph

def build_model(setup):
    res=setup.res
    device=setup.device
    start_time=time.time()
    config_path="config/"
    config='fw-bw'
    print(f"Building model based on '{config_path}{config}.yaml'...")
    config=build_config(config_name=config, config_path=config_path)
    
    graph_data=get_graph_data(res)
    datamodule = AnemoiDatasetsDataModule(config, graph_data) #need training just for this
    
    #brings in anemoi training dep
    #I am not opposed to this, but it means I have to do preproc etc
    model=AnemoiModelInterface(config=config, graph_data=graph_data, statistics=datamodule.statistics, data_indices=datamodule.data_indices, metadata=datamodule.metadata).to(device)
    
    print(f"Model built in {time.time()-start_time:.2f}s.")
    return model
    
def iter(model,setup):
    inputs = generate_inputs(res=setup.res,device=setup.device, grad=setup.bw, dtype=setup.dtype)

    #without torch.autocast I got an error in FW pass
    #     File "/perm/naco/venvs/aifs-fw-bw/lib/python3.11/site-packages/flash_attn/flash_attn_interface.py", line 96, in _flash_attn_forward
    #       out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
                                                   #^^^^^^^^^^^^^^^^^^^
    #       RuntimeError: FlashAttention only support fp16 and bf16 data type
    with torch.autocast(device_type=setup.device, dtype=torch.float16):
        y_pred=model.model.forward(inputs)
        if setup.bw:
            raise ValueError("BW pass not yet implemented")
        
def benchmark(model, setup, count=10, warmup=5):
    start_time=time.time()

    #Do warmup iters
    for i in range(0,warmup):
        iter(model, setup)
    warmup_finish_time=time.time()
    if warmup > 0:
        print(f"{warmup} warmup iterations completed in {warmup_finish_time-start_time:.2f}s")
        
    #Do the main iters
    for i in range(0,count):
        iter(model,setup)
    bm_finish_time=time.time()
    print(f"{count} iterations completed in {bm_finish_time - warmup_finish_time:.2f}s")
        
    
class Setup:
    def __init__(self, res, dtype=torch.float16, device="cuda:0", bw=True) -> None:
        self.res = res
        self.dtype = dtype
        self.device = device
        self.bw=bw
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default="")
    args = parser.parse_args()
    
    setup=Setup(res="n320", dtype=torch.float16, device="cuda:0", bw=False)
    
    model=parse_inputs(args, device=setup.device)
    if model is None:
        model = build_model(setup)
    
    benchmark(model,setup)

    
if __name__ == "__main__":
    main()
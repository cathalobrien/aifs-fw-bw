import argparse
import torch
import pprint

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

def generate_inputs(res,device,vars=99,shape=None,batch=1,time=2,ensemble=1,grad=True,dtype=torch.float32):
    #  x = batch[:, 0 : self.multi_step, None, ...]  #from predict_step
    # Preparing input tensor with shape (2, 99, 6599680)
    #batch time ensemble grid vars
    gridpoints=get_grid_points(res)
    if shape is None:
        shape=(batch,time,ensemble,gridpoints,vars)
    input=torch.randn(shape,dtype=dtype, device=device, requires_grad=grad)
    return input
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default="")
    args = parser.parse_args()
    
    device="cuda:0"
    
    maybe_model=parse_inputs(args, device=device)
    if maybe_model is None:
        raise ValueError("Error, please provide a model checkpoint")
    model = maybe_model
    #pprint.pp(model)

    inputs = generate_inputs(res="o1280",device=device, grad=True)
    output=model.model.forward(inputs)
    #model.
    #model.predict
    
if __name__ == "__main__":
    main()
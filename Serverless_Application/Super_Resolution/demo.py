import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import pandas as pd

parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--vgpu", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

sm = int(opt.vgpu)
batch_size = 2

while batch_size <=64: 
    print("Batch Size:", batch_size)
    count = 2

    print("Number of CPUs:", os.cpu_count())
        
    pid = 0
    affinity = os.sched_getaffinity(pid)
    
    # Print the result
    print("Process is eligible to run on:", affinity)  

    

    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


    cpu_count =[2,4,8,16,32, 48]
    for count in cpu_count:
        superresolution_file = pd.read_csv('resolution_result_1.csv')
        cpu_info, load_model, prepare_data, run_inference, total_time ,cpu_count, batch_size_list, inference_per_image= [], [], [], [], [], [], [], []

        
        start_total_time = time.time()

        start_cpu_set_affinity = time.time()

        affinity_mask = set(range(0,count))
        pid = 0
        os.sched_setaffinity(0, affinity_mask)
        print("CPU affinity mask is modified for process id % s" % pid) 

        pid = 0
        affinity = os.sched_getaffinity(pid)
        
        # Print the result
        print("Now, process is eligible to run on:", affinity)

        end_cpu_set_affinity = time.time()


        start_load_model = time.time()
        model = torch.load(opt.model)["model"]

        end_load_model = time.time()

        start_prepare_data = time.time()
        im_gt = sio.loadmat("testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_gt']
        im_b = sio.loadmat("testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_b']
        im_l = sio.loadmat("testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_l']
                
        im_gt = im_gt.astype(float).astype(np.uint8)
        im_b = im_b.astype(float).astype(np.uint8)
        im_l = im_l.astype(float).astype(np.uint8)      

        im_input = im_l.astype(np.float32).transpose(2,0,1)
        im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])

        im_input = Variable((torch.from_numpy(im_input/255.).float()))
        print("input ", im_input.size())

        im_input = torch.cat((im_input, )*batch_size, axis =0)

        print("input ", im_input.size())

        end_prepare_data = time.time()

        total_inference_time = 0
        infernce_run = min(10, 128// batch_size)

        for i in range(infernce_run):
            start_run_inference = time.time()
            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()
                
            start_time = time.time()
            out = model(im_input)
            elapsed_time = time.time() - start_time

            out = out.cpu()
            end_run_inference = time.time()
            total_inference_time += (end_run_inference - start_run_inference)
        total_inference_time = total_inference_time/infernce_run

        im_h = out.data[0].numpy().astype(np.float32)

        im_h = im_h*255.
        im_h[im_h<0] = 0
        im_h[im_h>255.] = 255.            
        im_h = im_h.transpose(1,2,0)

        print("Dataset=",opt.dataset)
        print("Scale=",opt.scale)
        print("It takes {}s for processing".format(elapsed_time))

        end_total_time = time.time()
        cpu_info.append(end_cpu_set_affinity-start_cpu_set_affinity)
        load_model.append(end_load_model-start_load_model)
        prepare_data.append(end_prepare_data-start_prepare_data)
        run_inference.append(total_inference_time)
        inference_per_image.append(total_inference_time/batch_size)

        total_time.append(end_total_time-start_total_time)
        cpu_count.append(count)
        batch_size_list.append(batch_size)
        

    
        result = pd.DataFrame(list(zip(cpu_count,cpu_info, load_model, prepare_data, run_inference,inference_per_image,  total_time, batch_size_list)),
        columns=['Number of CPU','Check available CPU and limit' ,'Load Model','Prepare data', 'Run Inference','Inference per Image','Total Time', 'Batch size'])

        result["Number of SM"] = 72-sm
        print(superresolution_file)
        result = pd.concat([superresolution_file,result])
        print(result)
        result.to_csv('resolution_result_1.csv', index = False)
    
    batch_size = batch_size * 2
        

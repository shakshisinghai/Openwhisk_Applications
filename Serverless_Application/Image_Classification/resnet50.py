import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
import os
import pandas as pd
import time
import sys



sm = int(sys.argv[1])
batch_size = 2

# Change the CPU affinity mask
while batch_size <=1024: 
    print("Batch Size:", batch_size)
    count = 2

    print("Number of CPUs:", os.cpu_count())
        
    pid = 0
    affinity = os.sched_getaffinity(pid)
    
    # Print the result
    print("Process is eligible to run on:", affinity)  
    cpu_count =[2,4,8,16,32, 48]
    for count in cpu_count:
        resnet_file = pd.read_csv('classification_result_1.csv')
        cpu_info, load_model, prepare_data, run_inference, total_time ,cpu_count, batch_size_list, inference_per_image= [], [], [], [], [], [], [], []

        resnet50=0
        start_total_time = time.time()

        start_cpu_set_affinity = time.time()
        # Python program to explain os.sched_setaffinity() method  
        
        # Get the number of CPUs in the system
           
        affinity_mask = set(range(0,count))
        pid = 0
        os.sched_setaffinity(0, affinity_mask)
        print("CPU affinity mask is modified for process id % s" % pid) 

        pid = 0
        affinity = os.sched_getaffinity(pid)
        
        # Print the result
        print("Now, process is eligible to run on:", affinity)

        end_cpu_set_affinity = time.time()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Using {device} for inference')

        start_load_model = time.time()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

        resnet50.eval().to(device)
        end_load_model = time.time()

        start_prepare_data = time.time()
        uris = [
            'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
        ]

        batch = torch.cat(
            [utils.prepare_input_from_uri(uris[0]),] * batch_size
        ).to(device)

    
        end_prepare_data = time.time()
        
        total_inference_time = 0
        infernce_run = min(10, 1024// batch_size)

        with torch.no_grad():
                output = torch.nn.functional.softmax(resnet50(batch), dim=1)
                
        results = utils.pick_n_best(predictions=output, n=5)

        for i in range(infernce_run):
            start_run_inference = time.time()
            with torch.no_grad():
                output = torch.nn.functional.softmax(resnet50(batch), dim=1)
                
            results = utils.pick_n_best(predictions=output, n=5)

            end_run_inference = time.time()
            total_inference_time += (end_run_inference - start_run_inference)
        total_inference_time = total_inference_time/infernce_run
            


        for uri, result in zip(uris, results):
            img = Image.open(requests.get(uri, stream=True).raw)
            #img.thumbnail((256,256), Image.ANTIALIAS)
            #plt.imshow(img)
            #plt.show()
            #print(result)

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
        print(resnet_file)
        result = pd.concat([resnet_file,result])
        print(result)
        result.to_csv('classification_result_1.csv', index = False)
    
    batch_size = batch_size * 2
        
        
        



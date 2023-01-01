import torch
import sys
import os 
import pandas as pd
import time

sm = int(sys.argv[1])
batch_size = 2

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

        yolo_file = pd.read_csv('yolo_result_1.csv')
        cpu_info, load_model, prepare_data, run_inference, total_time ,cpu_count, batch_size_list, inference_per_image= [], [], [], [], [], [], [], []

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

        # Model
        start_load_model = time.time()
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        end_load_model = time.time()
        print("Model Loaded")

        # Images
        start_prepare_data = time.time()
        imgs = ['https://ultralytics.com/images/zidane.jpg']* batch_size  # batch of images
        end_prepare_data = time.time()
        print("Data Prepared")

        print("First Inference time is not takein into account. Therefore total time is more.")
        total_inference_time = 0
        infernce_run = min(5, 1024// batch_size)
        results = model(imgs)

        for i in range(infernce_run):
            start_run_inference = time.time()
            # Inference
            results = model(imgs)

            end_run_inference = time.time()
            total_inference_time += (end_run_inference - start_run_inference)
        total_inference_time = total_inference_time/infernce_run

        print("Inference completed")

        # Results
        results.print()
        results.save()  # or .show()

        results.xyxy[0]  # img1 predictions (tensor)
        results.pandas().xyxy[0]  # img1 predictions (pandas)
        
        
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
        print(yolo_file)
        result = pd.concat([yolo_file,result])
        print(result)
        result.to_csv('yolo_result_1.csv', index = False)
    
    batch_size = batch_size * 2
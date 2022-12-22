import torch
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import sys
cpu_info, load_model, prepare_data, run_inference, total_time,cpu_count = [],[], [], [], [], []
count = 2


batch_size = 2
sm = int(sys.argv[1])

while batch_size <32: 
    print("Batch Size:", batch_size)
    count = 2

    print("Number of CPUs:", os.cpu_count())
        
    pid = 0
    affinity = os.sched_getaffinity(pid)
    
    # Print the result
    print("Process is eligible to run on:", affinity)  
    cpu_count =[2,4,8,16,32, 48]

    for count in cpu_count:
        segment_file = pd.read_csv('segment_result_1.csv')
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
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        model.eval()
        end_load_model = time.time()
        # Download an example image from the pytorch website
        
        print("Loaded Model")
        start_prepare_data = time.time()
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
        try: urllib.URLopener().retrieve(url, filename)
        except: urllib.request.urlretrieve(url, filename)

        # sample execution (requires torchvision)

        input_image = Image.open(filename)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        print(input_image)
        print(input_tensor.size())
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        input_batch = torch.cat((input_batch,)* batch_size, dim = 0)
        print(input_batch.size())
        print("image_preprocessing done")
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        end_prepare_data = time.time()

        infernce_run = min(10, 32// batch_size)
        total_inference_time = 0
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        for i in range(infernce_run):
            start_run_inference = time.time()
            with torch.no_grad():
                output = model(input_batch)['out'][0]
            output_predictions = output.argmax(0)
            end_run_inference = time.time()

            total_inference_time += (end_run_inference - start_run_inference)

        total_inference_time = total_inference_time/infernce_run

        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)
        
        plt.imshow(r)
        # plt.show()
        print("end")
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


        result = pd.concat([segment_file,result])

        print(result)
        result.to_csv('segment_result_1.csv', index = False)
    
    batch_size = batch_size * 2


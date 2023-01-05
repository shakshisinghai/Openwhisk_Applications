import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from ssim import SSIM
from PIL import Image
import pandas as pd

sm = 7
batch_size = 2


if __name__ == '__main__':
	start = time.process_time()
	opt = TestOptions().parse()
	opt.nThreads = 1   # test code only supports nThreads = 1
	opt.batchSize = 1  # test code only supports batchSize = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip

	while batch_size <= 1024:
		print("Batch Size:", batch_size)
		count = 2

		print("Number of CPUs:", os.cpu_count())

		pid = 0
		affinity = os.sched_getaffinity(pid)

		# Print the result
		print("Process is eligible to run on:", affinity)

		cpu_count = [2, 4, 8, 16]
		for count in cpu_count:
			deblur_file = pd.read_csv('deblur_result_1.csv')
			cpu_info, load_model, prepare_data, run_inference, total_time, cpu_count, batch_size_list, inference_per_image = [], [], [], [], [], [], [], []
			start_total_time = time.time()

			start_cpu_set_affinity = time.time()
			# Python program to explain os.sched_setaffinity() method
			# Get the number of CPUs in the system
			affinity_mask = set(range(0, count))
			pid = 0
			os.sched_setaffinity(0, affinity_mask)
			print("CPU affinity mask is modified for process id % s" % pid)

			pid = 0
			affinity = os.sched_getaffinity(pid)

			# Print the result
			print("Now, process is eligible to run on:", affinity)

			end_cpu_set_affinity = time.time()

			print("Start Data Prepare")
			start_prepare_data = time.time()
			data_loader = CreateDataLoader(opt)
			dataset = data_loader.load_data()
			end_prepare_data = time.time()

			print("Load Model")
			start_load_model = time.time()
			model = create_model(opt)
			end_load_model = time.time()
			print("Model Loaded")

			print("Inference Started")

			total_inference_time = 0
			infernce_run = min(10, 1024// batch_size)
			start = 1
			for i, data in enumerate(dataset):
				if i >= opt.how_many:
					break
				counter = i
				model.set_input(data, batch_size, start)
				model.test()
				start+=1

			for i in range(infernce_run):
				start_run_inference = time.time()
				for i, data in enumerate(dataset):
					if i >= opt.how_many:
						break
					counter = i
					model.set_input(data, batch_size, start)
					model.test()
				end_run_inference = time.time()
				total_inference_time += (end_run_inference - start_run_inference)
			total_inference_time = total_inference_time/infernce_run

			print("Inference Completed")
			end_total_time = time.time()

			cpu_info.append(end_cpu_set_affinity - start_cpu_set_affinity)
			
			load_model.append(end_load_model - start_load_model)
			prepare_data.append(end_prepare_data - start_prepare_data)
			run_inference.append(total_inference_time)
			inference_per_image.append(total_inference_time / batch_size)


			total_time.append(end_total_time - start_total_time)
			cpu_count.append(count)
			batch_size_list.append(batch_size)

			result = pd.DataFrame(list(zip(cpu_count,cpu_info, load_model, prepare_data, run_inference,inference_per_image,  total_time, batch_size_list)),
        	columns=['Number of CPU','Check available CPU and limit' ,'Load Model','Prepare data', 'Run Inference','Inference per Image','Total Time', 'Batch size'])
			
			result["Number of SM"]=sm
			print("Results we got 2:", result)
			print("File we got 2:", deblur_file)
			result = pd.concat([deblur_file, result])
			print("Results we got 3:", result)
			print("Results to put in file:", result)
			result.to_csv('deblur_result_1.csv', index=False)

		batch_size = batch_size * 2
		

	end = time.process_time()
	print("CPU TIME", end-start)


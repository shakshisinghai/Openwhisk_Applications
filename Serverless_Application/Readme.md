
Installing conda

```
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

bash Anaconda3-2022.10-Linux-x86_64.sh 
 
source /home/cc/anaconda3/bin/activate
```

Installing dependencies

```

conda env create -f environment.yml 

conda activate resnet
```

Commands to run differnet files:

```
cd Serverless_Application\Image_Classification
python resnet50.py

```


```
cd Serverless_Application\Segmentation
python segment.py
```

```
pip install tqdm
pip install opencv-python
pip install seaborn

cd Serverless_Application\Yolo_object_detection\
python yolo.py 0

```

```
cd Serverless_Application\DeblurGAN\
python test.py --dataroot C:\Users\ishus\Documents\NCSU\AWS_DL_Project\DeblurGAN\images\ --model test --dataset_mode single --learn_residual 

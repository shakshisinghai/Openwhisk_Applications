
Installing conda

```
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

bash Anaconda3-2022.10-Linux-x86_64.sh 
 
source /home/cc/anaconda3/bin/activate
```

Installing dependencies

```conda env create -f environment.yml 
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

# Application 1

## Function 1: [Image Super Resolution using Generative Adversarial Network](https://github.com/twtygqyy/pytorch-SRResNet)

**Input:** Image(in .mat format)

**Output:** Image of high Resolution

**Max batch size possible:** 200 


Test Command : 

` python eval.py --model /home/cc/Applications/super_resolution/pytorch-SRResNet/model/model_srresnet.pth  --dataset Set5 --cuda `

![image](https://user-images.githubusercontent.com/37688219/202236262-bd80ddd9-f7e9-45f2-be1f-fd448d1b8fde.png)

## Function 2: [Multi Style Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer.git)

**Input:** Image



<img src="https://user-images.githubusercontent.com/37688219/202237316-2a61d32f-e04c-4002-8c8f-caca0fcda885.png" width="300" height="300">


**Output:** Modified Image

<img src="https://user-images.githubusercontent.com/37688219/202237215-7b8e0f97-f30a-40f1-b319-119083e79b6d.png" width="300" height="300">

Max batch size possible : 4 

Test Command:

` python main.py eval --content-image images/content/sunflower.jpg --style-image images/21styles/pencil.jpg --model models/21styles.model --content-size 1024 `

## Function 3: [Image Classification] (https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/)

**Input:** Image

**Output:** Classification result as String 

**Max batch size possible:** 1024

 

## Sample Experiment

![image](https://user-images.githubusercontent.com/37688219/202245731-947a8d93-d099-418e-86d4-7ba623937125.png)

I tried running each function individually and passed the output of one function to another.


** Results:**

![image](https://user-images.githubusercontent.com/37688219/202244862-ce97f94e-1f60-4810-a1fd-c4dff56f5413.png)



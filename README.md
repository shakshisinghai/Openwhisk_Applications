# Openwhisk_Applications

## Master Node 
![image](https://user-images.githubusercontent.com/37688219/199644377-8db25125-d5fc-4c49-8a61-972c728ad67a.png)

## Worker Node 
![image](https://user-images.githubusercontent.com/37688219/199644487-d718ddea-15e4-48e0-8824-2cac4a80d667.png)


 Optional
```
sudo  rm /etc/apt/sources.list.d/cuda.list
sudo rm /etc/apt/sources.list.d/nvidia-ml.list
sudo apt-key del 7fa2af80
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update    && sudo apt-get -y install cuda-drivers
```

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update    && sudo apt-get install -y nvidia-docker2
sudo vi /etc/docker/daemon.json

{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}

```

https://pypi.org/project/torchvision/


![image](https://user-images.githubusercontent.com/37688219/200152514-c38f3091-a2ac-48e7-850d-a31e7db4c14f.png)

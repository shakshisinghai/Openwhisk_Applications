### Requirements
```
pip install validators matplotlib

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Running on Openwhisk

1. **Using [openwhisk/python3aiaction](https://hub.docker.com/r/openwhisk/python3aiaction)**

```
 docker build -t python3aiaction:resnet50 .
 docker tag python3aiaction:resnet50 ssingha2/python3aiaction:resnet
 docker push ssingha2/python3aiaction:resnet

 wsk action -i delete resnet50_ai
 wsk action -i create resnet50_ai --docker ssingha2/python3aiaction:resnet action.py
 wsk action -i invoke resnet50_ai --result

```

Tried running the code using [Creating and invoking Docker actions](https://github.com/apache/openwhisk/blob/master/docs/actions-docker.md) link

Error: 
 * Python Version : 3.5
 * Torch version: 0.4
 * AttributeError: module 'torch' has no attribute 'hub'
 
Solution : 
*  need torch >= 1.1.0 to use torch.hub attribute.


* Tried changing Torch version using below command

``` RUN pip install --upgrade torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html ```

Error on line ``` import torchvision.transforms as transforms ```: 
```Torchvision error TypeError: _resolve_type_from_object()```

Tried changing python version but Python version3.8 and 3.7 
``` Error:  Deploying with Dockerfile: error: Unable to locate package python3.8```

Tried changing python version but Python version3.9 but versio was not updating.

2. Using [python:3.9.15](https://hub.docker.com/layers/library/python/3.9.15/images/sha256-b5f024fa682187ef9305a2b5d2c4bb583bef83356259669fc80273bb2222f5ed?context=explore)
Error: 
![image](https://user-images.githubusercontent.com/37688219/200151790-3e365c37-30a5-4f07-bc66-15750e14bb0a.png)





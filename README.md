# transfer learning model with pytorch
## Usage
You can initialize network as following:
```python
!git clone https://github.com/TKeitacw/Transfer_Nets.git
from Transfer_Nets.Transfer_Net import Transfer_Net
model = Transfer_Net()
```

Please input argument by dictionary.
```python
model = Transfer_Net({"model":"vgg16_bn"})
```


## argments

model(string):

&emsp;&emsp;Framework of feature extractor. Default is VGG16.
```python
"alexnet",
"vgg11","vgg11_bn","vgg13","vgg13_bn","vgg16","vgg16_bn","vgg19","vgg19_bn",
"resnet18","resnet34","resnet50","resnet101","resnet152",
```

out(int):

&emsp;&emsp;The number of class to output. Default is 2.

Pretrained(boolean):

&emsp;&emsp;Control flag whether ImageNet pre-training is performed or not. Default is True


grad(boolean):

&emsp;&emsp;Flag to allow feature extractor training. Default is False

trainable_layer(tuple):

&emsp;&emsp;Layer types to enable training. Default is None.
```python
model = Transfer_Net({"trainable_layer":(torch.nn.modules.BatchNorm2d)})
```

FC_bias(boolean):

&emsp;&emsp;Flag to control bias of last FC layer. Default is False.

## functions
forward_withFmaps:

&emsp;&emsp;Forward function to output class probability and feature map

instance_CAMlayer:

&emsp;&emsp;Initialize one 1x1 convolution layer for making CAM.

forward_withCAM:

&emsp;&emsp;Forward function to output class probability and class activation map(CAM).
Warning!!!: please run instance_CAMlayer before running this function.

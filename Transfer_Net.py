import torch
from torch import nn
from torchvision import models

class Transfer_Net(nn.Module):
    def __init__(self, args={}):
        super().__init__()
        if "out" in args.keys():
            out = args["out"]
        else:
            out = 2
        if "grad" in args.keys():
            grad = args["grad"]
        else:
            grad = False
        if "FC_bias" in args.keys():
            FC_bias = args["FC_bias"]
        else:
            FC_bias = False
        if "pretrained" in args.keys():
            pretrained = args["pretrained"]
        else:
            pretrained = True
        if "trainable_layer" in args.keys():
            tr = args["trainable_layer"]
        else:
            tr = None
        if "model" in args.keys():
            f = args["model"]
        else:
            f = "vgg16"
        dic_resnet = {18:models.resnet18,
               34:models.resnet34,
               50:models.resnet50,
               101:models.resnet101,
               152:models.resnet152,
        }
        dic_vgg = {
            "vgg11":models.vgg11,
            "vgg11_bn":models.vgg11_bn,
            "vgg13":models.vgg13,
            "vgg13_bn":models.vgg13_bn,
            "vgg16":models.vgg16,
            "vgg16_bn":models.vgg16_bn,
            "vgg19":models.vgg19,
            "vgg19_bn":models.vgg19_bn,
        }
        if f==None:
            self.features = models.vgg16(pretrained=pretrained).features
            for param in self.features.parameters():
                param.requires_grad=grad
            for l in self.features:
                if isinstance(l, tr):
                    for param in l.parameters():
                        param.requires_grad = True
            self.classifier = torch.nn.Linear(512, out, bias=FC_bias)
        elif f == "alexnet":
            self.features = models.alexnet(pretrained=pretrained).features
            for param in self.features.parameters():
                param.requires_grad=grad
            for l in self.features:
                if isinstance(l, tr):
                    for param in l.parameters():
                        param.requires_grad = True
            self.classifier = torch.nn.Linear(256, out, bias=FC_bias)
        elif f[:3] == "vgg":
            try:
                self.features = dic_vgg[f](pretrained=pretrained).features
            except:
                print('error occured! plz check "model" argment.')
                print(list(dic_vgg.keys()), 'is available')
                return 
            for param in self.features.parameters():
                param.requires_grad=grad
            for l in self.features:
                if isinstance(l, tr):
                    for param in l.parameters():
                        param.requires_grad = True
            self.classifier = torch.nn.Linear(512, out, bias=FC_bias)
        elif f[:6] == "resnet":
            try:
                resnet = dic_resnet[int(f[6:])](pretrained=pretrained)
            except:
                print('error occured! plz check the number of layer.')
                print(list(dic_resnet.keys()), ' is available')
                return 
            modules = list(resnet.children())[:-2]
            self.features = nn.Sequential(*modules)
            for param in self.features.parameters():
                param.requires_grad=grad
            if tr is not None:
                self.__search4trainable_layer(self.features, tr)
            if int(f[6:]) > 34:
                self.classifier = torch.nn.Linear(2048, out, bias=FC_bias)
            else:
                self.classifier = torch.nn.Linear(512, out, bias=FC_bias)
        else:
            print(f, "is not supported... sorry :(")
    def __search4trainable_layer(self, features, layer_types):
        for name, layer in features._modules.items():
            if isinstance(layer, torch.nn.Sequential) or isinstance(layer, models.resnet.Bottleneck) or isinstance(layer, models.resnet.BasicBlock):
                self.__search4trainable_layer(layer, layer_types)
            elif isinstance(layer, layer_types):
                for param in layer.parameters():
                    param.requires_grad = True
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.mean([-2,-1]))
    def forward_withFmaps(self, x):
        x = self.features(x)
        return self.classifier(x.mean([-2,-1])), x
    def instance_CAMlayer(self):
        self.return_cam = torch.nn.Conv2d(self.classifier.weight.shape[1], self.classifier.weight.shape[0], 1, bias=False)
        self.return_cam.weight = torch.nn.Parameter(self.classifier.weight.unsqueeze(-1).unsqueeze(-1))
        self.return_cam.weight.requires_grad=False
    def forward_withCAM(self, x):
        x = self.features(x)
        cam = self.return_cam(x)
        return self.classifier(x.mean([-2,-1])), cam

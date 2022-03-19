import torch
import torch.nn as nn
import torchvision

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.num_classes = num_classes
        backbone = torchvision.models.efficientnet_b2()
        self.backbone = backbone
        
        eff_w1_3filt = self.backbone.features[0][0]
        eff_w1_summed = eff_w1_3filt.weight.detach().numpy().sum(axis=1)
        eff_w1_summed = eff_w1_summed.reshape((32,1,3,3))
        replace_w1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #replace_w1.weight = torch.tensor(eff_w1_summed)
        replace_w1.load_state_dict({'weight':torch.tensor(eff_w1_summed)})
        self.backbone.features[0][0] = replace_w1

        self.classifier = torch.nn.Linear(in_features=1000, out_features=num_classes, bias=True)
    
    def forward(self, x):
        out_1000 = self.backbone(x.unsqueeze(1))
        return self.classifier(out_1000)
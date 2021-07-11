import torchvision
import torchvision.models as models

import torch.nn as nn

class vgg16_tuned(nn.Module):

    def __init__(self):
        super(vgg16_tuned, self).__init__()
        # Number of classes of Data
        self.num_classes = 2
        # Load Pre-trained Model
        self.vgg16 = models.vgg16(pretrained=True, progress=True)
        # Modifying Last Layer of Model for our test case of 2 classes
        self.vgg16.classifier[6] = nn.Linear(4096,self.num_classes)

    def forward(self, input):
        return self.vgg16(input)









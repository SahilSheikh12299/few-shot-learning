from torch_snippets import *

def convBlock(ni, no):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(ni, no, kernel_size=3, padding=1, \
                  padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(no),)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(
            convBlock(1,4),
            convBlock(4,8),
            convBlock(8,8),
            nn.Flatten(),
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500), nn.ReLU(inplace=True),
            nn.Linear(500, 5)) 
    def forward(self, input1, input2):
        output1 = self.features(input1)
        output2 = self.features(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """    Contrastive loss function.Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, \
                                                 output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * \
                                      torch.pow(euclidean_distance, 2) + \
                                      (label) * torch.pow(torch.clamp( \
                                                                      self.margin - euclidean_distance, \
                                                                      min=0.0), 2))
        acc = ((euclidean_distance>0.6)==label).float().mean()
        return loss_contrastive, acc
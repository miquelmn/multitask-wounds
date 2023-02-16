import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.1

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        
        return 1. - dsc

class Biloss(nn.Module):
    
    def __init__(self):
        super(Biloss, self).__init__()
        self.loss = DiceLoss()

    def forward(self, y_pred, y_true, y_predw, y_truew):
               
        return   self.loss(y_predw, y_truew) + self.loss(y_pred, y_true) 
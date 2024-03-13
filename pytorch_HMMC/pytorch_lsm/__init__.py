import torch
import torch.nn.functional as F
import torch.nn as nn

def _lsmce(output, target, w_matrix, size_average = True):
    sco=torch.unsqueeze(target,1)
    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(output)
    c_soft_output = soft_output.gather(1, sco)
    log_CE = torch.log(c_soft_output)
    loss = sum(sum(-log_CE.t() * w_matrix))/100
    return loss

class lsmce(torch.nn.Module):
    def __init__(self, size_average = True):
        super(lsmce, self).__init__()
        self.size_average = size_average

    def forward(self, output, target, w_matrix):

        return _lsmce(output, target, w_matrix, self.size_average)


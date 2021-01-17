import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module

from ..utils.utils import expand_dim

class HomographyGridGen(Module):
    def __init__(self, out_h=256, out_w=256, device='cpu'):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.device = device

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X,requires_grad=False)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False)

        self.grid_X = self.grid_X.to(device)
        self.grid_Y = self.grid_Y.to(device)

    def forward(self, theta):
        b=theta.size(0)
        if theta.size(1)==9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0=H[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1=H[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2=H[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3=H[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4=H[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5=H[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6=H[:,6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7=H[:,7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8=H[:,8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X,0,b)
        grid_Y = expand_dim(self.grid_Y,0,b)

        grid_Xp = grid_X*h0+grid_Y*h1+h2
        grid_Yp = grid_X*h3+grid_Y*h4+h5
        k = grid_X*h6+grid_Y*h7+h8

        grid_Xp /= k; grid_Yp /= k

        return torch.cat((grid_Xp,grid_Yp),3)

def homography_mat_from_4_pts(theta):
    b=theta.size(0)
    if not theta.size()==(b,8):
        theta = theta.view(b,8)
        theta = theta.contiguous()

    xp=theta[:,:4].unsqueeze(2) ;yp=theta[:,4:].unsqueeze(2)

    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    y = Variable(torch.FloatTensor([-1,  1,-1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b,1,1)

    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()

    A=torch.cat([torch.cat([-x,-y,-o,z,z,z,x*xp,y*xp,xp],2),torch.cat([z,z,z,-x,-y,-o,x*yp,y*yp,yp],2)],1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h=torch.bmm(torch.inverse(A[:,:,:8]),-A[:,:,8].unsqueeze(2))
    # add h33
    h=torch.cat([h,single_o],1)

    H = h.squeeze(2)

    return H
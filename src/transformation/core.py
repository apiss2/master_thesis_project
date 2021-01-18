import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .tps import TpsGridGen
from .affine import AffineGridGen
from .homography import HomographyGridGen

class GeometricTnf(object):
    """
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )
    """
    def __init__(self, geometric_model='affine', padding_mode='zeros',\
        tps_grid_size=3, size=256, device='cpu'):
        self.padding_mode = padding_mode

        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h=size, out_w=size)
        elif geometric_model=='hom':
            self.gridGen = HomographyGridGen(out_h=size, out_w=size, device=device)
        elif geometric_model=='tps':
            self.gridGen = TpsGridGen(out_h=size, out_w=size, grid_size=tps_grid_size, device=device)

        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))

        self.theta_identity = self.theta_identity.to(device)

    def __call__(self, image_batch, theta_batch=None, return_warped_image=True, return_sampling_grid=False):
        if image_batch is None:
            b=1
        else:
            b=image_batch.size(0)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3).contiguous()
            theta_batch = Variable(theta_batch,requires_grad=False)

        sampling_grid = self.gridGen(theta_batch)

        if return_sampling_grid and not return_warped_image:
            return sampling_grid

        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid, align_corners=True, padding_mode=self.padding_mode)

        if return_sampling_grid and return_warped_image:
            return (warped_image_batch,sampling_grid)

        return warped_image_batch

class PointTnf(object):
    """
    Class with functions for transforming a set of points with affine/tps transformations
    """
    def __init__(self, device='cpu'):
        self.device=device
        self.tpsTnf = TpsGridGen(device=device)

    def tpsPointTnf(self,theta,points):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        points=points.unsqueeze(3).transpose(1,3)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta,points)
        # undo reshaping
        warped_points=warped_points.transpose(3,1).squeeze(3)
        return warped_points

    def affPointTnf(self,theta,points):
        theta_mat = theta.view(-1,2,3)
        warped_points = torch.bmm(theta_mat[:,:,:2],points)
        warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)
        return warped_points
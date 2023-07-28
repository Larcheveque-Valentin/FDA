# -*- coding: utf-8 -*-
"""
April - August 2023
Hyper Parameter Search for Functionnal Convolution 

@author:Valentin Larchevêque


"""

# Import modules
import inspect
import gc
import random
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from skfda.representation.basis import VectorValued as MultiBasis
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from numpy import *
import seaborn as sns
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# import skfda as fda
# from skfda import representation as representation
# from skfda.exploratory.visualization import FPCAPlot
# # from skfda.exploratory.visualization import FPCAPlot
# # from skfda.preprocessing.dim_reduction import FPCA
# # from skfda.representation.basis import BSpline, Fourier, Monomial
import scipy
from scipy.interpolate import BSpline
import os
import ignite
from tqdm import tqdm
import sklearn
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import random
from random import seed
from scipy import stats
import statistics
from statistics import stdev

import skfda
from skfda import FDataGrid as fd
from skfda.representation.basis import BSpline as B


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, std=0.005)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, std=0.005)
        torch.nn.init.constant_(m.bias.data, 0.0)



def conv_block_out(kernel_size,stride,padding,dilation,n):
    return ((n+2*padding-dilation*(kernel_size-1)-1)//stride)+1



class Smoothing_method:
    def __init__(self,knots_or_basis="knots",Mode="Smooth",basis_type="Bspline",order=3,
                 knots=np.linspace(1,12,6),period=2*pi,n_basis=13):
        self.Mode=Mode
        self.basis_type=basis_type
        # self.interpolation_order=interpolation_order
        self.order=order
        self.knots_or_basis=knots_or_basis
        self.knots=knots
        self.period=period
        self.n_basis=n_basis
    def smoothing(self):
        if 'inter' in self.Mode:
            # interpolation=skfda.representation.interpolation.SplineInterpolation(interpolation_order=self.interpolation_order)             
            smooth_meth= skfda.representation.interpolation.SplineInterpolation(interpolation_order=self.order)             
        else:
            if "knots" in self.knots_or_basis:  
                if ("spline" in self.basis_type) or ("Bsp" in self.basis_type) or ("spl" in self.basis_type):
                    smooth_meth= B(knots=self.knots,order=self.order)
                if self.basis_type=="Fourier":
                    smooth_meth=skfda.representation.basis.FourierBasis(domain_range=[min(self.knots),max(self.knots)],period=self.period)
            if "basis" in self.knots_or_basis:
                if ("spline" in self.basis_type) or ("Bsp" in self.basis_type) or ("spbasis" in self.basis_type):
                    smooth_meth= B(n_basis=self.n_basis,order=self.order)
                if ("fourier" in self.basis_type)or ("fourrier" in self.basis_type) or ("Fourier" in self.basis_type) or ("four" in self.basis_type):
                    smooth_meth=skfda.representation.basis.FourierBasis(n_basis=self.n_basis,period=self.period)
        return smooth_meth




class HyperParameters:
    def __init__(self,basis=skfda.representation.basis.VectorValuedBasis([Smoothing_method().smoothing()
    ,
    Smoothing_method().smoothing(),]),Smoothing_method=Smoothing_method(),batch_size=30, n_epochs=25, granulation=2000,
                 n_conv_in=32, n_conv_in2=512, n_conv_in3=256,n_conv_out=64, n_Flat_out=256,
                 stride_1=1, stride_2=1, stride_3=1,
                 stride_pool_1=2, stride_pool_2=2, stride_pool_3=1,
                 kernel_size_1=7, kernel_size_2=4, kernel_size_3=3,
                 kernel_size_pool_1=3, kernel_size_pool_2=3, kernel_size_pool_3=2,
                 dilation_1=1, dilation_2=1, dilation_3=1,
                 dilation_pool_1=1, dilation_pool_2=1, dilation_pool_3=1,
                 padding_1=2, padding_2=2, padding_3=2,derivative=[0],
                 padding_pool_1=0, padding_pool_2=0, padding_pool_3=0,
                 opt="Adam", lr=0.00089, loss=nn.CrossEntropyLoss(),activation=nn.Identity(),negative_slope=0.17,n_channel=1):
        self.derivative=derivative
        self.Smoothing_type=Smoothing_method.Mode
        self.Smoothing_method=Smoothing_method
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.activation=activation
        self.n_conv_out=n_conv_out
        self.n_channel=n_channel
        
        self.granulation = granulation
        self.n_conv_in = n_conv_in
        self.n_conv_in2 = n_conv_in2
        self.n_conv_in3 = n_conv_in3
        self.n_Flat_out = n_Flat_out
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.stride_3 = stride_3
        self.stride_pool_1 = stride_pool_1
        self.stride_pool_2 = stride_pool_2
        self.stride_pool_3 = stride_pool_3
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.kernel_size_3 = kernel_size_3
        self.kernel_size_pool_1 = kernel_size_pool_1
        self.kernel_size_pool_2 = kernel_size_pool_2
        self.kernel_size_pool_3 = kernel_size_pool_3
        self.dilation_1 = dilation_1
        self.dilation_2 = dilation_2
        self.dilation_3 = dilation_3
        self.dilation_pool_1 = dilation_pool_1
        self.dilation_pool_2 = dilation_pool_2
        self.dilation_pool_3 = dilation_pool_3
        self.padding_1 = padding_1
        self.padding_2 = padding_2
        self.padding_3 = padding_3
        self.padding_pool_1 = padding_pool_1
        self.padding_pool_2 = padding_pool_2
        self.padding_pool_3 = padding_pool_3
        self.opt = opt
        self.lr = lr
        self.loss = loss
        self.negative_slope=negative_slope
        if n_channel!=1:
            self.basis=basis
        else:
            self.basis=Smoothing_method.smoothing()



def conv_total_out(hyperparams=HyperParameters()):
    n=hyperparams.granulation
    stride_1 = [hyperparams.stride_1,
    hyperparams.stride_2,
    hyperparams.stride_3,]
    stride_pool_1 = [hyperparams.stride_pool_1,
    hyperparams.stride_pool_2,
    hyperparams.stride_pool_3,]
    kernel_size_1 = [hyperparams.kernel_size_1,
    hyperparams.kernel_size_2,
    hyperparams.kernel_size_3,]
    kernel_size_pool_1 = [hyperparams.kernel_size_pool_1,
    hyperparams.kernel_size_pool_2,
    hyperparams.kernel_size_pool_3,]
    dilation_1 = [hyperparams.dilation_1,
    hyperparams.dilation_2,
    hyperparams.dilation_3,]
    dilation_pool_1 = [hyperparams.dilation_pool_1,
    hyperparams.dilation_pool_2,
    hyperparams.dilation_pool_3,]
    # basis=hyperparams.basis
    padding_1 = [hyperparams.padding_1,
    hyperparams.padding_2,
    hyperparams.padding_3,]
    padding_pool_1 = [hyperparams.padding_pool_1,
    hyperparams.padding_pool_2,
    hyperparams.padding_pool_3,]
    
    
    for i in range(3):
        k=conv_block_out(n=n,
                        kernel_size=kernel_size_1[i],
                        stride=stride_1[i],
                        dilation=dilation_1[i],
                        padding=padding_1[i],
                        )
        j=conv_block_out(n=k,
                        kernel_size=kernel_size_pool_1[i],
                        stride=stride_pool_1[i],
                        dilation=dilation_pool_1[i],
                        padding=padding_pool_1[i],
                        )
        n=j

    return n   






class TSCNN(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(TSCNN, self).__init__()
        n_conv_out =hyperparams.n_conv_out
        self.basis = hyperparams.basis
        granulation = hyperparams.granulation
        n_conv_in = hyperparams.n_conv_in
        n_conv_in2 = hyperparams.n_conv_in2
        n_conv_in3 = hyperparams.n_conv_in3
        n_Flat_out = hyperparams.n_Flat_out
        stride_1 = hyperparams.stride_1
        stride_2 = hyperparams.stride_2
        stride_3 = hyperparams.stride_3
        stride_pool_1 = hyperparams.stride_pool_1
        stride_pool_2 = hyperparams.stride_pool_2
        stride_pool_3 = hyperparams.stride_pool_3
        kernel_size_1 = hyperparams.kernel_size_1
        kernel_size_2 = hyperparams.kernel_size_2
        kernel_size_3 = hyperparams.kernel_size_3
        kernel_size_pool_1 = hyperparams.kernel_size_pool_1
        kernel_size_pool_2 = hyperparams.kernel_size_pool_2
        kernel_size_pool_3 = hyperparams.kernel_size_pool_3
        dilation_1 = hyperparams.dilation_1
        dilation_2 = hyperparams.dilation_2
        dilation_3 = hyperparams.dilation_3
        dilation_pool_1 = hyperparams.dilation_pool_1
        dilation_pool_2 = hyperparams.dilation_pool_2
        dilation_pool_3 = hyperparams.dilation_pool_3
        # basis=hyperparams.basis
        negative_slope=hyperparams.negative_slope
        padding_1 = hyperparams.padding_1
        padding_2 = hyperparams.padding_2
        padding_3 = hyperparams.padding_3
        padding_pool_1 = hyperparams.padding_pool_1
        padding_pool_2 = hyperparams.padding_pool_2
        padding_pool_3 = hyperparams.padding_pool_3
        negative_slope = hyperparams.negative_slope
        # Reste du code pour l'initialisation de la classe model
        self.granulation=granulation
        self.n_channel=hyperparams.n_channel
        self.convlayer1=nn.Sequential(
            nn.Conv1d(self.n_channel*len(hyperparams.derivative),n_conv_in,kernel_size=kernel_size_1,stride=stride_1,padding=padding_1,dilation=dilation_1),
            nn.BatchNorm1d(n_conv_in),
            nn.LeakyReLU(negative_slope),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=kernel_size_pool_1,stride=stride_pool_1,padding=padding_pool_1,dilation=dilation_pool_1),
            nn.BatchNorm1d(n_conv_in),
            nn.LeakyReLU(negative_slope),
        )
        
        self.convlayer2=nn.Sequential(
            nn.Conv1d(n_conv_in,n_conv_in2,kernel_size=kernel_size_2,stride=stride_2,padding=padding_2,dilation=dilation_2),
            nn.BatchNorm1d(n_conv_in2),
            nn.LeakyReLU(negative_slope),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=kernel_size_pool_2,stride=stride_pool_2,padding=padding_pool_2,dilation=dilation_pool_2),
            nn.BatchNorm1d(n_conv_in2),
            nn.LeakyReLU(negative_slope),
        )
        
        self.convlayer3=nn.Sequential(

            nn.Conv1d(n_conv_in2,n_conv_in3,kernel_size=kernel_size_3,stride=stride_3,padding=padding_3,dilation=dilation_3),
            nn.BatchNorm1d(n_conv_in3),
            nn.LeakyReLU(negative_slope),
            hyperparams.activation,
            nn.MaxPool1d(kernel_size=kernel_size_pool_3,stride=stride_pool_3,padding=padding_pool_3,dilation=dilation_pool_3),
            nn.BatchNorm1d(n_conv_in3),
            nn.LeakyReLU(negative_slope),
        )

        self.fc_block=nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_conv_out*n_conv_in3,n_Flat_out),
            nn.BatchNorm1d(n_Flat_out),
            nn.LeakyReLU(negative_slope),
            hyperparams.activation,
            nn.Linear(n_Flat_out,output_size),
            
        )
        self.n_conv_out=n_conv_out
        self.Smoothing_type=hyperparams.Smoothing_type
        self.hyperparameters=hyperparams

        self.smoother=hyperparams.Smoothing_method
    
    def Granulator(self,x):
        x_grid=from_torch_to_Datagrid(x=x)
        if "inter" not in self.Smoothing_type:     
            Recons_train=torch.zeros([x_grid.shape[0],self.n_channel*len(self.hyperparameters.derivative),self.granulation]).float().cuda()
            i=0
            for channel in range(self.n_channel):
                for deriv in self.hyperparameters.derivative:
                    eval_points=linspace(1,x_grid.grid_points[0].shape[0],self.granulation)
                    coefs=torch.tensor(x_grid.to_basis(basis=self.basis).coefficients).float().cuda()
                    basis_eval=self.basis(eval_points=eval_points,derivative=deriv)
                    basis_fc = torch.from_numpy(basis_eval).float().cuda()
                    coefs_torch=torch.tensor(coefs).float().cuda()        
                    Recons_train[:,i,:]=torch.matmul(coefs_torch,basis_fc[:,:,channel])
                    i+=1    
        
        else:
            x.interpolation=self.smoother.smoothing()
            eval_points=linspace(1,x.grid_points[0].shape[0],self.granulation)
            Recons_train=x.interpolation._evaluate(fdata=x,eval_points=eval_points)[:,:,0]
            Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],self.n_channel,Recons_train.shape[1])
        
        return Recons_train.float().cuda()



    def forward(self,x):
        Granulated_x_train=self.Granulator(x)
        tresh_out=torch.relu(Granulated_x_train)
        Conv_out=self.convlayer1(Granulated_x_train)
        Conv_out2=self.convlayer2(Conv_out)
        Conv_out3=self.convlayer3(Conv_out2)
        Lin_out=self.fc_block(Conv_out3)
        return Lin_out.float().unsqueeze_(2).unsqueeze_(3)

def from_torch_to_Datagrid(x):
    if isinstance(x,torch.Tensor):
        x_grid=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:])
    elif isinstance(x,skfda.representation.grid.FDataGrid):
        x_grid=x
    else:
        raise ValueError("the NN argument must be either torch.tensor or skfda.representation.grid.FDataGrid")
    
    return x_grid
    



class FCNN(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(FCNN, self).__init__()
        n_conv_out = hyperparams.n_conv_out
        n_conv_in = hyperparams.n_conv_in
        n_conv_in2 = hyperparams.n_conv_in2
        n_conv_in3 = hyperparams.n_conv_in3
        n_Flat_out = hyperparams.n_Flat_out
        stride_1 = hyperparams.stride_1
        stride_2 = hyperparams.stride_2
        stride_3 = hyperparams.stride_3
        stride_pool_1 = hyperparams.stride_pool_1
        stride_pool_2 = hyperparams.stride_pool_2
        stride_pool_3 = hyperparams.stride_pool_3
        kernel_size_1 = hyperparams.kernel_size_1
        kernel_size_2 = hyperparams.kernel_size_2
        kernel_size_3 = hyperparams.kernel_size_3
        kernel_size_pool_1 = hyperparams.kernel_size_pool_1
        kernel_size_pool_2 = hyperparams.kernel_size_pool_2
        kernel_size_pool_3 = hyperparams.kernel_size_pool_3
        dilation_1 = hyperparams.dilation_1
        dilation_2 = hyperparams.dilation_2
        dilation_3 = hyperparams.dilation_3
        dilation_pool_1 = hyperparams.dilation_pool_1
        dilation_pool_2 = hyperparams.dilation_pool_2
        dilation_pool_3 = hyperparams.dilation_pool_3
        negative_slope=hyperparams.negative_slope
        padding_1 = hyperparams.padding_1
        padding_2 = hyperparams.padding_2
        padding_3 = hyperparams.padding_3
        padding_pool_1 = hyperparams.padding_pool_1
        padding_pool_2 = hyperparams.padding_pool_2
        padding_pool_3 = hyperparams.padding_pool_3
        negative_slope=hyperparams.negative_slope
        # Reste du code pour l'initialisation de la classe model

        self.Relu=nn.ReLU()

        self.convlayer1=nn.Sequential(
            nn.Conv1d(hyperparams.n_channel,n_conv_in,kernel_size=kernel_size_1,stride=stride_1,padding=padding_1,dilation=dilation_1),
            nn.BatchNorm1d(n_conv_in),
            nn.LeakyReLU(negative_slope),
            
            nn.MaxPool1d(kernel_size=kernel_size_pool_1,stride=stride_pool_1,padding=padding_pool_1,dilation=dilation_pool_1),
            nn.BatchNorm1d(n_conv_in),
            nn.LeakyReLU(negative_slope),
        )
        
        self.convlayer2=nn.Sequential(
            nn.Conv1d(n_conv_in,n_conv_in2,kernel_size=kernel_size_2,stride=stride_2,padding=padding_2,dilation=dilation_2),
            nn.BatchNorm1d(n_conv_in2),
            nn.LeakyReLU(negative_slope),
            
            nn.MaxPool1d(kernel_size=kernel_size_pool_2,stride=stride_pool_2,padding=padding_pool_2,dilation=dilation_pool_2),
            nn.BatchNorm1d(n_conv_in2),
            nn.LeakyReLU(negative_slope),
        )
        
        self.convlayer3=nn.Sequential(

            nn.Conv1d(n_conv_in2,n_conv_in3,kernel_size=kernel_size_3,stride=stride_3,padding=padding_3,dilation=dilation_3),
            nn.BatchNorm1d(n_conv_in3),
            nn.LeakyReLU(negative_slope),
            
            nn.MaxPool1d(kernel_size=kernel_size_pool_3,stride=stride_pool_3,padding=padding_pool_3,dilation=dilation_pool_3),
            nn.BatchNorm1d(n_conv_in3),
            nn.LeakyReLU(negative_slope),
        )

        self.fc_block=nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_conv_out*n_conv_in3,n_Flat_out),
            nn.BatchNorm1d(n_Flat_out),
            nn.LeakyReLU(negative_slope),
            
            nn.Linear(n_Flat_out,output_size),
            
        )
    
        
    def forward(self,x):
       
        Conv_out=self.convlayer1(x)
        Conv_out2=self.convlayer2(Conv_out)
        Conv_out3=self.convlayer3(Conv_out2)
        Lin_out=self.fc_block(Conv_out3)
        return Lin_out.float().unsqueeze_(2).unsqueeze_(3)






class MLP(nn.Module):
    def __init__(self,hyperparams,input_size,output_size):
        super(MLP,self).__init__()
        self.input_size=input_size
        self.fc_block=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,hyperparams.n_conv_in),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in,hyperparams.n_conv_in2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in3,output_size),
        )

    def forward(self,x):
        if isinstance(x,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x.data_matrix).float().cuda()
            Lin_out=self.fc_block(data_matrix)
        elif isinstance(x,torch.Tensor):
            Lin_out=self.fc_block(x)
        else:
            raise ValueError("if isinstance(x,skfda.representation.grid.FDataGrid):")
        return Lin_out.float().unsqueeze(2).unsqueeze(3)

        

class Project_classifier(nn.Module):
    def __init__(self,hyperparams,output_size):
        super(Project_classifier,self).__init__()
        self.basis=hyperparams.Smoothing_method.smoothing()
        self.fc_block=nn.Sequential(
            nn.Linear(self.basis.n_basis,hyperparams.n_conv_in),
            nn.BatchNorm1d(hyperparams.n_conv_in),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in,hyperparams.n_conv_in2),
            nn.BatchNorm1d(hyperparams.n_conv_in2),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
            nn.BatchNorm1d(hyperparams.n_conv_in3),
            nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
            hyperparams.activation,
            nn.Linear(hyperparams.n_conv_in3,output_size),
        )

        
    def Project(self, x, basis_fc):
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        # w = x.size(1)-1
        # W = torch.tensor([1/(2*w)]+[1/w]*(w-1)+[1/(2*w)])
        
        f = torch.matmul(torch.tensor(x.data_matrix[:,:,0]).float().cuda(), torch.t(basis_fc))
        return f
    def forward(self,x):
        if isinstance(x,skfda.representation.grid.FDataGrid):
            basis_fc=self.basis(x.grid_points[0],derivative=0)[:,:,0]
            basis_fc=torch.tensor(basis_fc).float().cuda()
            proj_out=self.Project(x,basis_fc=basis_fc)
            lin_out=self.fc_block(proj_out)
        if isinstance(x,torch.Tensor):
            grid=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:])
            basis_fc=self.basis(grid.grid_points[0],derivative=0)[:,:,0]
            basis_fc=torch.tensor(basis_fc).float().cuda()
            proj_out=self.Project(grid,basis_fc=basis_fc)
            lin_out=self.fc_block(proj_out)
        return lin_out.float().unsqueeze(2).unsqueeze(3)
    
class MLP(nn.Module):
            def __init__(self,hyperparams,input_size,output_size):
                super(MLP,self).__init__()
                self.input_size=input_size
                self.fc_block=nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size,hyperparams.n_conv_in),
                    nn.BatchNorm1d(hyperparams.n_conv_in),
                    nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
                    hyperparams.activation,
                    nn.Linear(hyperparams.n_conv_in,hyperparams.n_conv_in2),
                    nn.BatchNorm1d(hyperparams.n_conv_in2),
                    nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
                    hyperparams.activation,
                    nn.Linear(hyperparams.n_conv_in2,hyperparams.n_conv_in3),
                    nn.BatchNorm1d(hyperparams.n_conv_in3),
                    nn.LeakyReLU(negative_slope=hyperparams.negative_slope),
                    hyperparams.activation,
                    nn.Linear(hyperparams.n_conv_in3,output_size),
                )

            def forward(self,x):
                if isinstance(x,skfda.representation.grid.FDataGrid):
                    data_matrix=torch.tensor(x.data_matrix).float().cuda()
                    Lin_out=self.fc_block(data_matrix)
                elif isinstance(x,torch.Tensor):
                    Lin_out=self.fc_block(x)
                else:
                    raise ValueError("NN input must be fdatagrid or torch tensor")
                return Lin_out.float().unsqueeze(2).unsqueeze(3)
          





def Compile_class(model_class="TSCNN",hyperparams=HyperParameters(),output_size=1,x_train=torch.zeros(6,6,7)):
    basis = hyperparams.basis
    granulation = hyperparams.granulation
    n_conv_in = hyperparams.n_conv_in
    n_conv_in2 = hyperparams.n_conv_in2
    n_conv_in3 = hyperparams.n_conv_in3
    n_Flat_out = hyperparams.n_Flat_out
    stride_1 = hyperparams.stride_1
    stride_2 = hyperparams.stride_2
    stride_3 = hyperparams.stride_3
    stride_pool_1 = hyperparams.stride_pool_1
    stride_pool_2 = hyperparams.stride_pool_2
    stride_pool_3 = hyperparams.stride_pool_3
    kernel_size_1 = hyperparams.kernel_size_1
    kernel_size_2 = hyperparams.kernel_size_2
    kernel_size_3 = hyperparams.kernel_size_3
    kernel_size_pool_1 = hyperparams.kernel_size_pool_1
    kernel_size_pool_2 = hyperparams.kernel_size_pool_2
    kernel_size_pool_3 = hyperparams.kernel_size_pool_3
    dilation_1 = hyperparams.dilation_1
    dilation_2 = hyperparams.dilation_2
    dilation_3 = hyperparams.dilation_3
    dilation_pool_1 = hyperparams.dilation_pool_1
    dilation_pool_2 = hyperparams.dilation_pool_2
    dilation_pool_3 = hyperparams.dilation_pool_3
    padding_1 = hyperparams.padding_1
    padding_2 = hyperparams.padding_2
    padding_3 = hyperparams.padding_3
    padding_pool_1 = hyperparams.padding_pool_1
    padding_pool_2 = hyperparams.padding_pool_2
    padding_pool_3 = hyperparams.padding_pool_3
    negative_slope=hyperparams.negative_slope
    
    
    if ("Conv" in model_class) or("Smooth" in model_class) or ("TSCNN" in model_class):
        X=x_train
        if isinstance(X,skfda.representation.grid.FDataGrid):
                if X.data_matrix.shape[2]!=1:
                    hyperparams.n_channel=X.data_matrix.shape[2]
                    grid_T=X.data_matrix.shape[1]
                    hyperparams.basis=B(knots=linspace(1,grid_T,6),order=4)
                    basis_list=[]
                    for channel in range(hyperparams.n_channel):
                        basis_channel=B(knots=linspace(1,grid_T,6),order=4)
                        basis_list.append(basis_channel)
                    hyperparams.basis=MultiBasis(basis_list=basis_list)
        elif isinstance(X,torch.Tensor):
            grid_T=X.shape[2]
            hyperparams.basis=B(knots=linspace(1,grid_T,6),order=4)
        hyperparams.n_conv_out=conv_total_out(hyperparams=hyperparams)
        module=TSCNN(hyperparams=hyperparams,output_size=output_size)

    elif "mlp" in model_class:
        input_size=x_train.shape[2]
        if isinstance(x_train,skfda.representation.grid.FDataGrid):
            data_matrix=torch.tensor(x_train.data_matrix).float().cuda()
            input_size=data_matrix.shape[1]
        elif isinstance(x_train,torch.Tensor):
            input_size=x_train.shape[2]
            
                
        module=MLP(hyperparams,input_size=input_size,output_size=output_size)

    elif ("Proj" in model_class) or ("proj" in model_class):
        module=Project_classifier(hyperparams=hyperparams,output_size=output_size)

    else:
        raise ValueError("model_class must be either Project,mlp or Conv")
    
    return module.cuda().apply(weights_init_normal)


def Compile_train(module, hyperparams,X,Y):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,shuffle=True)    
    opt=hyperparams.opt
    lr=hyperparams.lr
    loss=hyperparams.loss
    batch_size=hyperparams.batch_size
    betas = [0.5, 0.999]
    if opt == "Adam":
        optimizer = optim.Adam(module.parameters(), lr=lr, betas=betas)
    else:
        optimizer = optim.SGD(module.parameters(), lr=lr)
    def train(n_epochs, module, optimizer, loss, batch_size):
        training_acc=torch.tensor([0])
        testing_acc=torch.tensor([0])
        for epoch in (range(n_epochs)):
            train_loss = torch.tensor(0).cuda().long()
            
            # Mélanger les données d'entraînement
            indices = list(range(len(x_train)))
            random.shuffle(indices)
            
            batch_index = 0  # Indice de batch
            
            for i in range(int(len(x_train) / batch_size)):
                # Obtenir les indices des données mélangées
                batch_indices = indices[batch_index:batch_index+batch_size]
                functions_train = x_train[batch_indices]
                labels_train = y_train[batch_indices]
                optimizer.zero_grad()
                output = module(functions_train)
                loss_value = loss(input=output, target=labels_train)
                loss_value.backward()
                optimizer.step()
                train_loss += loss_value.long()
                batch_index += batch_size  # Passer au prochain batch
            
            
            accuracy_training=((torch.sum(torch.argmax(module(x_train),dim=1)==y_train)/x_train.shape[0])*100).cpu().unsqueeze(0)
            accuracy=((torch.sum(torch.argmax(module(x_test),dim=1)==y_test)/x_test.shape[0])*100).cpu().unsqueeze(0)
            testing_acc=torch.cat([testing_acc,accuracy],dim=0)
            training_acc=torch.cat([training_acc,accuracy_training],dim=0)
        
        return training_acc,testing_acc
    
    return lambda n_epochs: train(n_epochs, module, optimizer, loss, batch_size)




def Hyperparameter_Test(x,y,supra_epochs=50,alpha=0.95,model_class="smooth",hyperparameters=HyperParameters(),output_size=1):
    monte_carlo_test_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    monte_carlo_train_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    
    for epoch in (range(supra_epochs)):
        
        from scipy.stats import norm

        chiffre = alpha
        quartile = norm.ppf((1 + chiffre) / 2)

        
        ##Compilation de la classe 

        Model=Compile_class(hyperparams=hyperparameters,output_size=output_size,model_class=model_class,x_train=x)
        train_fn = Compile_train(module=Model, hyperparams=hyperparameters,X=x,Y=y)

        
        monte_carlo_train,monte_carlo_test=train_fn(n_epochs=hyperparameters.n_epochs)
        monte_carlo_test_acc=torch.cat([monte_carlo_test_acc,monte_carlo_test.unsqueeze(1)],dim=1)
        monte_carlo_train_acc=torch.cat([monte_carlo_train_acc,monte_carlo_train.unsqueeze(1)],dim=1)

        gc.collect()
        torch.cuda.empty_cache()

                

    mean_acc_train=torch.mean(monte_carlo_train_acc[1:,1:],dim=1).float()
    var_acc_train=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    
    
    mean_acc_test=torch.mean(monte_carlo_test_acc[1:,1:],dim=1).float()
    var_acc_test=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    IC_acc_test=torch.cat([(mean_acc_test-quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1),(mean_acc_test+quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1)],dim=1)
    IC_acc_train=torch.cat([(mean_acc_train-quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1),(mean_acc_train+quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1)],dim=1)

    return monte_carlo_test_acc[1:,1:],monte_carlo_train_acc[1:,1:],mean_acc_train,var_acc_train,IC_acc_train, mean_acc_test,var_acc_test,IC_acc_test


class model_printer():
    def __init__(self,name="model_name",color="model_color"):
        super(model_printer,self)
        self.name=name
        self.color=color


def Compare_epochs(params=None,
            datasets=None,
            models=None,
            supra_epochs=50,
            alpha=0.95,
            colors=None):
    colors=["darkred","darkblue"]
    label=[" sans dérivées"," avec dérivées"]
    # fig, axes = plt.subplots(int(floor(sqrt(len(datasets)))),int(floor(sqrt(len(datasets)))), figsize=(16, 16))
    num_datasets = len(datasets)
    num_plots_per_row = int(sqrt(num_datasets))
    num_plots_per_col = (num_datasets + num_plots_per_row - 1) // num_plots_per_row

    # Création de la figure et des axes
    fig, axes = plt.subplots(num_plots_per_col, num_plots_per_row, figsize=(12, 12))
    # fig, axes = plt.subplots(((len(datasets))), figsize=(16, 16))
    string1='Précision de '

    for i,dataset in enumerate(datasets):    
    # Boucle pour créer chaque subplot
        X,Y=dataset['X'],dataset['Y']
        window_left = i // num_plots_per_row
        window_right = i % num_plots_per_row
        ax = axes[window_left, window_right]
                
        for j,hyperparams in enumerate(params):
            hyperparams.batch_size=len(X)//3
            
            for k,model in enumerate(models):
                monte_carlo_test_acc=torch.zeros(hyperparams.n_epochs,supra_epochs,len(datasets),len(params),len(models))
                monte_carlo_train_acc=torch.zeros(hyperparams.n_epochs,supra_epochs,len(datasets),len(params),len(models))
                mean_acc_train=torch.zeros(hyperparams.n_epochs,len(datasets),len(params),len(models))
                mean_acc_test=torch.zeros(hyperparams.n_epochs,len(datasets),len(params),len(models))
                var_acc_train=torch.zeros(hyperparams.n_epochs,len(datasets),len(params),len(models))
                var_acc_test=torch.zeros(hyperparams.n_epochs,len(datasets),len(params),len(models))
                IC_acc_train=torch.zeros(hyperparams.n_epochs,2,len(datasets),len(params),len(models))
                IC_acc_test=torch.zeros(hyperparams.n_epochs,2,len(datasets),len(params),len(models))
                
                
                monte_carlo_test_acc[:,:,i,j,k],monte_carlo_train_acc[:,:,i,j,k],mean_acc_train[:,i,j,k],var_acc_train[:,i,j,k],IC_acc_train[:,:,i,j,k],mean_acc_test[:,i,j,k],var_acc_test[:,i,j,k],IC_acc_test[:,:,i,j,k]=Hyperparameter_Test(model_class=model,

                    hyperparameters=hyperparams,
                    supra_epochs=supra_epochs,
                    output_size=len(torch.unique(Y)),
                    x=X,
                    y=Y,
                    alpha=alpha,
                    
                    )
                
                
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+' '+"hyperparamètre"+str(j+1),color=colors[j])
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color=colors[j])
                ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+label[j],color=colors[j])
                ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color=colors[j])
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test[:,:,i,j,k],linestyle="dashed",color="")
                # ax.plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test[:,i,j,k], label=string1+model+' '+str(j+1)+"ème hyperparamètre")
                ax.set_title(dataset['dataset_name'])
                ax.set_xlabel("epochs")
                ax.set_ylabel("Pourcentage bien classés dans l'ensemble de test")
                ax.legend(loc="lower right")
            




        

    plt.tight_layout()  # Ajuster automatiquement les espacements entre les subplots
    
    plt.show()

    return fig,monte_carlo_test_acc,monte_carlo_train_acc,mean_acc_train,mean_acc_test,var_acc_train,var_acc_test,IC_acc_train,IC_acc_test
    # Premier dataset 

            
#     hyperparams.granulation=100*X2.shape[2]
#     hyperparams.Smoothing_method=Smoothing_method(knots=linspace(1,X2.shape[2],6),order=4)
    
#     monte_carlo_test_acc2,monte_carlo_train_acc2,mean_acc_train2,var_acc_train2,IC_acc_train2,mean_acc_test2,var_acc_test2,IC_acc_test2=Hyperparameter_Test(
#         hyperparameters=hyperparams,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y2)),
#         x=X2,
#         y=Y2,
#         alpha=alpha,
        
#         )
    
#     hyperparams.granulation=100*X3.shape[2]
#     hyperparams.Smoothing_method=Smoothing_method(knots=linspace(1,X3.shape[2],6),order=4)

#     monte_carlo_test_acc3,monte_carlo_train_acc3,mean_acc_train3,var_acc_train3,IC_acc_train3,mean_acc_test3,var_acc_test3,IC_acc_test3=Hyperparameter_Test(
#         hyperparameters=hyperparams,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y3)),
#         x=X3,
#         y=Y3,
#         alpha=alpha,
        
#         )
#     hyperparams.granulation=100*X4.shape[2]
#     hyperparams.Smoothing_method=Smoothing_method(knots=linspace(1,X4.shape[2],6),order=4)
    
    
#     monte_carlo_test_acc4,monte_carlo_train_acc4,mean_acc_train4,var_acc_train4,IC_acc_train4,mean_acc_test4,var_acc_test4,IC_acc_test4=Hyperparameter_Test(
#         hyperparameters=hyperparams_mlp,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y4)),
#         x=X4,
#         y=Y4,
#         alpha=alpha,
        
#         )
    

#     monte_carlo_test_acc1_mlp,monte_carlo_train_acc1_mlp,mean_acc_train1_mlp,var_acc_train1_mlp,IC_acc_train1_mlp,mean_acc_test1_mlp,var_acc_test1_mlp,IC_acc_test1_mlp=Hyperparameter_Test(model_class="mlp",
#         hyperparameters=hyperparams_mlp,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y1)),
#         x=X1,
#         y=Y1,
#         alpha=alpha,
        
#         )
    
#     monte_carlo_test_acc2_mlp,monte_carlo_train_acc2_mlp,mean_acc_train2_mlp,var_acc_train2_mlp,IC_acc_train2_mlp,mean_acc_test2_mlp,var_acc_test2_mlp,IC_acc_test2_mlp=Hyperparameter_Test(model_class="mlp",
#         hyperparameters=hyperparams_mlp,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y2)),
#         x=X2,
#         y=Y2,
#         alpha=alpha,
        
#         )
    
#     monte_carlo_test_acc3_mlp,monte_carlo_train_acc3_mlp,mean_acc_train3_mlp,var_acc_train3_mlp,IC_acc_train3_mlp,mean_acc_test3_mlp,var_acc_test3_mlp,IC_acc_test3_mlp=Hyperparameter_Test(model_class="mlp",
#         hyperparameters=hyperparams_mlp,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y3)),
#         x=X3,
#         y=Y3,
#         alpha=alpha,
        
#         )
#     monte_carlo_test_acc4_mlp,monte_carlo_train_acc4_mlp,mean_acc_train4_mlp,var_acc_train4_mlp,IC_acc_train4_mlp,mean_acc_test4_mlp,var_acc_test4_mlp,IC_acc_test4_mlp=Hyperparameter_Test(model_class="mlp",
#         hyperparameters=hyperparams_mlp,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y4)),
#         x=X4,
#         y=Y4,
#         alpha=alpha,
        
#         )
    
#     monte_carlo_test_acc1_Proj,monte_carlo_train_acc1_Proj,mean_acc_train1_Proj,var_acc_train1_Proj,IC_acc_train1_Proj,mean_acc_test1_Proj,var_acc_test1_Proj,IC_acc_test1_Proj=Hyperparameter_Test(model_class="Project",
#         hyperparameters=hyperparams_proj,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y1)),
#         x=X1,
#         y=Y1,
#         alpha=alpha,
        
#         )
    
#     monte_carlo_test_acc2_Proj,monte_carlo_train_acc2_Proj,mean_acc_train2_Proj,var_acc_train2_Proj,IC_acc_train2_Proj,mean_acc_test2_Proj,var_acc_test2_Proj,IC_acc_test2_Proj=Hyperparameter_Test(model_class="Project",
#         hyperparameters=hyperparams_proj,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y2)),
#         x=X2,
#         y=Y2,
#         alpha=alpha,
        
#         )
    
#     monte_carlo_test_acc3_Proj,monte_carlo_train_acc3_Proj,mean_acc_train3_Proj,var_acc_train3_Proj,IC_acc_train3_Proj,mean_acc_test3_Proj,var_acc_test3_Proj,IC_acc_test3_Proj=Hyperparameter_Test(model_class="Project",
#         hyperparameters=hyperparams_proj,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y3)),
#         x=X3,
#         y=Y3,
#         alpha=alpha,
#         )    
#     monte_carlo_test_acc4_Proj,monte_carlo_train_acc4_Proj,mean_acc_train4_Proj,var_acc_train4_Proj,IC_acc_train4_Proj,mean_acc_test4_Proj,var_acc_test4_Proj,IC_acc_test4_Proj=Hyperparameter_Test(model_class="Project",
#         hyperparameters=hyperparams_proj,
#         supra_epochs=supra_epochs,
#         output_size=len(torch.unique(Y4)),
#         x=X4,
#         y=Y4,
#         alpha=alpha,
#         )    
    
#     fig, axes = plt.subplots(2,2, figsize=(16, 16))
#     # Premier dataset 
#     axes[0,0].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test1, color='darkred', label='Précision de Convolution ')
#     axes[0,0].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test1, 'darkred', linestyle='dashed',)
#     axes[0,0].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test1_mlp, color='darkgreen', label='Précision de MLP ')
#     axes[0,0].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test1_mlp, 'darkgreen', linestyle='dashed',)
#     axes[0,0].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test1_Proj, 'darkblue', linestyle='dashed',)
#     axes[0,0].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test1_Proj, color='darkblue', label='Précision de project ')
#     axes[0,0].set_title("Phoneme")
#     axes[0,0].set_xlabel("epochs")
#     axes[0,0].set_ylabel("Pourcentage bien classés dans l'ensemble de test)")
#     axes[0,0].legend(loc="lower right")
#     # Second dataset
#     axes[0,1].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test2, color='darkred', label='Précision de Convolution ')
#     axes[0,1].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test2, 'darkred', linestyle='dashed',label="Intervalle de confiance ")
#     axes[0,1].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test2_mlp, color='darkgreen', label='Précision de MLP ')
#     axes[0,1].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test2_mlp, 'darkgreen', linestyle='dashed')
#     axes[0,1].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test2_Proj, color='darkblue', label='Précision de project ')
#     axes[0,1].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test2_Proj, 'darkblue', linestyle='dashed')
#     axes[0,1].set_title("El_nino")
#     axes[0,1].set_xlabel("epochs")
#     axes[0,1].set_ylabel("Pourcentage bien classés dans l'ensemble de test)")
#     axes[0,1].legend(loc="lower right")

#     axes[1,0].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test3, color='darkred', label='Précision de Convolution ')
#     axes[1,0].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test3_mlp, color='darkgreen', label='Précision de MLP')
#     axes[1,0].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test3, 'darkred', linestyle='dashed')
#     axes[1,0].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test3_mlp, 'darkgreen', linestyle='dashed')
#     axes[1,0].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test3_Proj, color='darkblue', label='Précision de project ')
#     axes[1,0].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test3_Proj, 'darkblue', linestyle='dashed')
#     axes[1,0].set_title("SOFA")
#     axes[1,0].set_xlabel("epochs")
#     axes[1,0].set_ylabel("Pourcentage bien classés dans l'ensemble de test)")
#     axes[1,0].legend(loc="lower right")
#     # Premier dataset 
#     axes[1,1].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test4, color='darkred', label='Précision de Convolution ')
#     axes[1,1].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test4, 'darkred', linestyle='dashed')
#     axes[1,1].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test4_mlp, color='darkgreen', label='Précision de MLP ')
#     axes[1,1].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test4_mlp, 'darkgreen', linestyle='dashed')
#     axes[1,1].plot(np.arange(hyperparams.n_epochs+1)[1:],IC_acc_test4_Proj, 'darkblue', linestyle='dashed')
#     axes[1,1].plot(np.arange(hyperparams.n_epochs+1)[1:],mean_acc_test4_Proj, color='darkblue', label='Précision de project ')
#     axes[1,1].set_title("Données simulées")
#     axes[1,1].set_xlabel("epochs")
#     axes[1,1].set_ylabel("Pourcentage bien classés dans l'ensemble de test)")
#     axes[1,1].legend(loc="lower right")

    
#     plt.tight_layout()  # Ajuster automatiquement les espacements entre les subplots
#     plt.show()

# # fig.savefig("C:/Users/Utilisateur/Pictures/Images et animation/TSCNNacc.png")

#     return fig,monte_carlo_test_acc1,monte_carlo_train_acc1,mean_acc_train1,var_acc_train1,IC_acc_train1,mean_acc_test1,var_acc_test1,IC_acc_test1,monte_carlo_test_acc2,monte_carlo_train_acc2,mean_acc_train2,var_acc_train2,IC_acc_train2,mean_acc_test2,var_acc_test2,IC_acc_test2,monte_carlo_test_acc3,monte_carlo_train_acc3,mean_acc_train3,var_acc_train3,IC_acc_train3,mean_acc_test3,var_acc_test3,IC_acc_test3,monte_carlo_test_acc1_Proj,monte_carlo_train_acc1_Proj,mean_acc_train1_Proj,var_acc_train1_Proj,IC_acc_train1_Proj,mean_acc_test1_Proj,var_acc_test1_Proj,IC_acc_test1_Proj,monte_carlo_test_acc2_Proj,monte_carlo_train_acc2_Proj,mean_acc_train2_Proj,var_acc_train2_Proj,IC_acc_train2_Proj,mean_acc_test2_Proj,var_acc_test2_Proj,IC_acc_test2_Proj,monte_carlo_test_acc3_Proj,monte_carlo_train_acc3_Proj,mean_acc_train3_Proj,var_acc_train3_Proj,IC_acc_train3_Proj,mean_acc_test3_Proj,var_acc_test3_Proj,IC_acc_test3_Proj




































def Hyperparameter_Test_Mse(hyperparameters,model_class,x,y,output_size=1,supra_epochs=50,alpha=1.96):
    for epochs in tqdm(range(supra_epochs)):
        
        from scipy.stats import norm

        chiffre = alpha
        quartile = norm.ppf((1 + chiffre) / 2)

        

        x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,shuffle=True)    
        ##Compilation de la classe 
        T=x.shape[2]

        Model=Compile_class(hyperparams=hyperparameters,output_size=output_size,model_class=model_class,x_train=x_train)
        
        Loss_min_monte_carlo=torch.tensor([0])
        mean_accuracy=torch.tensor([0])
        train_fn = Compile_train(module=Model, hyperparams=hyperparameters,x_train=x_train,y_train=y_train)


        for i in tqdm(range(hyperparameters.n_epochs)):
                train_fn(n_epochs=1)
                accuracy=nn.MSELoss()(Model(x_test),y_test)
                mean_accuracy=torch.cat([mean_accuracy,accuracy.cpu().unsqueeze(0)],dim=0)
                Loss_min=torch.min(mean_accuracy[1:].float())
                n_epochs_max_acc=torch.argmax(mean_accuracy)    
        
        Loss_min_monte_carlo=torch.cat([Loss_min,Loss_min_monte_carlo.unsqueeze(0)],dim=0)
        mean_mse=torch.mean(Loss_min_monte_carlo[1:].float())
        var_mse=torch.var(Loss_min_monte_carlo[1:].float())
        IC_mse=[mean_mse-quartile*sqrt(var_mse)/sqrt(supra_epochs),mean_mse+quartile*sqrt(var_mse)/sqrt(supra_epochs)]
# print("Loss moyenne =",((torch.mean(mean_accuracy[1:].float()))).detach().cpu().numpy())  
        # print("Loss min=",((torch.min(mean_accuracy[1:].float()))).detach().cpu().numpy())  
        # print("Variance des précisions =",((torch.var(mean_accuracy[1:].float()))).detach().cpu().numpy())  
        # # print(mean_accuracy.unsqueeze(1)[1:]) 
        
        
    gc.collect()
    torch.cuda.empty_cache()

    return Model, mean_mse,var_mse,IC_mse


def Hyper_parameter_GridSearch(hyperparams,parameter, grid,model_class,x,y,output_size):
    Total_accuracy_max = torch.tensor([0])
    Optimum_parameter = grid[0]

    
    
    # Obtenir l'attribut correspondant au paramètre spécifié
    attribute = getattr(hyperparams, parameter)
    
    for value in grid:
        # Modifier la valeur de l'attribut de la classe HyperParameters
        setattr(hyperparams, parameter, value)
        print(parameter,"=",value)
        # Utiliser l'instance de HyperParameters pour effectuer les tests
        Model, accuracy,n_best = Hyperparameter_Test(hyperparameters=hyperparams,model_class=model_class,x=x,y=y,output_size=output_size)
        Total_accuracy_max = torch.cat([Total_accuracy_max, torch.tensor([torch.max(accuracy)])])

    Optimum_parameter = grid[torch.argmax(Total_accuracy_max[1:])]
    return Model,Optimum_parameter, torch.max(Total_accuracy_max[1:]),n_best


def Hyperparameter_Search(hyperparams, grids, parameters,model_class,x,y,output_size):
    best_parameters = hyperparams
    best_accuracy = 0.0
    mean_acc_base=0.0
    var_acc=0.0
    
    # Boucle sur les paramètres
    for param in parameters:
        # Vérifier si le paramètre est dans la grille
        if param in grids:
            grid_values = grids[param]  # Récupérer les valeurs de la grille pour le paramètre donné

            # Boucle sur les valeurs de la grille pour le paramètre
            for value in grid_values:
                # Mettre à jour les hyperparamètres avec la valeur actuelle du paramètre
                setattr(best_parameters, param, value)

                # Appeler la fonction de Grid Search avec les paramètres spécifiés
                Model_opt,optimum_parameter, total_accuracy,n_best = Hyper_parameter_GridSearch(best_parameters,grid=grid_values,parameter=param,model_class=model_class,x=x,y=y,output_size=output_size)
                Mean_acc=((torch.mean(total_accuracy.float()))).detach().cpu().numpy()
                Max_acc=((torch.max(total_accuracy.float()))).detach().cpu().numpy()
                Var_acc=((torch.var(total_accuracy.float()))).detach().cpu().numpy()
            
                # Mettre à jour le meilleur résultat si nécessaire
                if total_accuracy >= best_accuracy:
                    if total_accuracy==best_accuracy:
                        if var_acc>Var_acc:
                            best_parameters.param = optimum_parameter
                            best_accuracy = total_accuracy
                    else:
                        best_parameters.param = optimum_parameter
                        best_accuracy = total_accuracy
                    
    return Model_opt,best_parameters, best_accuracy,n_best


def Hyperparameter_Test_Project(x,y,supra_epochs=50,alpha=0.95,model_class="smooth",hyperparameters=HyperParameters(),output_size=1):
    monte_carlo_test_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    monte_carlo_train_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    
    for epoch in tqdm(range(supra_epochs)):
        
        from scipy.stats import norm

        chiffre = alpha
        quartile = norm.ppf((1 + chiffre) / 2)

        
        ##Compilation de la classe 

        Model=Project_classifier(hyperparams=hyperparameters,output_size=output_size).cuda().apply(weights_init_normal)
        train_fn = Compile_train(module=Model, hyperparams=hyperparameters,X=x,Y=y)

        
        monte_carlo_train,monte_carlo_test=train_fn(n_epochs=hyperparameters.n_epochs)
        monte_carlo_test_acc=torch.cat([monte_carlo_test_acc,monte_carlo_test.unsqueeze(1)],dim=1)
        monte_carlo_train_acc=torch.cat([monte_carlo_train_acc,monte_carlo_train.unsqueeze(1)],dim=1)

        gc.collect()
        torch.cuda.empty_cache()

                

    mean_acc_train=torch.mean(monte_carlo_train_acc[1:,1:],dim=1).float()
    var_acc_train=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    
    
    mean_acc_test=torch.mean(monte_carlo_test_acc[1:,1:],dim=1).float()
    var_acc_test=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    IC_acc_test=torch.cat([(mean_acc_test-quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1),(mean_acc_test+quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1)],dim=1)
    IC_acc_train=torch.cat([(mean_acc_train-quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1),(mean_acc_train+quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1)],dim=1)

    return monte_carlo_test_acc[1:,1:],monte_carlo_train_acc[1:,1:],mean_acc_train,var_acc_train,IC_acc_train, mean_acc_test,var_acc_test,IC_acc_test

def Hyperparameter_Test_mlp(x,y,supra_epochs=50,alpha=0.95,model_class="smooth",hyperparameters=HyperParameters(),output_size=1):
    monte_carlo_test_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    monte_carlo_train_acc=torch.zeros(hyperparameters.n_epochs+1,1)
    
    for epoch in tqdm(range(supra_epochs)):
        
        from scipy.stats import norm

        chiffre = alpha
        quartile = norm.ppf((1 + chiffre) / 2)

        
        ##Compilation de la classe 
        input_size=x.shape[2]
        Model=MLP(hyperparams=hyperparameters,output_size=output_size,input_size=input_size).cuda().apply(weights_init_normal)

        train_fn = Compile_train(module=Model, hyperparams=hyperparameters,X=x,Y=y)

        
        monte_carlo_train,monte_carlo_test=train_fn(n_epochs=hyperparameters.n_epochs)
        monte_carlo_test_acc=torch.cat([monte_carlo_test_acc,monte_carlo_test.unsqueeze(1)],dim=1)
        monte_carlo_train_acc=torch.cat([monte_carlo_train_acc,monte_carlo_train.unsqueeze(1)],dim=1)

        gc.collect()
        torch.cuda.empty_cache()

                

    mean_acc_train=torch.mean(monte_carlo_train_acc[1:,1:],dim=1).float()
    var_acc_train=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    
    
    mean_acc_test=torch.mean(monte_carlo_test_acc[1:,1:],dim=1).float()
    var_acc_test=torch.var(monte_carlo_test_acc[1:,1:],dim=1).float()
    IC_acc_test=torch.cat([(mean_acc_test-quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1),(mean_acc_test+quartile*sqrt(var_acc_test/supra_epochs)).unsqueeze(1)],dim=1)
    IC_acc_train=torch.cat([(mean_acc_train-quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1),(mean_acc_train+quartile*sqrt(var_acc_train/supra_epochs)).unsqueeze(1)],dim=1)

    return monte_carlo_test_acc[1:,1:],monte_carlo_train_acc[1:,1:],mean_acc_train,var_acc_train,IC_acc_train, mean_acc_test,var_acc_test,IC_acc_test


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

class Smoothing_method:
    def __init__(self,knots_or_basis="knots",Mode="Smooth",basis_type="Bspline",interpolation_order=3,B_spline_order=4,
                 knots=np.linspace(1,12,6),period=2*pi,n_basis=13):
        self.Mode=Mode
        self.basis_type=basis_type
        self.interpolation_order=interpolation_order
        self.B_spline_order=B_spline_order
        self.knots_or_basis=knots_or_basis
        self.knots=knots
        self.period=period
        self.n_basis=n_basis
    def smoothing(self):
        if 'inter' in self.Mode:
            # interpolation=skfda.representation.interpolation.SplineInterpolation(interpolation_order=self.interpolation_order)             
            smooth_meth= skfda.representation.interpolation.SplineInterpolation(interpolation_order=self.interpolation_order)             
        else:
            if "knots" in self.knots_or_basis:  
                if ("spline" in self.basis_type) or ("Bsp" in self.basis_type) or ("spl" in self.basis_type):
                    smooth_meth= B(knots=self.knots,order=self.B_spline_order)
                if self.basis_type=="Fourier":
                    smooth_meth=skfda.representation.basis.FourierBasis(domain_range=[min(self.knots),max(self.knots)],period=self.period)
            if "basis" in self.knots_or_basis:
                if ("spline" in self.basis_type) or ("Bsp" in self.basis_type) or ("spbasis" in self.basis_type):
                    smooth_meth= B(n_basis=self.n_basis,order=self.B_spline_order)
                if ("fourier" in self.basis_type)or ("fourrier" in self.basis_type) or ("Fourier" in self.basis_type) or ("four" in self.basis_type):
                    smooth_meth=skfda.representation.basis.FourierBasis(n_basis=self.n_basis,period=self.period)
        return smooth_meth



class HyperParameters:
    def __init__(self,Smoothing_method=Smoothing_method(),batch_size=30, n_epochs=25, basis=B(knots=linspace(1, 12, 6), order=3), granulation=2000,
                 n_conv_in=32, n_conv_in2=512, n_conv_in3=256,n_conv_out=64, n_Flat_out=256,
                 stride_1=1, stride_2=1, stride_3=1,
                 stride_pool_1=2, stride_pool_2=2, stride_pool_3=1,
                 kernel_size_1=7, kernel_size_2=4, kernel_size_3=3,
                 kernel_size_pool_1=3, kernel_size_pool_2=3, kernel_size_pool_3=2,
                 dilation_1=1, dilation_2=1, dilation_3=1,
                 dilation_pool_1=1, dilation_pool_2=1, dilation_pool_3=1,
                 padding_1=2, padding_2=2, padding_3=2,
                 padding_pool_1=0, padding_pool_2=0, padding_pool_3=0,
                 opt="Adam", lr=0.00089, loss=nn.CrossEntropyLoss(),activation=nn.Identity(),negative_slope=0.17):

        self.basis=Smoothing_method.smoothing()
        self.Smoothing_type=Smoothing_method.Mode
        self.Smoothing_method=Smoothing_method
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.activation=activation
        self.n_conv_out=n_conv_out
        
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

class TSCNN(nn.Module):
    def __init__(self, hyperparams,output_size):
        super(TSCNN, self).__init__()
        n_conv_out =hyperparams.n_conv_out
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
        basis=hyperparams.basis
        negative_slope=hyperparams.negative_slope
        padding_1 = hyperparams.padding_1
        padding_2 = hyperparams.padding_2
        padding_3 = hyperparams.padding_3
        padding_pool_1 = hyperparams.padding_pool_1
        padding_pool_2 = hyperparams.padding_pool_2
        padding_pool_3 = hyperparams.padding_pool_3
        negative_slope = hyperparams.negative_slope
        # Reste du code pour l'initialisation de la classe model
        self.basis=basis
        self.granulation=granulation
        self.convlayer1=nn.Sequential(
            nn.Conv1d(1,n_conv_in,kernel_size=kernel_size_1,stride=stride_1,padding=padding_1,dilation=dilation_1),
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
        if "inter" not in self.Smoothing_type:     
            if isinstance(x,skfda.representation.grid.FDataGrid):
                eval_points=linspace(1,x.grid_points[0].shape[0],self.granulation)
                coefs=x.to_basis(basis=self.basis).coefficients
                basis_eval=self.basis.evaluate(eval_points=eval_points)[:, :, 0]
                basis_fc = torch.from_numpy(basis_eval).float().cuda()
                
            elif isinstance(x,torch.Tensor):
                eval_points=linspace(1,x.shape[2],self.granulation)
                basis_eval=self.basis.evaluate(eval_points=eval_points)[:, :, 0]
                basis_fc = torch.from_numpy(basis_eval).float().cuda()
                coefs=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:]).to_basis(basis=self.basis).coefficients

            else:
                raise ValueError("the NN argument must be either torch.tensor or skfda.representation.grid.FDataGrid")
        
                
            coefs_torch=torch.tensor(coefs).float().cuda()
            Recons_train=torch.matmul(coefs_torch,basis_fc)
            Recons_train=Recons_train.reshape(Recons_train.shape[0],1,Recons_train.shape[1])
        
        else:
            if isinstance(x,skfda.representation.grid.FDataGrid):
                x.interpolation=self.smoother.smoothing()
                eval_points=linspace(1,x.grid_points[0].shape[0],self.granulation)
                Recons_train=x.interpolation._evaluate(fdata=x,eval_points=eval_points)[:,:,0]
                Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],1,Recons_train.shape[1])
            if isinstance(x,torch.Tensor):
                grid=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:])
                eval_points=linspace(1,x.shape[2],self.granulation)
                grid.interpolation=self.smoother.smoothing()
                Recons_train=grid.interpolation._evaluate(fdata=grid,eval_points=eval_points)
                Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],1,Recons_train.shape[1])
                

        return Recons_train.float().cuda()


    def forward(self,x):
        Granulated_x_train=self.Granulator(x)
        tresh_out=torch.relu(Granulated_x_train)
        Conv_out=self.convlayer1(tresh_out)
        Conv_out2=self.convlayer2(Conv_out)
        Conv_out3=self.convlayer3(Conv_out2)
        Lin_out=self.fc_block(Conv_out3)
        return Lin_out.float().unsqueeze_(2).unsqueeze_(3)




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
            nn.Conv1d(1,n_conv_in,kernel_size=kernel_size_1,stride=stride_1,padding=padding_1,dilation=dilation_1),
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


def Compile_class(model_class,hyperparams,x_train,output_size):
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
    if model_class == "Conv1D":
        CNN = FCNN(hyperparams=hyperparams,output_size=output_size).cuda()
        Conv_out = CNN.convlayer1(x_train[:2,:,:])
        Conv_out2=CNN.convlayer2(Conv_out)
        Conv_out3=CNN.convlayer3(Conv_out2)
        n_conv_out1=Conv_out.shape[2]
        n_conv_out2=Conv_out2.shape[2]
        n_conv_out3=Conv_out3.shape[2]
        hyperparams.n_conv_out=n_conv_out3
        class Bsp_classifier(nn.Module): 
            def __init__(self, hyperparams):
                super(Bsp_classifier, self).__init__()
                
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

                self.convlayer1=nn.Sequential(
                    nn.Conv1d(1,n_conv_in,kernel_size=kernel_size_1,stride=stride_1,padding=padding_1,dilation=dilation_1),
                    nn.BatchNorm1d(n_conv_in),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                    nn.MaxPool1d(kernel_size=kernel_size_pool_1,stride=stride_pool_1,padding=padding_pool_1,dilation=dilation_pool_1),
                    nn.BatchNorm1d(n_conv_in),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                )
                
                self.convlayer2=nn.Sequential(
                    nn.Conv1d(n_conv_in,n_conv_in2,kernel_size=kernel_size_2,stride=stride_2,padding=padding_2,dilation=dilation_2),
                    nn.BatchNorm1d(n_conv_in2),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                    nn.MaxPool1d(kernel_size=kernel_size_pool_2,stride=stride_pool_2,padding=padding_pool_2,dilation=dilation_pool_2),
                    nn.BatchNorm1d(n_conv_in2),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
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
                    nn.Linear(n_conv_out3*n_conv_in3,n_Flat_out),
                    nn.BatchNorm1d(n_Flat_out),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                    nn.Linear(n_Flat_out,output_size),
                    
                )
                self.n_conv_out1=n_conv_out1
                self.n_conv_out2=n_conv_out2
                self.n_conv_out3=n_conv_out3
            def forward(self,x):
                Conv_out=self.convlayer1(x)
                Conv_out2=self.convlayer2(Conv_out)
                Conv_out3=self.convlayer3(Conv_out2)
                Lin_out=self.fc_block(Conv_out3)
                return Lin_out.float().unsqueeze_(2).unsqueeze_(3)
        module=Bsp_classifier(hyperparams=hyperparams) 

    else:
        CNN=TSCNN(hyperparams=hyperparams,output_size=output_size).cuda()
        Granul=CNN.Granulator(x_train)
        Conv_out=CNN.convlayer1(Granul)
        Conv_out2=CNN.convlayer2(Conv_out)
        Conv_out3=CNN.convlayer3(Conv_out2)
        n_conv_out1=Conv_out.shape[2]
        n_conv_out2=Conv_out2.shape[2]
        n_conv_out3=Conv_out3.shape[2]
        hyperparams.n_conv_out=n_conv_out3
        class Bsp_classifier(nn.Module): 
            def __init__(self, hyperparams):
                super(Bsp_classifier, self).__init__()
                basis = hyperparams.basis
                granulation = hyperparams.granulation
                n_conv_in = hyperparams.n_conv_in
                n_conv_in2 = hyperparams.n_conv_in2
                n_conv_in3 = hyperparams.n_conv_in3
                n_conv_out=hyperparams.n_conv_out
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

                self.granulation=granulation
                self.convlayer1=nn.Sequential(
                    nn.Conv1d(1,n_conv_in,kernel_size=kernel_size_1,stride=stride_1,padding=padding_1,dilation=dilation_1),
                    nn.BatchNorm1d(n_conv_in),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                    nn.MaxPool1d(kernel_size=kernel_size_pool_1,stride=stride_pool_1,padding=padding_pool_1,dilation=dilation_pool_1),
                    nn.BatchNorm1d(n_conv_in),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                )
                
                self.convlayer2=nn.Sequential(
                    nn.Conv1d(n_conv_in,n_conv_in2,kernel_size=kernel_size_2,stride=stride_2,padding=padding_2,dilation=dilation_2),
                    nn.BatchNorm1d(n_conv_in2),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                    nn.MaxPool1d(kernel_size=kernel_size_pool_2,stride=stride_pool_2,padding=padding_pool_2,dilation=dilation_pool_2),
                    nn.BatchNorm1d(n_conv_in2),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
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
                    nn.Linear(n_conv_out3*n_conv_in3,n_Flat_out),
                    nn.BatchNorm1d(n_Flat_out),
                    nn.LeakyReLU(negative_slope),
                    hyperparams.activation,
                    nn.Linear(n_Flat_out,output_size),
                    
                )
                self.basis=basis
                self.n_conv_out1=n_conv_out1
                self.n_conv_out2=n_conv_out2
                self.n_conv_out3=n_conv_out3
                    
                self.Smoothing_type=hyperparams.Smoothing_type
                self.hyperparameters=hyperparams
                self.smoother=hyperparams.Smoothing_method
            def Granulator(self,x):
                if "inter" not in self.Smoothing_type:     
                    if isinstance(x,skfda.representation.grid.FDataGrid):
                        eval_points=linspace(1,x.grid_points[0].shape[0],self.granulation)
                        coefs=x.to_basis(basis=self.basis).coefficients
                        basis_eval=self.basis.evaluate(eval_points=eval_points)[:, :, 0]
                        basis_fc = torch.from_numpy(basis_eval).float().cuda()
                        
                    if isinstance(x,torch.Tensor):
                        eval_points=linspace(1,x.shape[2],self.granulation)
                        basis_eval=self.basis.evaluate(eval_points=eval_points)[:, :, 0]
                        basis_fc = torch.from_numpy(basis_eval).float().cuda()
                        coefs=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:]).to_basis(basis=self.basis).coefficients

                    else:
                        raise ValueError("the NN argument must be either torch.tensor or skfda.representation.grid.FDataGrid")
                
                        
                    coefs_torch=torch.tensor(coefs).float().cuda()
                    Recons_train=torch.matmul(coefs_torch,basis_fc)
                    Recons_train=Recons_train.reshape(Recons_train.shape[0],1,Recons_train.shape[1])
                
                else:
                    if isinstance(x,skfda.representation.grid.FDataGrid):
                        x.interpolation=self.smoother.smoothing()
                        eval_points=linspace(1,x.grid_points[0].shape[0],self.granulation)
                        Recons_train=x.interpolation._evaluate(fdata=x,eval_points=eval_points)[:,:,0]
                        Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],1,Recons_train.shape[1])
                    if isinstance(x,torch.Tensor):
                        grid=fd(x[:,0,:].cpu(),grid_points=np.arange(x.shape[2]+1)[1:])
                        eval_points=linspace(1,x.shape[2],self.granulation)
                        grid.interpolation=self.smoother.smoothing()
                        Recons_train=grid.interpolation._evaluate(fdata=grid,eval_points=eval_points)
                        Recons_train=torch.tensor(Recons_train).reshape(Recons_train.shape[0],1,Recons_train.shape[1])
                        

                return Recons_train.float().cuda()
                    
            def forward(self,x):
                Granulated_x_train=self.Granulator(x)
                tresh_out=torch.relu(Granulated_x_train)
                Conv_out=self.convlayer1(Granulated_x_train)
                Conv_out2=self.convlayer2(Conv_out)
                Conv_out3=self.convlayer3(Conv_out2)
                Lin_out=self.fc_block(Conv_out3)
                return Lin_out.float().unsqueeze_(2).unsqueeze_(3)
            
        module=Bsp_classifier(hyperparams=hyperparams)
    
    return module.cuda().apply(weights_init_normal)


def Compile_train(module, hyperparams,x_train,y_train):
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
        for epoch in range(n_epochs):
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
                
            return train_loss, loss_value
    
    return lambda n_epochs: train(n_epochs, module, optimizer, loss, batch_size)


def Hyperparameter_Test(hyperparameters,model_class,x,y,output_size):

    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,shuffle=True)    
    ##Compilation de la classe 
    T=x.shape[2]

    Model=Compile_class(hyperparams=hyperparameters,output_size=output_size,model_class=model_class,x_train=x_train)
    
    
    mean_accuracy=torch.tensor([0])
    train_fn = Compile_train(module=Model, hyperparams=hyperparameters,x_train=x_train,y_train=y_train)


    for i in tqdm(range(hyperparameters.n_epochs)):
            train_fn(n_epochs=1)
            
            accuracy=((torch.sum(torch.argmax(Model(x_test),dim=1)==y_test)/x_test.shape[0])*100)
            mean_accuracy=torch.cat([mean_accuracy,accuracy.cpu().unsqueeze(0)],dim=0)
    print("Précision moyenne =",((torch.mean(mean_accuracy[1:].float()))).detach().cpu().numpy(),"%")  
    print("Précision max=",((torch.max(mean_accuracy[1:].float()))).detach().cpu().numpy(),"%")  
    print("Variance des précisions =",((torch.var(mean_accuracy[1:].float()))).detach().cpu().numpy())  
    # print(mean_accuracy.unsqueeze(1)[1:]) 
    n_epochs_max_acc=torch.argmax(mean_accuracy)    
    gc.collect()
    torch.cuda.empty_cache()
    return Model, mean_accuracy[1:],n_epochs_max_acc

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


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLPRegressor(nn.Module):
    def __init__(self,n_inputs,n_outputs,hidden):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(n_inputs,hidden)
        self.fc2 = nn.Linear(hidden,n_outputs)
        self.relu = torch.nn.ReLU() 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)
        return x
    
titer=0
lac=0
glc=0
Amon=0
GCPD=0
IVCC=0
VCD=0
viability=0 
#Vector of ground truths of titer 
Final_Titer=880 
time = np.array([np.arange(-3,15)])
#What size to set the hidden layer
hidden=torch.zeros(1,256)
#Our meeting said to initialize layers and units. But it is unclear for me which are the units. Is it the reactor ( 3 units) or the variables ( Lac. Glc..etc)
f=MLPRegressor(layers)
g=MLPRegressor(layers=2,n_inputs=0)
h=MLPRegressor(layers=2,n_inputs=0)
i=MLPRegressor(layers=2,n_inputs=0)
j=MLPRegressor(layers=2,n_inputs=0)
k=MLPRegressor(layers=2,n_inputs=0)
L=MLPRegressor(layers=2,n_inputs=0)
m=MLPRegressor(layers=2,n_inputs=0)
n=MLPRegressor(layers=2,n_inputs=0)


for t in time:   
    #This vector has all the cumulative Oxygen time profile (from -3 to 14 dpi) of a single batch
    cO2=[]  
    #This vector has all the cumulative Carbon Dioxide time profile (from -3 to 14 dpi) of a single batch
    cCO2=[]
    #This vector has all the pH (from -3 to 14 dpi) of a single batch
    pH=[]
    #This vector has all the base volume added (from -3 to 14 dpi) of a single batch
    Base=[]
    hidden=f(cO2,cCO2,pH,Base,titer,lac,glc,Amon,GCPD,IVCC,VCD,viability,hidden)
    titer=g(hidden)+titer
    lac=h(hidden)+lac
    glc=i(hidden)+glc
    Amon=j(hidden)+Amon
    GCPD=k(hidden)+GCPD
    IVCC=L(hidden)+IVCC
    VCD=m(hidden)+VCD
    viability=n(hidden)+viability

    error= Final_Titer-titer
    
    print(error)
    

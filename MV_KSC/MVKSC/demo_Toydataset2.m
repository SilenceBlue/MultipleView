%Demo studying the performance of the MVKSC method on a synthetic dataset

%Author: Lynn Houthuys

%Citation: 

%L. Houthuys, R. Langone, and J. A. K. Suykens, Multi-View Kernel Spectral
%Clustering, Internal Report 17-71, ESAT-SISTA, KU Leuven (Leuven, Belgium), 2017.

clear all
close all
addpath('MVKSCutils');

%% load data
load('synth3views_2clusters');

%% Settings - parameters obtained by tuning
sigma=2.2593;
gamma=[0.4338,3.0149,2.4768];
k=2;

%% Run algorithm
model = MVKSC(X,'RBF_kernel',sigma,k,gamma,'mean');

%% Evaluate performance
NMI_avg= getNMI(model.qtrain,truth);
ARI_avg = getARI(model.qtrain,truth);
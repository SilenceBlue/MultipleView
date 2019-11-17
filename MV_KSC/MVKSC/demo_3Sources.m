%Demo studying the performance of the MVKSC method on a real-world dataset

%Author: Lynn Houthuys

%Citation: 

%L. Houthuys, R. Langone, and J. A. K. Suykens, Multi-View Kernel Spectral
%Clustering, Internal Report 17-71, ESAT-SISTA, KU Leuven (Leuven, Belgium), 2017.

% Dataset : D. Greene, P. Cunningham, A matrix factorization approach for integrating multiple
%data views, European Conference on Machine Learning and Principles and Practice
%of Knowledge Discovery in Databases (2009) 423{438.

clear all
close all
addpath('MVKSCutils');

%% load data
load('data3Sources');

%% Settings - parameters obtained by tuning
t=5.1200;
gamma=[54.2679,95.5312,92.9132];
k=6;
d=2;

%% Run algorithm
model = MVKSC(X,'normpoly_kernel',[t,d],k,gamma,'mean',X);

%% Evaluate performance
NMI_avg= getNMI(model.qtest,Y);
ARI_avg = getARI(model.qtest,Y);
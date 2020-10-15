#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:49:19 2020

@author: koen
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#matrice A
def matrice_A(n):
    C = np.eye(n)
    B = np.zeros((n,n))
    B[0,:] = np.ones(n)
    A = np.vstack((B,C))    
    for i in range(1,n):
        B = np.zeros((n,n))
        B[i,:] = np.ones(n)
        D = np.vstack((B,C))

        A = np.hstack((A,D))
    return A
        
        
F=matrice_A(4)

def solveOT(mu,nu,C):
    A = matrice_A(len(mu))
    b = np.hstack((mu,nu))

    sol = opt.linprog(C,A_eq=A,b_eq=b,bounds = (0,None),method = 'simplex')
    gamma = np.reshape(sol.x, (len(mu),len(mu)))
    plt.matshow(gamma)
    return sol
    
def mu(x):
    n = len(x)
    xnew = np.zeros(n)
    for i in range(n):
        if x[i] <= 1 and x[i]>=0:
            xnew[i] = 1
    return xnew

def vu(x):
    return np.exp(-10*(x-0.5)**2)

x = np.linspace(0,1,20)

def c(x,y):
    return abs(x-y)**2

mu_ = mu(x)/sum(mu(x))
vu_ = vu(x)/sum(vu(x))

def matrice_C(c,x,y):
    n = len(x)
    C = np.zeros((n,n))
    for i in range(n):
        for j  in range(n):
            C[i,j]= c(x[i],y[j])
    C = C.flatten()
    return C

C = matrice_C(c,x,x)
sol1 = solveOT(mu_,vu_,C)

def solveOTdual(mu,vu,C):
    n = len(mu)
    A = np.transpose(matrice_A(n))
    b = np.hstack((mu,vu))
    sol = opt.linprog(-b,A_ub=A,b_ub=C,bounds= (None,None),method = 'simplex')
    return sol.x , sol.slack, sol

sol2x, slack, sol2 = solveOTdual(mu_,vu_,C)
u_opt = sol2x[0:20]
v_opt = sol2x[20:40]
plt.figure()
plt.plot(u_opt, label='u*')
plt.plot(v_opt,label = 'v*')
plt.legend()

print(np.dot(sol1.x,slack))
N=10
x = np.random.rand(N)
y= np.random.rand(N)

mu1 = np.ones(N)/N
mu1 = mu1
vu1 = mu1
C1 = matrice_C(c,x,y)
sol3 = solveOT(mu1,vu1,C1)
gamma_etoile = np.reshape(sol3.x,(N,N))

def c2(x,y):
    return (abs(x[0]-y[0])**2 + abs(x[1]-y[1])**2)

def matrice_C2(c,x,y):
    n = len(x[:,1])
    C = np.zeros((n,n))
    for i in range(n):
        for j  in range(n):
            C[i,j]= c2(x[i,:],y[j,:])
    C = C.flatten()
    return C

x = np.random.rand(N,2)
y= np.random.randn(N,2)

mu1 = np.ones(N)/N
mu1 = mu1
vu1 = mu1
C2 = matrice_C2(c2,x,y)
sol4 = solveOT(mu1,vu1,C2)
gamma_etoile = np.reshape(sol3.x,(N,N))

plt.figure()

plt.scatter(x[:,0],x[:,1],c='blue',label = 'x')
plt.scatter(y[:,0],y[:,1],c='red',label = 'y')
plt.legend()
map_i_to_permi = np.nonzero(gamma_etoile)[1]



yis = np.concatenate((x[:,1],y[map_i_to_permi,1]))

plt.plot([x[:,0], y[map_i_to_permi,0]],[x[:,1],y[map_i_to_permi,1]],'--',c='grey')
plt.legend()

def GradDualReg(mu,vu,C,eps,t):
    N =len(mu)
    C = np.reshape(C,(N,N))
    u = np.zeros(N)
    grad =  mu - np.sum([np.exp((u-C[:,j])/eps)*vu[j]/np.sum(np.exp((u-C[:,j])/eps), axis =0) for j in range(N)],axis=0)
    norm = np.linalg.norm(grad)
    k = 0
    while norm> 1e-8 and k<5000:
        u = u + t*grad
        grad = mu - np.sum([np.exp((u-C[:,j])/eps)*vu[j]/np.sum(np.exp((u-C[:,j])/eps), axis =0) for j in range(N)],axis=0)
        norm = np.linalg.norm(grad)
        k+=1
        
    v = np.zeros(N)
    for i in range(N):
        v[i] = np.amin(C[:,i]-u)
    return u, v, k 
  
x = np.linspace(0,1,20)
C = matrice_C(c,x,x) 
eps = [1,0.5,0.1,0.05,0.01,.005]
t=0.1
err_graddual = []
err_graddual2 = []
for i in eps:
    u,v, k = GradDualReg(mu_,vu_,C, i, t)
    err_graddual.append(np.linalg.norm(u-u_opt))
    err_graddual2.append(np.linalg.norm(v-v_opt))

    print(k)
    plt.figure(8)
    plt.plot(v,label = str(i))
    plt.legend()
    plt.figure(9)
    plt.plot(u, label = str(i))
    plt.legend()
plt.figure(8)
plt.plot(v_opt,label='sol exacte')
plt.legend()
plt.figure(9)
plt.plot(u_opt,label ='sol exacte')
plt.legend()
plt.figure(20)
plt.loglog(eps,err_graddual)
plt.figure(21)
plt.loglog(eps,err_graddual2)

def F(gamma,C,eps): 
    n = len(gamma)
    a = eps*np.dot(gamma,(np.log(gamma) +C/eps) - np.ones(n))
    return a

def gradF(u,v,mu,vu,C,eps):
    n = len(u)
    C = np.reshape(C,(n,n))
    grad1 = np.zeros(n)
    grad2 = np.zeros(n)
    for j in range(n):
        grad1 -= np.exp((u+ v[j]*np.ones(n) - C[:,j])/eps)
    for j in range(n):
        grad2 -= np.exp((v+ u[j]*np.ones(n) - C[j,:])/eps)
    grad1 = mu + grad1
    grad2 = vu + grad2
    grad = np.hstack((grad1,grad2))
    return grad

def EntDual(mu,vu,C,eps,t):
    n =len(mu)
    uv = np.zeros(2*n)
    u= uv[0:n]
    print(np.shape(u))
    v = uv[n:2*n]
    print(np.shape(v))
    grad = gradF(u,v,mu_,vu_,C,eps)
    print(grad)
    norm = np.linalg.norm(grad)
    k = 0
    gamma_ent_dual = np.eye(n)
    while norm> 1e-5 and k<10000:
        uv = uv + t*grad
        u= uv[0:n]
        v = uv[n:2*n]
        grad = gradF(u,v,mu_,vu_,C,eps)
        norm = np.linalg.norm(grad)
        k+=1
    print(norm)
    C_reshaped = np.reshape(C,(n,n))
    for i in range(n):
        for j in range(n):
           gamma_ent_dual[i,j] = np.exp(u[i]/eps)*np.exp(v[j]/eps)*np.exp(-C_reshaped[i,j]/eps)
            
    return u, v, gamma_ent_dual, k
err1_ent = []
err2_ent=[]
for i in [1,0.5,0.1,0.05,0.01,.005]:
    u_ent, v_ent, gamma_entdual , k_ent = EntDual(mu_,vu_,C,i,0.1)
    err1_ent.append(np.linalg.norm(u_ent-u_opt))
    err2_ent.append(np.linalg.norm(v_ent - v_opt))
    plt.figure(10)
    plt.plot(u_ent, label = 'eps='+str(i))
    plt.legend()
    plt.figure(11)
    plt.plot(v_ent,label = 'eps='+str(i)) 
    plt.legend()
plt.figure(10)
plt.plot(u_opt, label ='sol optimale')
plt.ylim([-0.2,0.2])
plt.legend()
plt.figure(11)
plt.plot(v_opt, label = 'sol optimale')
plt.ylim([-0.2,0.2])
plt.legend()

plt.matshow(gamma_entdual)

def Sinkhorn(mu,vu,C,eps,N):
    n = len(mu)
    v = np.zeros(n)
    b = np.exp(v/eps)
    C_reshaped = np.reshape(C,(n,n))
    a = np.zeros(n)
    for k in range(N):
        for i in range(n):
            a[i] = mu[i]/np.sum(np.dot(b,np.exp(-C_reshaped[i,:]/eps)))
            b[i] = vu[i]/np.sum(np.dot(a,np.exp(-C_reshaped[:,i]/eps)))
    u = np.log(a)*eps
    v = np.log(b)*eps
    gamma = np.eye(n)
    for i in range(n):
        for j in range(n):
            gamma[i,j] = a[i]*b[j]*np.exp(-C_reshaped[i,j]/eps)
    return u, v , gamma

err1_sink =[]
err2_sink =[]
for i in [1,0.5,0.1,0.05,0.01,.005]:
    u_sink,v_sink,gamma_sink = Sinkhorn(mu_,vu_,C,i,1000)
    err1_sink.append(np.linalg.norm(u_sink - u_opt))
    err2_sink.append(np.linalg.norm(v_sink-v_opt))
    
    plt.figure(13)
    plt.plot(u_sink, label = 'eps ='+ str(i) )
    plt.legend()
    plt.figure(14)
    plt.plot(v_sink, label= 'eps =' + str(i))
    plt.legend()

plt.figure(13)
plt.plot(u_opt, label = 'sol optimale')
plt.ylim([-0.1,0.1])
plt.legend()
plt.figure(14)
plt.plot(v_opt, label = 'sol optimale')
plt.ylim([-0.1,0.1])
plt.legend()

plt.matshow(gamma_sink)

def Sinkhorn3(mu,vu,C,eps,N):
    n = len(mu)
    v = np.zeros(n)
    b = np.exp(v/eps)
    C_reshaped = np.reshape(C,(n,n))
    a = np.zeros(n)
    for k in range(N):
        for i in range(n):
            a[i] = mu[i]/np.sum(np.dot(b,np.exp(-C_reshaped[i,:]/eps)))
            b[i] = vu[i]/np.sum(np.dot(a,np.exp(-C_reshaped[:,i]/eps)))
    u = np.log(a)*eps
    v = np.log(b)*eps
    gamma = np.eye(n)
    for i in range(n):
        for j in range(n):
            gamma[i,j] = a[i]*b[j]*np.exp(-C_reshaped[i,j]/eps)
    return u, v , gamma
        
plt.figure(30)
plt.plot(eps,err1_sink,label = 'Sinkhorn u')

plt.plot(eps,err1_ent,label = 'Entropie u')

plt.plot(eps,err_graddual,label = 'Graddual u')
plt.ylim([0,0.1])
plt.legend()


plt.figure(31)
plt.plot(eps,err2_sink, label = 'Sinkhorn v')
plt.plot(eps,err2_ent, label = 'Entropie v')
plt.plot(eps,err_graddual2, label = 'Graddual v')
plt.ylim([0,1])
plt.legend()
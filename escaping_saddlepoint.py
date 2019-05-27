#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:05:39 2019

@author: hango
"""
import numpy as np
from matplotlib import pyplot as plt
import math 
import time



#method1 escaping from the saddle point using perturbation
class saddlepoint:
    def __init__(self,x0=None,l=1,row=1,epsilon=0.01, c=1, delta=0.1,Delta=3,eigenvalue=[-0.005,1,1],max_itr=500):
        self.eigenvalue=eigenvalue
        self.eta=1/l
        self.max_itr=max_itr
        self.epsilon=epsilon
        self.d=len(eigenvalue)
        self.grad=np.zeros([self.d,1])
        if not x0.any():
            self.x=np.random.randn(self.d).reshape(self.d,1)
        else :
            self.x=x0
        self.x0=self.x
        self.x_tnoise=self.x
        self.hessian=np.diag(eigenvalue)
        self.prediction=self.obj_func(self.x)
        self.pred_seq=[]
        
        #in this case Delta can be computed directly   
        Delta=self.obj_func(self.x)


        self.chisq=3*max(np.log(self.d*l*Delta/(c*epsilon**2*delta)),4)
        self.eta=c/l
        self.r=c**(1/2)*epsilon/self.chisq**2/l*10000
        self.g_thres=c**(1/2)*epsilon/self.chisq**2*4000
        self.f_thres=c/self.chisq**3*(epsilon**3/row)**(1/2)*1000000
        self.t_thres=math.floor(self.chisq/c**2*l/(row*epsilon)**(1/2))
        self.t_noise=-self.t_thres-1

    def obj_func(self,x):
        return np.sum(np.dot(np.dot(x.T,self.hessian),x))
    def gradient(self,x):
        self.grad=np.dot(2*self.hessian,x)
    def fit(self,method='GD',verbose=True):
        assert method in ['SGD','GD','PGD']
        if method=='SGD':
            return self._sgd(verbose)
        elif method == 'GD':
            return self._gd(verbose)
        elif method== 'PGD':
            return self._pgd(verbose)
    def _gd(self,verbose=True):
        self.grad+=1
        self.x=self.x0
        self.pred_seq=[]
        i=0
        while np.linalg.norm(self.grad,ord=2)>self.epsilon*1 and i<self.max_itr:
            self.gradient(self.x)
            self.x=self.x-self.eta*self.grad
            self.prediction=self.obj_func(self.x)
            self.pred_seq.append(self.prediction)
            i=i+1
            if verbose:
                #print("iteration:{0},prediction={1:04f}".format(i,self.prediction))
                print("gradient:%1.04f,iteration:%d,prediction:%1.04f"%(np.linalg.norm(self.grad,ord=2),i,self.prediction))
    def _pgd(self,verbose=True):
        self.grad+=1
        self.x=self.x0
        self.pred_seq=[]
        t=0
        while True:
            if np.linalg.norm(self.grad,ord=2)<=self.g_thres and t-self.t_noise>self.t_thres:
                self.t_noise=t
                self.x_tnoise=self.x
                print('Before perturbabtion')
                print('x[0]:%1.04f,x[1]:%1.04f,x[2]:%1.04f'%(self.x[0][0],self.x[1][0],self.x[2][0]))
                self.__perturbation(self.r)
                print('After perturbabtion')
                print('x[0]:%1.04f,x[1]:%1.04f,x[2]:%1.04f'%(self.x[0][0],self.x[1][0],self.x[2][0]))

            if t-self.t_noise==self.t_thres and self.obj_func(self.x)-self.obj_func(self.x_tnoise)> -self.f_thres:
                break;
            elif t>self.max_itr :
                break;
            else :
                self.gradient(self.x)
                self.x=self.x-self.eta*self.grad
                self.prediction=self.obj_func(self.x)
                self.pred_seq.append(self.prediction)
            t=t+1
            if verbose:
                print("gradient:%1.04f,iteration:%d,prediction:%1.04f"%(np.linalg.norm(self.grad,ord=2),t,self.prediction))


    def __perturbation(self,r):
        rand=np.random.uniform(0,2*math.pi,size=self.d-1)
        for i in range(self.d):
            if i ==0:
                self.x[i][0]+=self.__multi_cos(rand[0:self.d-i-1])*r
            else :
                self.x[i][0]+=self.__multi_cos(rand[0:self.d-i-1])*math.sin(rand[self.d-i-1])*r
            
    def __multi_cos(self,a):
        if not a.any():
            return 1
        else:
            result=1
            for i in range(len(a)):
                result=result*math.cos(a[i])
        return result
    

#test for method_escaping from the saddle point by random permutation   
#x0=np.array([0,2,3])
time_start=time.time()
x0=np.random.uniform(1,6,3)
result=saddlepoint(x0=x0.reshape(3,1),l=1,row=10,epsilon=0.01, c=1, delta=0.1,Delta=3,eigenvalue=[-0.01,0.85,0.85],max_itr=1000)
result.fit('GD',verbose=False)
plt.ylabel('objective function ')
plt.xlabel('test number')
plt.plot(result.pred_seq)
time_end=time.time()
print('totally cost',time_end-time_start)





'''
t=[]
for i in range(300):
    eps=0.0001+0.00001*i
    time_start=time.time()
    result=saddlepoint(x0=x0.reshape(3,1),l=1,row=10,epsilon=eps, c=1, delta=0.1,Delta=3,eigenvalue=[0.61,0.85,0.85],max_itr=10000)
    result.fit('PGD',verbose=False)
    plt.plot(result.pred_seq)
    time_end=time.time()
    t.append((time_end-time_start)*eps**2/np.log(3*1*3/eps**2/0.1)**4)
    
t=[]
for d in range(500):
    eigen=np.random.randint(2,8,d)
    time_start=time.time()
    result=saddlepoint(x0=x0.reshape(3,1),l=1,row=10,epsilon=0.01, c=1, delta=0.1,Delta=3,eigenvalue=[0.61,0.85,0.85],max_itr=10000)
    result.fit('PGD',verbose=False)
    plt.plot(result.pred_seq)
    time_end=time.time()
    t.append((time_end-time_start)/np.log(d*3/0.01**2/0.1)**4)
    
    
t   
plt.ylabel('t/log(d*l*Delta_f/delta/eps^2)^4')
plt.xlabel('test numbe')
plt.plot(t[120:500])
plt.show()    
    
    
result.fit('GD',verbose=False)
plt.plot(result.pred_seq)
#in this case we can see that random permutation did help escaping from the saddle point.

'''



#Another method-- Natasha 

#part1 Natasha 1.5
import pandas as pd
class Natasha:
    def __init__(self,n,epsilon,sigma,L,L2,alpha,eigenvalue,v,ddelta,Delta_f,x0):
        self.n=n
        self.L=L
        self.L2=L2
        self.ddelta=ddelta
        self.p0=len(x0)
        self.p=math.floor((sigma/epsilon/L)**(2/3))
        self.B=math.floor(1/epsilon**2*0.001)
        self.m=math.floor(self.B/self.p*10)
        self.T=L**(2/3)*sigma**(1/3)/epsilon**(10/3)
        self.T2=math.floor(self.T/self.B*0.001)
        self.obj_eig_value=self.init_eigen(eigenvalue)
        self.x0=x0.reshape([self.p0,1])
        self.x1=self.x0
        self.x2=self.x0
        self.y=self.x0
        self.mu=np.zeros([self.p0,1])
        self.X=[]
        self.delta=np.zeros([self.p0,1])
        self.sigma=sigma
        self.alpha=alpha
        self.sigma_til=L2*v**(1/3)*epsilon**(1/3)/ddelta
        self.prediction=[]
        if L*ddelta/v**(1/3)/epsilon**(1/3):
            self.L_til=self.sigma_til
        else :
            self.L_til=L
            self.sigma_til=max(v*epsilon*L2**(3)/L**2/ddelta**(3),epsilon*L/v**(1/2))
        
        self.N1=self.sigma_til*Delta_f/self.p/epsilon**2*0.01
        self.Y=[]
        self.y_k=self.x0
        
            
        
    def objective_func(self,x,eigenvalue)  :
        return np.sum(np.dot(np.dot(x.T,np.diag(eigenvalue)),x))
    def init_eigen(self,eigenvalue):
        result=np.zeros([self.n,len(eigenvalue)])
        for i in range(self.n):
            result[i]=eigenvalue+np.random.randn(len(eigenvalue))*0.05*self.L
        return result
    def gradient(self,x,eigenvalue):
        if np.linalg.norm(x-self.y_k)<=self.ddelta/self.L2:
            return np.dot(2*np.diag(eigenvalue),x.reshape([self.p0,1]))
        else :
            return  np.dot(2*np.diag(eigenvalue),x.reshape([self.p0,1]))+2*self.L*(np.linalg.norm(x-self.y_k)-self.ddelta/self.L2)*(x-self.y_k)/np.linalg.norm(x-self.y_k)
        
        
    def Natasha1_5(self,x0):
        self.X=[]
        self.x0=x0
        self.x1=x0
        self.x2=x0
        for k in range(self.T2):
            self.x2=self.x1
            self.mu_renew()
            for s in range(self.p):
                x_choose=np.zeros([self.p0,self.m+1])
                x_choose[:,0]=self.x1.reshape([1,self.p0])
                self.X.append(self.x1)
                for t in range(self.m):
                    self.delta_renew(x_choose[:,t].reshape([self.p0,1]))
                    x_choose[:,t+1]=x_choose[:,t]-self.alpha*self.delta.reshape([1,self.p0])
                self.x1=x_choose.mean(1).reshape(self.p0,1)
            
        #self.y=self.mean_list(self.X)    
        self.y=self.X[-1]
        x_output=self.sgd(self.y,self.alpha,100)
        return x_output
          
    def mu_renew(self):
        result=np.zeros([self.p0,1])
        for i in range(self.B):
            result=result+self.gradient(self.x2,self.obj_eig_value[np.random.choice(self.n)])
        result=result/self.B
        self.mu=result
        
    def delta_renew(self,x):
        i=np.random.choice(self.n)
        self.delta=self.gradient(x,self.obj_eig_value[i])-self.gradient(self.x2,self.obj_eig_value[i])+self.mu.reshape([self.p0,1])+2*self.sigma*((x-self.x1))
        
    def mean_list(self,X):
        return sum(X)/len(X)
    
    def sgd(self,y,eta,maxitr):
        x_iter=y
        for j in range(maxitr):
            i=np.random.choice(self.n)
            delta=self.gradient(x_iter,self.obj_eig_value[i])+2*self.sigma*(x_iter-y)
            x_iter=x_iter-eta*delta
        return x_iter
        
    
    #Oja_algorithm 
    #input the basic eta,p,L,delta,d,C and output [judge,v]
    #if judge ==yes means we found the vector v so that we can move in the direction of v
    #if judge ==False means the minimum eigenvalue is above the threshold, so wecan go into the first order step
    def Oja_alg(self,eta,p,L,delta,d,C):
        T1=math.floor(np.log(1/p))
        T2=math.floor(12**2*C**2*L**2/delta**2*(np.log(d/p))**2)
        s=[]
        vector_s=[]
        for k in range(T1):
            W=[]
            a=np.random.uniform(0,1,d)
            W.append(a/(sum(a**2))**(1/2))
            sum_eigen=0
            for i in range(T2-1):
                #从n里面抽样
                eigen=self.obj_eig_value[np.random.choice(self.n)]
                #迭代得到下一个w
                times=np.dot((np.identity(d)+eta*  (0.5*np.identity(d)-np.diag(eigen)/2/L   )    ),W[i])
                W.append(times/sum(times**2)**(1/2))
                sum_eigen=sum_eigen+eigen
            sum_eigen=sum_eigen/T2
            #从0...T2-1中随机抽取一个W_i
            i_rand=np.random.choice(T2)   
            #计算得到s
            s.append(np.dot(np.dot(W[i_rand],(0.5*np.identity(d)-np.diag(sum_eigen)/2/L) ),W[i_rand].reshape([d,1])))
            vector_s.append(W[i_rand])
        smin=max(s)
        row=L-2*L*smin
        v=vector_s[s.index(smin)]
        if row>=-4*C*2*L*np.log(d/p)/T2**(1/2):
            judge=False
        else :
            judge=True
        return [judge,v]    

    
    
    def Natasha2(self,y0,eps,delta):
        count=0
        count2=0
        while(True):
            result=self.Oja_alg(eta=0.5,p=0.0001,L=1,delta=delta,d=3,C=2*10**(-3))
            if result[0]==True:
                self.y_k=self.y_k+(np.random.choice(2)*2-1)*self.ddelta/self.L2*result[1].reshape([self.p0,1])
            else:
                self.y_k=self.Natasha1_5(self.y_k)
                count=count+1
                self.Y.append(self.y_k)
            count2=count2+1
            self.prediction.append(self.objective_func(self.y_k,np.sum(self.obj_eig_value,axis=0)/self.n ))
            if count>=self.N1 or count2>100:
                break
        return self.y_k
    
        
            
        
        
 
result=Natasha(n=50,epsilon=0.01,sigma=1,L=1,L2=1,alpha=0.01,eigenvalue=[-0.05,0.85,0.85],v=1,ddelta=1,Delta_f=10,x0=np.array([1,2,3]))
result.Natasha1_5(result.x0)
result.Oja_alg(eta=0.5,p=0.0001,L=1,delta=0.05,d=3,C=2*10**(-3))
result.Natasha2(result.x0,eps=0.01,delta=0.05)









     
#test for Natasha1.5

t=[]
for i in range(50):
    L=1+0.01*i
    result=Natasha(n=50,epsilon=0.01,sigma=1,L=1,L2=1,alpha=0.01,eigenvalue=[-0.05,0.85,0.85],v=1,ddelta=1,Delta_f=10,x0=np.array([1,2,3]))
    t_start=time.time()
    result.Natasha1_5(result.x0)
    t_end=time.time()
    t.append((t_end-t_start)/L**(2/3))
    
    
plt.ylabel('t*eps^(3.25)')
plt.xlabel('test number')
plt.plot(t)
plt.show()


 


#test for Natasha2

t=[]
for i in range(50):
    eps=0.01+0.001*i
    result=Natasha(n=50,epsilon=0.01,sigma=1,L=1,L2=1,alpha=0.01,eigenvalue=[0.2,0.8,0.8],v=1,ddelta=1,Delta_f=10,x0=np.array([1,2,3]))
    #result.Natasha1_5(result.x0)
    #result.Oja_alg(eta=0.5,p=0.0001,L=1,delta=0.05,d=3,C=2*10**(-3))
    t_start=time.time()
    result.Natasha2(result.x0,eps=0.01,delta=1)
    t_end=time.time()
    t.append((t_end-t_start)*eps**(3.25))
    
    
plt.ylabel('t/L^(2/3)')
plt.xlabel('test number')
plt.plot(t)
plt.show()


    
    
     
     
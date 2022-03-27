# -*- coding: utf-8 -*-

import numpy as np

from fealpy.decorator import cartesian

pi = np.pi
cos = np.cos
sin = np.sin

class PDE2():
    
    """
    
    -div(c(x) u'(x))= f in Omega
    u = 0  on Omega's boundary
    c(x) = 1/((2+P*sin(2*pi*x/epsion)*(2+P*sin(2*pi*y/epsion))
    f(x) = -1,     Omega = (0,1)*(0,1)
    """
    def __init__(self, P, eps, box=np.array([0, 1, 0, 1])):
        self.P = P
        self.eps = eps
        self.box = box
        
        self.x0 = box[0]
        self.x3 = box[1]
        self.hx = self.x3 - self.x0
        self.y0 = box[2]
        self.y3 = box[3]
        self.hy = self.y3 - self.y0 

    def domain(self):
        return self.box
    
    
    @cartesian # 函数增加输入参数坐标类型的标签 coordtype，这里指明是 笛卡尔 坐标 
    def c(self, p):      
#        print('This is function c')
        x = p[..., 0]
        y = p[..., 1]
        
        ## pde model 1
        val = 2+self.P*sin(2*pi*x/self.eps)
        val *= 2+self.P*sin(2*pi*y/self.eps)
        val = 1/val
        
        # ### pde model 2
        # val = 2+self.P*sin(2*pi*(x-y)/self.eps)
        # val = 1/val
        
        ### pde model 3
        # eps1 = 0.2
        # eps2 = 0.8
        # val = (1.5+sin(2*pi*x/eps1))*(1.5+cos(2*pi*y/eps1))
        # val /= (1.5+cos(2*pi*x/eps2))*(1.5+sin(2*pi*y/eps2))
        
        ### pde model 4
        # val = (2+self.P*sin(2*pi*x/self.eps))/(2+self.P*cos(2*pi*y/self.eps))
        # val += (2+sin(2*pi*x/self.eps))/(2+self.P*sin(2*pi*y/self.eps))
        
        return val 

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
#        print("source.p.shape: ",p.shape)
        x = p[..., 0]
        y = p[..., 1]
        
        ##### pde model 0
        # val = np.ones_like(p[...,0])
        
        #### pde model 1
        val = sin(pi*x/self.eps)*sin(pi*y/self.eps)
        val += self.P*cos(pi*x/self.eps)*sin(pi*y/self.eps)*cos(2*pi*x/self.eps)/(
            2+self.P*sin(2*pi*x/self.eps))
        val += self.P*cos(pi*y/self.eps)*sin(pi*x/self.eps)*cos(2*pi*y/self.eps)/(
            2+self.P*sin(2*pi*y/self.eps))
        val *= 2*pi*pi/self.eps/self.eps/(
            (2+self.P*sin(2*pi*x/self.eps))*(2+self.P*sin(2*pi*y/self.eps)))
        
        #### pde model 2
        # val = sin(pi*x)*sin(pi*y)
        # val += self.P/self.eps*cos(pi*x)*sin(pi*y)*cos(2*pi*x/self.eps)/(2+self.P*sin(2*pi*x/self.eps))
        # val += self.P/self.eps*cos(pi*y)*sin(pi*x)*cos(2*pi*y/self.eps)/(2+self.P*sin(2*pi*y/self.eps))
        # val *= 2*pi*pi/((2+self.P*sin(2*pi*x/self.eps))*(2+self.P*sin(2*pi*y/self.eps)))
        
        ##### pde model 3
        # val = 2*(x-x*x+y-y*y)
        # val += (1-2*x)*(y-y*y)*self.P*2*pi/self.eps*cos(2*pi*x/self.eps)/(2+self.P*sin(2*pi*x/self.eps))
        # val += (1-2*y)*(x-x*x)*self.P*2*pi/self.eps*cos(2*pi*y/self.eps)/(2+self.P*sin(2*pi*y/self.eps))
        # val /= ((2+self.P*sin(2*pi*x/self.eps))*(2+self.P*sin(2*pi*y/self.eps)))
        
        ##### pde model 4
        # val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
        
        # ##### pde model 5
        # val = (6*x*x-1)*(y*y-1)*y*y
        # val += (6*y*y-1)*(x*x-1)*x*x
        # val /= -2
        
        ####### pde model 6, c=c1
        # eps2 = 0.08
        # val = sin(pi*x/eps2)*sin(pi*y/eps2)/eps2
        # val += self.P*cos(2*pi*x/self.eps)*sin(pi*y/eps2)*cos(pi*x/eps2)/(
        #     2+self.P*sin(2*pi*x/self.eps))/self.eps
        # val += self.P*cos(2*pi*y/self.eps)*sin(pi*x/eps2)*cos(pi*y/eps2)/(
        #     2+self.P*sin(2*pi*y/self.eps))/self.eps
        # val *= 2*pi*pi/eps2/(
        #     (2+self.P*sin(2*pi*x/self.eps))*(2+self.P*sin(2*pi*y/self.eps)))
        
        
        return val
    
    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        
        # ##### pde model 1
        val = sin(pi*x/self.eps)*sin(pi*y/self.eps)
        # ##### pde model 2
        # val = sin(pi*x)*sin(pi*y)
        # ##### pde model 3
        # # val = x*(1-x)*y*(1-y)
        # ##### pde model 4
        # eps2 = 0.08
        # val = sin(pi*x/eps2)*sin(pi*y/eps2)
        return val
    
    @cartesian
    def subsource(self, p):
        return np.zeros_like(p[...,0])
    
    @cartesian
    def dirichlet(self, p):
        # return np.zeros_like(p[...,0])
        return self.solution(p)
import time
import numpy as np
from scipy.sparse import  spdiags
from scipy.sparse.linalg import spsolve

from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.mesh import MeshFactory as MF


import sys
sys.path.append('..')
from fealpython.PDEmodel_MsFEM1 import PDE2
from fealpython.MultiscaleFiniteElementSpace import MultiscaleFiniteElementSpace



'''------------------------------------FEM----------------------------------'''

def FEM(box, nx, ny, q):
    pde = PDE2(P, eps, box)
    mesh = MF.boxmesh2d(box, nx, ny, meshtype='quad', p=1)
    # 在 mesh 上创建一个双 p 次的有限元函数空间
    space = ParametricLagrangeFiniteElementSpace(mesh, p=1, spacetype='C', q=q)
    # 数值解函数
    uh = space.function()    
    # 组装刚度矩阵
    A = space.stiff_matrix(c=pde.c, q=q)   # c:变系数的系数  q: GaussLengendre积分次数   
    # 右端载荷  
    F = space.source_vector(pde.source, q=q)
    # 定义边界条件
    bc = DirichletBC(space, pde.dirichlet)
    # 处理边界条件
    A, F = bc.apply(A, F, uh)
    # 求解
    uh[:] = spsolve(A, F).reshape(-1)
    
    error = space.integralalg.L2_error(pde.solution, uh)
    return uh, error



'''----------------------------------MsFEM----------------------------------'''

def MsFEM(box, nxc, nyc, ur, nx, ny, nxf=8, nyf=8, qth=None, basisboundarytype='L'):
    """
    

    Parameters
    ----------
    box : 求解区域.
    nxc : 粗网格 x 方向剖分段数.
    nyc : 粗网格 y 方向剖分段数.
    ur : FEM 参考解.
    nx : FEM 参考解 x 方向剖分段数.
    ny : FEM 参考解 y 方向剖分段数.
    nxf : 细网格 x 方向剖分段数. 默认值是 8.
    nyf : 细网格 y 方向剖分段数. 默认值是 8.
    qth : 高斯积分阶数. 默认值是 None.
    basisboundarytype : 多尺度有限元子问题边界条件类型. 默认值是 'L'.

    Returns
    -------
    error_u : 真解误差.
    error_ur : FEM 数值解误差.

    """
    # PDE 模型
    pde = PDE2(P, eps, box)
    qth = qth if qth is not None else 4
    # 空间生成
    space = MultiscaleFiniteElementSpace(pde.c, box, nxc, nyc, 
                                         nxf, nyf, qth, basisboundarytype)
    # 基函数与刚度矩阵的生成
    Phi, A = space.basis_stiffmatrix(q=qth)
    # 载荷向量的生成
    F = space.source_vector(pde.source, Phi=Phi, q=qth)
    # 边界点生成及边界处理
    idx = space.is_boundary_dof()
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[idx] = 1
    Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    A = T@A@T + Tbd
    F[idx] = 0
    # 数值解
    uh_ms = spsolve(A, F).reshape(-1)
    # 真解误差
    error_u = space.L2_error_u(pde.solution, uh_ms, Phi, qth)
    # 精细有限元解误差
    error_ur = space.L2_error_FEM_u_reference(np.array(u_FEM), uh_ms,
                                              nx, ny, Phi, q=qth)
    return error_u, error_ur


mylog = open('data_space.log', mode='a', encoding='utf-8')
print("The code Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=mylog)
mylog.close()


pi = np.pi

P = 1.8
# P=1

# q = 20


#eps = 1/15*pi   #epsilon
#eps = 0.64/nxc   #epsilon
eps = 0.25
# eps = 8e-03
#eps = 0.0001
# eps = 1
# eps = 0.16


# 求解区域
box = np.array([0, 1, 0, 1])

# 有限元精细解
NX = NY = 2**7
u_FEM, error = FEM(box, NX, NY, q=20)

n1 = 1
n2 = 7


er_FEM = np.zeros(n2-n1)         # 有限元误差
er_Lms1 = np.zeros(n2-n1)        # 线性边界多尺度有限元真解误差
er_Lms2 = np.zeros(n2-n1)        # 线性边界多尺度有限元数值解误差
er_Oms1 = np.zeros(n2-n1)        # 振荡边界多尺度有限元真解误差
er_Oms2 = np.zeros(n2-n1)        # 振荡边界多尺度有限元数值解误差

ords_FEM = np.zeros(n2-n1-1)     # 有限元误差收敛阶
ords_Lms1 = np.zeros(n2-n1-1)    # 多尺度有限元真解误差收敛阶
ords_Lms2 = np.zeros(n2-n1-1)    # 多尺度有限元数值解误差收敛阶
ords_Oms1 = np.zeros(n2-n1-1)    # 多尺度有限元真解误差收敛阶
ords_Oms2 = np.zeros(n2-n1-1)    # 多尺度有限元数值解误差收敛阶

# 计算误差
for i in range(n1, n2):
    nxc = nyc = 2**i
    uh_FEM, er_FEM[i-n1] = FEM(box, nxc, nyc, q=20)
    er_Lms1[i-n1], er_Lms2[i-n1] = MsFEM(box, nxc, nyc, u_FEM, NX, NY,
                                       qth=20, basisboundarytype='L')
    er_Oms1[i-n1], er_Oms2[i-n1] = MsFEM(box, nxc, nyc, u_FEM, NX, NY,
                                       qth=20, basisboundarytype='O')

# 计算误差收敛阶
for i in range(n2-n1-1):
    ords_FEM[i] = -np.log(er_FEM[i+1]/er_FEM[i]) /np.log(2)
    ords_Lms1[i] = -np.log(er_Lms1[i+1]/er_Lms1[i]) /np.log(2)
    ords_Lms2[i] = -np.log(er_Lms2[i+1]/er_Lms2[i]) /np.log(2)
    ords_Oms1[i] = -np.log(er_Oms1[i+1]/er_Oms1[i]) /np.log(2)
    ords_Oms2[i] = -np.log(er_Oms2[i+1]/er_Oms2[i]) /np.log(2)


mylog = open('data.log', mode='a', encoding='utf-8')
print("The code Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=mylog)
mylog.close()

# 打印结果
mylog = open('data_space.log', mode='a', encoding='utf-8')
print("er_FEM: ", er_FEM, file=mylog)
print("er_Lms1: ", er_Lms1, file=mylog)
print("er_Lms2: ", er_Lms1, file=mylog)
print("er_Oms1: ", er_Oms2, file=mylog)
print("er_Oms2: ", er_Oms2, file=mylog)
print("ords_FEM: ", ords_FEM, file=mylog)
print("ords_Lms1: ", ords_Lms1, file=mylog)
print("ords_Lms2: ", ords_Lms2, file=mylog)
print("ords_Oms1: ", ords_Oms1, file=mylog)
print("ords_Oms2: ", ords_Oms2, file=mylog)
print("\n", file=mylog)
mylog.close()

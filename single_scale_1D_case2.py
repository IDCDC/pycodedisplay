import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh


### 生成网格对象
def mesh_generator(N=10, xs=0, xe=1):
    """
    

    Parameters
    ----------
    N : int, optional
        剖分段数. The default is 10.
    xs : float, optional
        求解区域左端点. The default is 0.
    xe : float, optional
        求解区域右端点. The default is 1.

    Returns
    -------
    mesh : fealpy.IntervalMesh
        fealpy 的区间网格调用.

    """
    node = np.linspace(xs, xe, N+1)
    cell = np.c_[np.arange(N), np.arange(1, N+1)]
    mesh = IntervalMesh(node, cell)
    return mesh
    
### 构造 PDE 模型
class PDE:
    def __init__(self, a, b, eps):
        self.a = a
        self.b = b
        self.eps = eps
    
    # 真解
    def solution(self, p):
        val = np.cos(p/self.eps) * self.eps / self.a * (p-0.5)
        val += self.eps / self.a / 2 + p*self.b/self.a/2
        val -= p**2 * self.b/self.a/2 
        val -= np.sin(p/self.eps) * self.eps**2 / self.a
        return val
    
    # 源项
    def source(self, p=None):
        return np.ones_like(p)
    
    # 渗透率场函数
    def k(self, p):
        val = self.a / (self.b + np.sin(p/self.eps))
        return val

### 有限元
def FEM(N, pde, q=20, xs=0, xe=1):
    """
    

    Parameters
    ----------
    N : int
        网格剖分段数.
    pde : class
        PDE 模型.
    q : int, optional
        高斯求积阶数. The default is 20.
    xs : float, optional
        求解区域左端点. The default is 0.
    xe : float, optional
        求解区域右端点. The default is 1.

    Returns
    -------
    error : float
        L2 误差.

    """
    # 生成网格
    mesh = mesh_generator(N, xs, xe)
    # Gauss-Legendre 数值积分
    qf = mesh.integrator(q)
    bcs, ws = qf.get_quadrature_points_and_weights()
    # bcs.shape = (NQ, ldof)    ws.shape = (NQ, )
    
    # 每个单元內部的求积点
    quadpts = mesh.bc_to_point(bcs)     # (NQ, NC, 1)
    # 渗透率场中每个求积点的函数值
    coef = pde.k(quadpts)               # (NQ, NC, 1)
    # 单元测度
    cellmeasure = mesh.entity_measure() # (NC, )
    # 基函数梯度
    gphi = mesh.grad_lambda()           # (NC, ldof, 1)
    
    # 单元和网格节点的对应关系
    cell2dof = mesh.entity('cell')
    
    # 刚度矩阵
    A = np.einsum('lim, ljm, l, qlm, q -> lij',
                  gphi, gphi, cellmeasure, coef, ws)
    I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(N+1, N+1))
    
    # 右端项
    bb = pde.source(quadpts)   # (NQ, NC, 1)
    bb = np.einsum('qlm, q, qi, l -> li', bb, ws, bcs, cellmeasure) 
    F = np.zeros(N+1)
    np.add.at(F, cell2dof, bb)
    
    # 边界处理
    A[(0, N), :] = 0
    F[0] = F[N] = 0
    A[0, 0] = A[N, N] = 1
    
    # 数值解求解
    uh = spsolve(A, F).reshape(-1)
    
    ## 真解在积分点处的函数值
    val1 = pde.solution(quadpts).reshape(-1)      #(NQ*NC, 1)
    val1 = val1.reshape((N, q), order='F')
    ## 数值解在积分点处的函数值
    val2 = uh[cell2dof] # (NC, ldof)
    val2 = np.einsum('li, qi -> lq', val2, bcs)
    # 计算 L2 误差
    error = np.einsum('lq, lq, q, l -> ', val2-val1, 
                      val2-val1, ws, cellmeasure)
    # 画图
    plt.xlabel("x")
    plt.ylabel("y")
    node = mesh.node.reshape(-1)
    pic1 = plt.plot(node, uh, color='Red', label="FEM solution")
    u = pde.solution(node)
    pic2 = plt.plot(node, u, color='blue', label="Real solution")
    plt.legend(loc = 'upper right')
    plt.show()
    return error

def MsFEM(nc, nf, pde, q=20, xs=0, xe=1):
    """
    

    Parameters
    ----------
    nc : int
        粗网格剖分段数.
    nf : int
        细网格剖分段数.
    pde : class
        PDE 模型.
    q : int, optional
        Gauss-Legendre 积分阶数. The default is 20.
    xs : float, optional
        求解区域左端点. The default is 0.
    xe : float, optional
        求解区域右端点. The default is 1.

    Returns
    -------
    error : float
        L2 误差.

    """
    mesh_c = mesh_generator(nc, xs, xe)
    
    # Gauss-Legendre 数值积分
    qf = mesh_c.integrator(q)
    bcs, ws = qf.get_quadrature_points_and_weights()
    
    # 多尺度基函数
    Phi = np.zeros((nc, nf+1, 2))
    # 所有单元刚度矩阵
    A = np.zeros((nc, 2, 2))
    # 所有单元载荷向量
    bb = np.zeros((nc, 2))
    
    # 逐单元计算基函数
    for ith in range(nc):
        mesh_f = mesh_generator(nf, mesh_c.node[ith,0],
                                mesh_c.node[ith+1, 0])
        
        # 每个单元內部的求积点
        quadpts_f = mesh_f.bc_to_point(bcs)      # (q, nf, 1)
        # 渗透率场中每个求积点的函数值
        coef = pde.k(quadpts_f)                  # (q, nf, 1)
        # 单元测度
        cellmeasure_f = mesh_f.entity_measure()  # (nf, )
        # 基函数梯度
        gvarphi = mesh_f.grad_lambda()           # (nc, ldof, 1)
        
        # 细网格单元和细网格网格节点的对应关系
        cell2dof_f = mesh_f.entity('cell')
        # 细网格上刚度矩阵
        A_f = np.einsum('lim, ljm, l, qlm, q -> lij',
                      gvarphi, gvarphi, cellmeasure_f, coef, ws)
        I_f = np.broadcast_to(cell2dof_f[:, :, None], shape=A_f.shape)
        J_f = np.broadcast_to(cell2dof_f[:, None, :], shape=A_f.shape)
        A_f = csr_matrix((A_f.flat, (I_f.flat, J_f.flat)), 
                                      shape=(nf+1, nf+1))
        # 细网格上右端项
        F_f = np.zeros(nf+1)
        
        # 多尺度基函数边界处理
        A_ff = A_f.copy()
        A_ff[(0, nf), :] = 0
        F_f[0] = 1
        F_f[nf] = 0
        A_ff[0, 0] = A_ff[nf, nf] = 1
        
        # 数值解求解多尺度基函数
        phi_0 = spsolve(A_ff, F_f).reshape(-1)
        phi_1 = 1 - phi_0
        D = np.c_[phi_0, phi_1]     # (nf+1, 2)
        
        # 多尺度单元刚度矩阵 
        Phi[ith] = D
        A[ith] = D.T @ A_f @ D
        
        # 多尺度单元载荷向量
        bb_f = pde.source(quadpts_f)
        bb_f = np.einsum('qlm, q, qi, l -> li', bb_f, 
                          ws, bcs, cellmeasure_f) 
        F_f = np.zeros(nf+1)
        np.add.at(F_f, cell2dof_f, bb_f)
        bb[ith] = D.T @ F_f
    
    cell2dof = mesh_c.entity('cell')
    I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(nc+1, nc+1))
    F = np.zeros(nc+1)
    np.add.at(F, cell2dof, bb)
    
    # 边界处理
    A[(0, nc), :] = 0
    F[0] = F[nc] = 0
    A[0, 0] = A[nc, nc] = 1

    # 数值解求解
    uh = spsolve(A, F).reshape(-1)
    # 真解在积分点处的值
    quadpts = mesh_c.bc_to_point(bcs)             # (q, nc, 1)
    val1 = pde.solution(quadpts).reshape(-1)
    val1 = val1.reshape((nc, q), order='F')
    
    # Phi (nc, nf+1, 2)
    bcs = bcs[:, 0]
    xidx = (bcs // (1/nf)).astype(np.int_)        # (q, )
    xbcs = bcs * nf - xidx
    xbcs = np.r_[xbcs, 1-xbcs].reshape(2, q)      # (2, q)
    
    val2 = uh[cell2dof]                           # (nc, 2)
    val2 = np.einsum('kl, kijl -> kij', val2,
                     Phi[:, cell2dof_f[xidx], :]  # (nc, q, 2, 2)
                     )                            # (nc, q, 2)
    val2 = np.einsum('kij, ji -> ki', val2, xbcs)

    cellmeasure_c = mesh_c.entity_measure()
    error = np.einsum('ki, ki, i, k -> ', val2-val1, 
                      val2-val1, ws, cellmeasure_c)
    
    # 画图
    plt.xlabel("x")
    plt.ylabel("y")
    mesh = mesh_generator(nc*nf)
    node = mesh.node.reshape(-1)
    val2 = np.einsum('li, lji -> lj', uh[cell2dof], Phi)  # (nc, nf+1, 2)
    cell2dof = np.arange(nc).reshape(-1, 1) * nf
    cell2dof = cell2dof + np.arange(nf)
    uh = np.zeros(nc*nf+1)
    np.add.at(uh, cell2dof, val2[:,:-1])
    
    pic1 = plt.plot(node, uh, color='Red', label="MsFEM solution")
    u = pde.solution(node)
    pic2 = plt.plot(node, u, color='blue', label="Real solution")
    plt.legend(loc = 'upper right')
    plt.show()
    return error

# 调用 PDE 模型
pde = PDE(1, 1+1e-4, 1/(15*np.pi))

# # 真解精细画图
# N = 1e4
# x = np.linspace(0, 1, int(N))
# u = pde.solution(x)
# pic2 = plt.plot(x, u, color='blue', label="Real solution")
# plt.legend(loc = 'upper right')
# plt.show()


# 设置加密的开始与终止剖分段数
n1 = 3
n2 = 3

# 存储误差 & 误差阶的数组
errors_FEM = np.zeros(n2-n1+1)
errors_MsFEM = np.zeros(n2-n1+1)
ords_FEM = np.zeros(n2-n1)
ords_MsFEM = np.zeros(n2-n1)

# 逐次计算误差
for ith in range(n1, n2+1):
    N = 2 ** ith
    errors_FEM[ith-n1] = FEM(N, pde)
    errors_MsFEM[ith-n1] = MsFEM(N, 64, pde)
    
    
# 逐次计算误差收敛阶
for ith in range(n2-n1):
    ords_FEM[ith] = -np.log(errors_FEM[ith+1] /
                              errors_FEM[ith]) / np.log(2)
    ords_MsFEM[ith] = -np.log(errors_MsFEM[ith+1] / 
                              errors_MsFEM[ith]) / np.log(2)
    
print("errors_FEM: ", errors_FEM)
print("errors_MsFEM: ", errors_MsFEM)
print("ords_FEM: ", ords_FEM)
print("ords_MsFEM: ", ords_MsFEM)
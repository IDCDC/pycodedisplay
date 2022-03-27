import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.decorator import cartesian
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.mesh import MeshFactory as MF
from fealpy.quadrature import GaussLegendreQuadrature, QuadrangleQuadrature
from fealpy.boundarycondition import DirichletBC


class MultiscaleFiniteBasis:
    def __init__(self, subspace, subbox, fun_a, nxf=8, nyf=8, 
                 basisboundarytype='L',q=None):
        """
        

        Notes
        ----------
        SubProblem 这一个类用于存放子问题的一些方法。
        
        subspace : fealpy.functionspace
            子问题求解空间.
        subbox : tuple
            子问题求解区域.
        fun_a : function
            子问题变系数函数.
        nxf : int, optional
            子问题 x 方向上剖分段数. The default is 8.
        nyf : int, optional
            子问题 y 方向上剖分段数. The default is 8.
        q : int, optional
            Gauss-lengendre 积分阶数. The default is None.

        """
        
        self.subspace = subspace
        self.box = subbox
        self.a = fun_a
        self.q = q if q is not None else 4
        self.subx0 = subbox[0]                    # 子问题求解区域 x 方向起始坐标
        self.subx3 = subbox[1]                    # 子问题求解区域 x 方向起始坐标
        self.sublx = self.subx3 - self.subx0      # 子问题求解区域 x 方向区域长度
        self.suby0 = subbox[2]                    # 子问题求解区域 x 方向起始坐标
        self.suby3 = subbox[3]                    # 子问题求解区域 x 方向起始坐标
        self.subly = self.suby3 - self.suby0      # 子问题求解区域 x 方向区域长度
        self.bbt = basisboundarytype              # 子问题边界条件类型，'L' or 'O'.
    
    def __str__(self):
        return "SubProlem class is used to hold methods for subproblems!"

    @cartesian
    def subsource(self, p):
        return np.zeros_like(p[..., 0])
    
    @cartesian
    def Lsub0dirichlet(self, p):                   #第 0 个子问题线性边界
        x = p[..., 0]
        y = p[..., 1]
        val = (self.subx3-x)*(self.suby3-y)/(self.sublx*self.subly)
        return val
    
    @cartesian
    def Lsub1dirichlet(self, p):                   #第 1 个子问题线性边界
        x = p[..., 0]
        y = p[..., 1]
        val = (self.subx3-x)*(y-self.suby0)/(self.sublx*self.subly)
        return val
    
    @cartesian
    def Lsub2dirichlet(self, p):                   #第 2 个子问题线性边界
        x = p[..., 0]
        y = p[..., 1]
        val = (x-self.subx0)*(self.suby3-y)/(self.sublx*self.subly)
        return val
    
    @cartesian
    def Lsub3dirichlet(self, p):                   #第 2 个子问题线性边界        
        x = p[..., 0]
        y = p[..., 1]
        val = (x-self.subx0)*(y-self.suby0)/(self.sublx*self.subly)
        return val
    
    @cartesian
    def Osubdirichlet(self, p, oth, q=None):
        """
        振荡边界条件

        Parameters
        ----------
        p : array object,
            笛卡尔坐标下的边界点坐标.
        oth : int
            同一单元中不同子问题的编号.

        Returns
        -------
        val : array object
        """
        q=q if q is not None else self.q
        # 第 oth 个 MsFEM 顶点 x 坐标
        xvc = np.array([self.subx0, self.subx0, self.subx3, self.subx3])
        # 第 oth 个 MsFEM 顶点 y 坐标
        yvc = np.array([self.suby0, self.suby3, self.suby0, self.suby3])
        # 第 oth 个 MsFEM 相对顶点 x 坐标
        xau = np.array([self.subx3, self.subx3, self.subx0, self.subx0])
        # 第 oth 个 MsFEM 相对顶点 y 坐标
        yau = np.array([self.suby3, self.suby0, self.suby3, self.suby0])
        
        x = p[..., 0]
        y = p[..., 1]
        qf = GaussLegendreQuadrature(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        val = np.zeros_like(x)
        
        bidx = np.where(x==xvc[oth])                  # 平行于 y 轴的边界条件  
        nbd = y[bidx]
        di = np.where(nbd ==yvc[oth])
        cc = np.c_[yau[oth]*np.ones_like(nbd), nbd]
        cc = np.einsum('ik, jk->ij', cc, bcs)
        cc = np.dstack((xvc[oth]*np.ones_like(cc), cc))
        cc = 1/self.a(cc)
        cc = np.einsum('ij, j->i', cc, ws)
        cc = cc*np.abs(yau[oth]*np.ones_like(nbd)-nbd)
        cc = cc/cc[di]
        val[bidx] = cc
        
        bidx = np.where(y==yvc[oth])                   # 平行于 x 轴的边界条件
        nbd = x[bidx]
        di = np.where(nbd==xvc[oth])
        cc = np.c_[xau[oth]*np.ones_like(nbd), nbd]
        cc = np.einsum('ik, jk->ij', cc, bcs)
        cc = np.dstack((cc, yvc[oth]*np.ones_like(cc)))
        cc = 1/self.a(cc)
        cc = np.einsum('ij, j->i', cc, ws)
        cc = cc*np.abs(xau[oth]*np.ones_like(nbd)-nbd)
        cc = cc/cc[di]
        val[bidx] = cc
        return val
    
    @cartesian
    def Osub0dirichlet(self, p):                        #第 0 个子问题振荡边界
        return self.Osubdirichlet(p, 0)
    
    @cartesian
    def Osub1dirichlet(self, p):                        #第 1 个子问题振荡边界
        return self.Osubdirichlet(p, 1)
    
    @cartesian
    def Osub2dirichlet(self, p):                        #第 2 个子问题振荡边界
        return self.Osubdirichlet(p, 2)
    
    @cartesian
    def Osub3dirichlet(self, p):                        #第 3 个子问题振荡边界
        return self.Osubdirichlet(p, 3)
    
    def subboundarychoice(self, subA, subF, bct):
        """
        此函数为了在不改变刚度矩阵与载荷向量的值的情况下，
        应用同一区域不同子问题下边界条件从而解出 phi 在单元的值.
        
        Parameters
        ----------
        subAA : array object
            子问题刚度矩阵.
        subFF : array obeject
            子问题载荷向量.
        bct : TYPE
            边界条件编号.

        Returns
        -------
        subphi : array object.
            一个单元中的 phi.

        """
        subAA, subFF = subA.copy(), subF.copy()  # 为了不改变传入的两个数组
        if self.bbt[0] == 'L':
            if bct == 'L0':
                subbc = DirichletBC(self.subspace, self.Lsub0dirichlet) 
            elif bct == 'L1':
                subbc = DirichletBC(self.subspace, self.Lsub1dirichlet)
            elif bct == 'L2':
                subbc = DirichletBC(self.subspace, self.Lsub2dirichlet)
            elif bct == 'L3':
                subbc = DirichletBC(self.subspace, self.Lsub3dirichlet)
        else:
            if bct == 'O0':
                subbc = DirichletBC(self.subspace, self.Osub0dirichlet)
            if bct == 'O1':
                subbc = DirichletBC(self.subspace, self.Osub1dirichlet)
            if bct == 'O2':
                subbc = DirichletBC(self.subspace, self.Osub2dirichlet)
            if bct == 'O3':
                subbc = DirichletBC(self.subspace, self.Osub3dirichlet)
        subAA, subFF = subbc.apply(subAA, subFF)
        subphi = spsolve(subAA, subFF)
        return subphi         # subphi.shape == subFF.shape
        
        

class MultiscaleFiniteElementSpace:
    def __init__(self, func_a, box, nxc, nyc, nxf=8, nyf=8,
                 q=None, basisboundarytype='L', mesh=None):
        """
        

        Parameters
        ----------
        func_a : function
            变系数函数.
        box : array object or tuple
            原问题求解区域.
        nxc : int
            粗网格沿 x 方向剖分段数.
        nyc : TYPE
            粗网格沿 y 方向剖分段数.
        nxf : TYPE, optional
            细网格沿 x 方向剖分段数. The default is 8.
        nyf : TYPE, optional
            细网格沿 y 方向剖分段数. The default is 8.
        q : int, optional
            Gauss-lengendre 积分阶数. The default is None.
        basisboundarytype : str, optional
            子问题边界条件类型, 'L' 表示线性，'O' 表示振荡. The default is 'L'. 
        mesh : fealpy.mesh
            原问题剖分网格.
        """

        self.a = func_a                             # 原问题的变系数函数
        self.box = box                              # 原问题求解区域
        self.nxc = nxc                              # 原问题求解区域沿 x 方向的剖分段数
        self.nyc = nyc                              # 原问题求解区域沿 y 方向的剖分段数
        self.nxf = nxf                              # 子问题求解区域沿 x 方向的剖分段数
        self.nyf = nyf                              # 子问题求解区域沿 y 方向的剖分段数
        self.bbt = basisboundarytype                # 子问题边界条件类型
        self.lx = box[1] - box[0]                   # 原问题求解区域沿 x 方向的长度
        self.ly = box[3] - box[2]                   # 原问题求解区域沿 y 方向的长度
        self.ncc = nxc * nyc                        # 原问题剖分网格内单元个数
        self.ncf = nxf * nyf                        # 子问题剖分网格内单元个数
        self.nnc = (nxc+1) * (nyc+1)                # 原问题剖分网格内节点个数
        self.nnf = (nxf+1) * (nyf+1)                # 子问题剖分网格内节点个数
        self.mesh = mesh if mesh is not None else MF.boxmesh2d(   # 原问题剖分网格
                    box, nx=nxc, ny=nyc, meshtype='quad', p=1)
        self.cellmeasure = (box[3]-box[2])*(box[1]-box[0]) / (self.nxc*self.nyc) # 原问题单元测度
        
        self.q = q if q is not None else 4
        
    def __str__(self):
        return "Multiscale finite element space!"
    
    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.entity('cell')[index]
    
    def sub_cell_to_dof(self, index=np.s_[:]):
        """ 这个函数是为了拿到子网格的 Node to cell 数组
        """
        mesh_sub = MF.boxmesh2d(self.box, self.nxf, self.nyf, meshtype='quad', p=1)
        return mesh_sub.entity('cell')
    
    def is_boundary_dof(self):
        """ 返回网格的边界点，以此进行边界处理
        """
        ibd = np.zeros(2*(self.nxc+self.nyc), dtype=np.int_)
        ibd[:self.nyc+1] = np.arange(self.nyc+1)
        ibd[-(self.nyc+1):] = np.arange(self.nxc*(self.nyc+1), self.nnc)
        ibd[self.nyc+1:-(self.nyc+2):2] = (self.nyc+1) * np.arange(1, self.nxc)
        ibd[self.nyc+2:-(self.nyc+1):2] = ibd[self.nyc+1:-(self.nyc+2):2] + self.nyc
        return ibd
    
    def subbox_generator(self, ith): 
        """ 生成原问题第 ith 个子单元作为子问题求解区域的 box 形式。
        """
        subbox = np.zeros(4)
        subbox[0] =  (ith//self.nxc)*self.lx/self.nxc + self.box[0]
        subbox[1] =  subbox[0] + self.lx/self.nxc
        subbox[2] =  (ith%self.nyc)*self.ly/self.nyc + self.box[2]
        subbox[3] =  subbox[2] + self.ly/self.nyc
        return subbox
    
    def basis_stiffmatrix(self, returntype='PhiA', q=None):
        """
        考虑到基函数与其刚度矩阵的特殊性，特将其起计算结合到一起

        Parameters
        ----------
        bs : str, optional
            决定函数的返回形式. The default is 'YY'.
        q : int, optional
            Gauss-lengendre 积分阶数. The default is None.
            
        Returns
        -------
        array object or tuple
            returntype == 'Phi', return Phi
            returntype == 'A', return A
            returntype == 'PhiA', return Phi, A

        """

        q=q if q is not None else self.q
        
        if returntype == 'Phi':
            Phi = np.zeros((self.ncc, 4, self.nnf))
        elif returntype == 'A':
            A = np.zeros((self.ncc, 4, 4))
        elif returntype == 'PhiA':
            Phi, A = np.zeros((self.ncc, 4, self.nnf)), np.zeros((self.ncc, 4, 4))
        
        # 逐单元求解子问题
        for ith in range(self.ncc):
            box_sub = self.subbox_generator(ith)
            mesh_sub = MF.boxmesh2d(box_sub, self.nxf, self.nyf, 
                                    meshtype='quad', p=1)
            space_sub = ParametricLagrangeFiniteElementSpace(mesh_sub, p=1, q=q)
            subpro = MultiscaleFiniteBasis(space_sub, box_sub, self.a, 
                                           self.nxf, self.nyf, self.bbt,q=q)
            A_sub = space_sub.stiff_matrix(c=self.a, q=q)
            F_sub = space_sub.source_vector(f=subpro.subsource, q=q)
            phi_cell = np.zeros((4, self.nnf))
            
            if self.bbt == 'L':             # 子问题边界条件为线性边界的求解
                phi_cell[0] = subpro.subboundarychoice(A_sub, F_sub, 'L0')
                phi_cell[1] = subpro.subboundarychoice(A_sub, F_sub, 'L1')
                phi_cell[2] = subpro.subboundarychoice(A_sub, F_sub, 'L2')
                phi_cell[3] = subpro.subboundarychoice(A_sub, F_sub, 'L3')
            else:                           # 子问题边界条件为振荡边界的求解
                phi_cell[0] = subpro.subboundarychoice(A_sub, F_sub, 'O0')
                phi_cell[1] = subpro.subboundarychoice(A_sub, F_sub, 'O1')
                phi_cell[2] = subpro.subboundarychoice(A_sub, F_sub, 'O2')
                phi_cell[3] = subpro.subboundarychoice(A_sub, F_sub, 'O3')
            
            if returntype == 'Phi':
                Phi[ith] = phi_cell
            elif returntype == 'A':
                A[ith] = phi_cell @ A_sub @ phi_cell.T
            elif returntype == 'PhiA':
                Phi[ith] = phi_cell
                A[ith] = phi_cell @ A_sub @ phi_cell.T
        
        if returntype == 'Phi':
            return Phi
        else:
            cell2dof = self.cell_to_dof()
            I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
            J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)
            A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(self.nnc, self.nnc))
            if returntype == 'A':
                return A
            elif returntype == 'PhiA':
                return Phi, A 
    
    def phi(self, q=None):
        """
        基函数的计算

        Returns
        -------
        array boject,      (self.ncc, 4, self.nnf)
            逐单元存储多尺度有限元基函数的数值表达形式.

        """
        q=q if q is not None else self.q
        return self.basis_stiffmatrix(returntype='Phi', q=q)
    
    def stiff_matrix(self, q=None):
        """    组装刚度矩阵， 返回矩阵 A 的 sparse.csr_matrix 格式
        """
        q=q if q is not None else self.q
        return self.basis_stiffmatrix(returntype='A', q=q)
        
    def mass_matrix(self, Phi=None, q=None):
        """    组装质量矩阵， 返回矩阵 M 的 sparse.csr_matrix 格式
        """
        Phi = Phi if Phi is not None else self.phi()
        q=q if q is not None else self.q
        M = np.zeros((self.ncc, 4, 4))
        
        # 逐单元计算单元质量矩阵
        for ith in range(self.ncc):
            box_sub = self.subbox_generator(ith)
            mesh_sub = MF.boxmesh2d(box_sub, self.nxf, self.nyf, meshtype='quad', p=1)
            space_sub = ParametricLagrangeFiniteElementSpace(mesh_sub, p=1, q=q)
            M_sub = space_sub.mass_matrix(c=self.a, q=q)
            M[ith] = Phi[ith] @ M_sub @ Phi[ith].T
        cell2dof = self.cell_to_dof()
        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(self.nnc, self.nnc))
        return M
    
    def source_vector(self, f, Phi=None, q=None):
        """     组装载荷向量， f 为右端项函数
        """
        Phi = Phi if Phi is not None else self.phi()
        q=q if q is not None else self.q
        bb = np.zeros((self.ncc, 4))
        
        # 逐单元计算单元载荷向量
        for ith in range(self.ncc):
            box_sub = self.subbox_generator(ith)
            mesh_sub = MF.boxmesh2d(box_sub, self.nxf, self.nyf, meshtype='quad', p=1)
            space_sub = ParametricLagrangeFiniteElementSpace(mesh_sub, p=1, q=q)
            F_sub = space_sub.source_vector(f, q=q)
            bb[ith] = Phi[ith] @ F_sub
        cell2dof = self.cell_to_dof()
        F = np.zeros(self.nnc)
        np.add.at(F, cell2dof, bb)
        return F
    
    def L2_error_u(self, u, uh, Phi=None, q=None):
        q=q if q is not None else self.q
        qf = QuadrangleQuadrature(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        # 真解在各积分点的数值
        val2 = self.mesh.bc_to_point(bcs)
        val2 = np.concatenate(val2, axis=1).reshape(self.ncc, q*q, 2)
        val2 = u(val2)
        val2 = val2.reshape(self.ncc, q, q)
        
        # 数值解在各积分点的数值
        bc = bcs[0][:,1]
        xidx = (bc // (1/self.nxf)).astype(np.int_)
        xbcs = (bc - xidx/self.nxf) * self.nxf
        yidx = (bc // (1/self.nyf)).astype(np.int_)
        ybcs = (bc - yidx/self.nyf) * self.nyf
        idx = self.nxf * xidx.reshape(-1,1) + yidx
        basisbcs = np.zeros((4, q, q))
        basisbcs[0] = np.einsum('i,j -> ij', 1-xbcs, 1-ybcs)
        basisbcs[1] = np.einsum('i,j -> ij', 1-xbcs, ybcs)
        basisbcs[2] = np.einsum('i,j -> ij', xbcs, 1-ybcs)
        basisbcs[3] = np.einsum('i,j -> ij', xbcs, ybcs)
        cell2dof = self.cell_to_dof()
        Phi = Phi if Phi is not None else self.phi()
        cell2dof_sub = self.sub_cell_to_dof()
        Phi = Phi[:,:,cell2dof_sub]
        val1 = np.einsum('klij, kl -> kij', Phi, uh[cell2dof])
        val1 = val1[:, idx, :]
        val1 = np.einsum('kijl, lij -> kij', val1, basisbcs)
        error = self.cellmeasure*np.einsum('kij, kij, ij ->', val2-val1, val2-val1, ws)
        return error**(0.5)
    
    def interpolation_zoom(self, xratio, yratio, bcs=None, q=None):
        """
        计算粗网格单元内部积分点在细网格单元内的重心坐标

        Parameters
        ----------
        xratio : int
            两套网格在 x 方向上的比率.
        yratio : int
            两套网格在 x 方向上的比率.
        bcs : array, optional
            一维 Gauss-Legendre 积分的中心坐标形式. The default is None.
        q : TYPE, optional
            Gauss-Legendre 积分阶数. The default is None.

        Returns
        -------
        idx : array object          shape: (q, q)
            粗网格积分点在细网格内单元编号.
        basisbcs : array object,    shape: (4, q, q)
            积分点在各单元内的重心坐标.

        """
        if bcs is None:
            q=q if q is not None else self.q
            qf = GaussLegendreQuadrature(q)
            bcs, ws = qf.get_quadrature_points_and_weights()
            bcs = bcs[:, 1]
        xidx = (bcs // (1/xratio)).astype(np.int_)
        xbcs = (bcs - xidx/xratio) * xratio
        yidx = (bcs // (1/yratio)).astype(np.int_)
        ybcs = (bcs - yidx/yratio) * yratio
        idx = xratio*xidx.reshape(-1,1) + yidx
        basisbcs = np.zeros((4, q, q))
        basisbcs[0] = np.einsum('i,j -> ij', 1-xbcs, 1-ybcs)
        basisbcs[1] = np.einsum('i,j -> ij', 1-xbcs, ybcs)
        basisbcs[2] = np.einsum('i,j -> ij', xbcs, 1-ybcs)
        basisbcs[3] = np.einsum('i,j -> ij', xbcs, ybcs)
        return idx, basisbcs
    
    def L2_error_FEM_u_reference(self, u_FEM, uh, nx, ny, Phi=None, q=None):
        if nx > self.nxc and ny > self.nyc:   # 判断参考解是否比数值解更精细
            q=q if q is not None else self.q
            qf = QuadrangleQuadrature(q)
            bcs, ws = qf.get_quadrature_points_and_weights()
            bcs = bcs[0][:,1]
            xr, yr = int(nx/self.nxc), int(ny/self.nyc)
            
            ###    参考解在高斯积分点处的值       
            idx, basisbcs = self.interpolation_zoom(xr, yr, q=q)
            mesh_u = MF.boxmesh2d(self.box, nx, ny, meshtype='quad', p=1)
            cell2dof = mesh_u.entity('cell')
            val2 = u_FEM[cell2dof].reshape(nx, self.nyc, yr, 4)
            val2 = np.concatenate(val2, axis=1).reshape(self.nxc, self.nyc, xr*yr, 4)
            val2 = np.concatenate(val2, axis=1).reshape(self.ncc, xr*yr, 4)
            val2 = val2[:, idx, :]
            val2 = np.einsum('kijl, lij -> kij', val2, basisbcs)
            
            ###    数值解在高斯积分点处的值
            idx, basisbcs = self.interpolation_zoom(self.nxf, self.nyf, q=q)
            cell2dof = self.cell_to_dof()
            Phi = Phi if Phi is not None else self.phi()
            cell2dof_sub = self.sub_cell_to_dof()
            Phi = Phi[:,:,cell2dof_sub]
            val1 = np.einsum('klij, kl -> kij', Phi, uh[cell2dof])
            val1 = val1[:, idx, :]
            val1 = np.einsum('kijl, lij -> kij', val1, basisbcs)
            error = self.cellmeasure*np.einsum('kij, kij, ij ->', val2-val1, val2-val1, ws)
            return error**(0.5)
        else:
            raise ValueError("Maybe you need to think about a more elaborate solution!")
        

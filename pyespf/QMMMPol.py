import numpy as np
from scipy.linalg import sqrtm, inv, solve
from scipy.optimize import fsolve
from copy import deepcopy, copy
from pyscf import gto, scf, ao2mo, dft, lib
from pyscf.data import radii

from timeit import default_timer as timer


"""
This class and helper functions are designed to facilitate QM/MM-pol calculations using multipole expansions of the QM 
charge density.

Classes:
QMMMPol - general helper class that stores minimal informaiton about the QM and MM systems, such as the fixed charges 
and dipoles and polarizabilities of the MM atoms, as well as QM and MM atom positions.
"""

def solveSCFSimple(A,b,tol,x0):
    """
    A simple SCF solver for the equation:
    x = A x + b
    """
    
    N, N_b = b.shape
    x = np.zeros(b.shape)
    
    for r in range(0,N_b):
        x_r = b[:,r]
        x_r = x0[:,r]
        x_0 = x_r + 0
        conv_meas = 2 * tol
        while conv_meas > tol:
            x_r = b[:,r] + A.dot(x_r)
            conv_meas = np.linalg.norm(x_r - x_0) / (N)
            x_0 = x_r + 0 
        x[:,r] = x_r + 0
        
    
    return x

def calculateFieldAndPot(X,q,x_0,R0=[],n=6.0):
    """
    Calculates the electric field and potential at X due to a point charge q at x_0
    :param X: Test points 3 x N
    :param q: charge 
    :param x_0: location of charge 3 x 1
    :return F: fields at each test point 3 x N
    :return Phi: potential at N points
    """
    # calculate the undamped potentials and fields
    if len(R0)==0:
        N = X.shape[1]
        Delta_x = X - x_0 
        r_sq = np.sum(Delta_x * Delta_x,axis=0)
        r = np.sqrt(r_sq)
        Phi = q / r 
        F = Delta_x*(q / (r_sq*r))

        return F, Phi
    
    # calcualte the damped potential and fields with rational damping 1/r -> 1/(r^n+R0^n)^(1/n)
    else:
        
        N = X.shape[1]
        Delta_x = X - x_0 
        r_sq = np.sum(Delta_x * Delta_x,axis=0)
        r = np.sqrt(r_sq)
        r_eff = (r**n + R0**n)**(1.0/n)
        Phi = q / r_eff 
        
        F = q * (Delta_x/r) * (1.0/(r_eff*r_eff)) * ((r/r_eff)**(n-1))

        return F, Phi
        

def calculateDipoleFieldAndPot(X,mu,x_0,R0=[],n=6.0):
    """
    Calculates the electric field and potential at X due to a point dipole mu at x_0
    :param X: Test points 3 x N
    :param mu: dipole
    :param x_0: location of charge 3 x 1
    :return F: fields at each test point 3 x N
    :return Phi: potential at N points
    """
    # calculate the undamped potentials and fields
    if len(R0)==0:
        N = X.shape[1]
        Delta_x = X - x_0 
        r_sq = np.sum(Delta_x * Delta_x,axis=0)
        r = np.sqrt(r_sq)
        #Delta_x_norm = Delta_x/r.reshape((1,N))
        #if self.print_info : print( np.sum(Delta_x_norm*mu.reshape((3,1)),axis=0) )
        Phi = 1.*np.sum( Delta_x*mu.reshape((3,1)),axis=0) / (r.reshape((1,N))**3) 
        #Phi = np.zeros((N,))
        ##if self.print_info : print(mu,r,Delta_x)
        #for k in range(0,N):
        #    Phi[k] = Delta_x[:,k].dot(mu)/(r[k]*r[k]*r[k])

        F = (3.*(mu.dot(Delta_x)*Delta_x))/(r_sq*r_sq*r) - (mu.reshape((3,1)))/ (r_sq*r)
    
        return F, Phi
    else:
        N = X.shape[1]
        Delta_x = X - x_0 
        r_sq = np.sum(Delta_x * Delta_x,axis=0)
        r = np.sqrt(r_sq)
        Delta_x_norm = Delta_x / r 
        r_eff = (r**n + R0**n)**(1.0/n)
        Phi = np.sum( Delta_x_norm*mu.reshape((3,1)),axis=0) * (1.0/ (r_eff*r_eff) ) * ((r/r_eff)**(n-1))

        F = -(1.0/ (r_eff*r_eff) ) * ((r/r_eff)**(n-1)) * (1.0/r) * ( mu.reshape((3,1)) -  (mu.dot(Delta_x_norm)*Delta_x_norm)  )
    
        F = F + (np.sum( Delta_x_norm*mu.reshape((3,1)),axis=0)).reshape((1,N)) * Delta_x_norm * ((n+1)*(r**(2*n-2))/(r_eff**(2*n+1)) - (n-1) * (r**(n-2)) / (r_eff**(n+1)) )
    
        return F, Phi

def getPySCFAOInfo(mol,get_overlaps=True,get_ao_atom_inds=True,get_nuc_charges=True,get_dip_ints=True):
    """
    Gets AO overlap intergals and AO atom inds from a PySCF GTO molecule object.
    :param mol: PySCF gto.M type object
    :return info: Dictionary of necessary info.
    """
    
    info = {} 
    N_AO = mol.nao
    info["N_AO"] = N_AO
    
    if get_overlaps:
        info["overlaps"] = mol.intor('cint1e_ovlp_sph')
        
    if get_ao_atom_inds:
        full_ao_info = mol.ao_labels(fmt=False)
        info["ao_atom_inds"] = [full_ao_info[n][0] for n in range(0,N_AO)] 
    
    if get_nuc_charges:
        info["nuc_charges"] = mol.atom_charges() 
    
    if get_dip_ints:
        info["dip_ints"] = mol.intor('cint1e_r_sph',comp=3)
        
    return info



class QMMMPol:
    """
    A general helper class for implementing the Direct Reaction Field method.
    This handles all of the solutions of the MM induction equations.
    
    The class contains basic information about the positions of MM and QM atoms, 
    and the electrostatics and polarizabilities of the MM atoms. 
    
    Currently only fixed charges for the MM atoms are implemented, but static dipoles
    would also be straightforward to implement.
    """
    def __init__(self):
        # variables for storing improtant info like coordinates and multipoles of MM atoms
        self.x_MM = np.empty(shape=(3,0))
        self.x_QM = np.empty(shape=(3,0))
        self.q_MM = np.empty(shape=(0,))
        self.mu_MM = np.empty(shape=(3,0))
        self.quad_MM = np.empty(shape=(3,3,0))
        self.alpha_MM = np.empty(shape=(0,))
        self.eta_MM = np.empty(shape=(0,))
        self.chi_MM = np.empty(shape=(0,))
        self.fixed_Q_groups = []


        # options for the QM/MM
        self.charge_op_method = "esp" # default charge operator
        self.dipole_op_method = None # default dipole oeprator method
        self.esp_grid = (8,38) # number of radial and angular grid points for ESP operator evaluation
        self.l_max_Q = 0
        self.grid_method = "lebedev"
        self.box_pad = 16.0
        self.N_cart = 10
        self.grid_block_size = 10 
        
        # stored charge operators
        self.Q = None
        self.add_rep1e = False 
        self.add_exrep = False
        self.single_gto_exrep = False
        self.damp_elec = False
        self.no_charge = False
        
        self.scal_perm = []
        self.scal_indperm = []
        self.thole_a = None
        self.exrep_cut = 10.0
        self.exrep_scal = 2.0
        self.use_exact_qmee = False
        self.drf_mode = "dipole"
        self.n_damp = 6.0 
        self.use_mf_pol = False
        self.scal_N_val = 1.0
        self.ex_rep_bas = "STO-3G"
        
        self.ind_mode = "linear solver"
        self.ind_tol = 1e-6
        
        self.print_info = False
        return
    
    def setupMM(self,x_MM_in,q_MM_in,alpha_MM_in):
        self.x_MM = deepcopy(x_MM_in)
        self.q_MM = deepcopy(q_MM_in)
        self.alpha_MM = deepcopy(alpha_MM_in)
        return
    
    def setupMMFQ(self,x_MM_in,q_MM_in,eta_MM_in,chi_MM_in,fixed_Q_groups_in):
        self.x_MM = deepcopy(x_MM_in)
        self.q_MM = deepcopy(q_MM_in)
        self.chi_MM = deepcopy(chi_MM_in)
        self.eta_MM = deepcopy(eta_MM_in)
        self.fixed_Q_groups = deepcopy(fixed_Q_groups_in)
        return
    
    def setupQM(self,x_QM_in):
        self.x_QM = deepcopy(x_QM_in)
        return
    
    def setupQMMMPol(self,x_MM_in,q_MM_in,alpha_MM_in,x_QM_in):
        self.drf_mode = "dipole"
        self.setupMM(x_MM_in,q_MM_in,alpha_MM_in)
        self.setupQM(x_QM_in)
        if len(self.scal_perm)==0:
            N_MM = len(q_MM_in)
            self.scal_perm = np.ones((N_MM,N_MM))
        if len(self.scal_indperm)==0:
            N_MM = len(q_MM_in)
            self.scal_indperm = np.ones((N_MM,N_MM))
        return
    
    def setupQMMMPolFQ(self,x_MM_in,q_MM_in,eta_MM_in,chi_MM_in,fixed_Q_groups_in,x_QM_in):
        self.drf_mode = "FQ"
        self.setupMMFQ(x_MM_in,q_MM_in,eta_MM_in,chi_MM_in,fixed_Q_groups_in)
        self.setupQM(x_QM_in)
        
        return
    
    def calculateFixedChargeFieldsAndPot(self,X):
        """ 
        Calculates external fields and potential from fixed charges at given points.
        :param X: 3 x N array of points to evaluate fields at.
        :return F_ext: 3 x N array external fields at points 
        :return Phi_ext: N array of potential at points 
        """
        
        N = X.shape[1] 
        F_ext = np.zeros((3,N))
        Phi_ext = np.zeros((N,))
        
        # loop over points and calculate fields and potentials
        for p in range(0,N):
            X_p = X[:,p].reshape((3,1))
            Delta_x = X_p - self.x_MM 
            r_sq = np.sum(Delta_x * Delta_x,axis=0)
            r = np.sqrt(r_sq)
            Phi_ext[p] = np.sum(self.q_MM / r)
            F_ext[:,p] = np.sum(((Delta_x*(self.q_MM / (r_sq*r)))),axis=1)
            
        
        return F_ext, Phi_ext 
    
    def calculatePolMatrix(self,use_vec=True):
        """
        Calculates the polarization matrix for the MM system.
        The induction equations are given by mu = (alpha^-1 - T)^-1 F_ext 
        :return T: the polarization matrix
        :return L:  alpha^-1 - T
        """
        # Use the Numpy vectorised version
        if use_vec:
            return self.calculatePolMatrixVec()
        # Use the original for loop version
        else:
            
            # number of dipoles
            N_P = self.alpha_MM.shape[0]
            # empty T matrix
            T = np.zeros((3*N_P,3*N_P))


            # T_pq mu_q is electric field due to dipole q at p 
            # T_pq = (3 (e_pq e_pq^T) - 1) / r_pq^3, with e_pq = (x_p - x_q) / r_pq
            for p in range(0,N_P):
                x_p = self.x_MM[:,p].reshape((3,1))
                for q in range(p+1,N_P): 
                    x_q = self.x_MM[:,q].reshape((3,1))
                    x = x_p-x_q 
                    r = np.linalg.norm(x)
                    if self.thole_a == None:
                        f_E = 1.0
                        f_T = 1.0 
                    else:
                        s_pq = self.thole_a * r / ((self.alpha_MM[p] * self.alpha_MM[q])**(1.0/6.0))
                        f_E = 1.0 - (1+s_pq + 0.5*s_pq *s_pq)*np.exp(-s_pq)
                        f_T = f_E - ((s_pq**3)/6.0) *np.exp(-s_pq)

                    T_pq = (f_T*3. * np.outer(x,x)/(r**5) - f_E*np.eye(3) / (r**3))
                    T[(3*p):(3*p+3),(3*q):(3*q+3)] = T_pq 
                    T[(3*q):(3*q+3),(3*p):(3*p+3)] = T_pq.T

            # L = alpha^-1 - T 
            L = np.diag( (1./self.alpha_MM).repeat(3) ) - T

            return T, L
    
    def calculatePolMatrixVec(self):
        """
        Calculates the polarization matrix for the MM system.
        The induction equations are given by mu = (alpha^-1 - T)^-1 F_ext 
        :return T: the polarization matrix
        :return L:  alpha^-1 - T
        """
        
        # number of dipoles
        N_P = self.alpha_MM.shape[0]
        # empty T matrix
        T = np.zeros((3*N_P,3*N_P))
        
        # distances
        dx = np.zeros((N_P,N_P,3))
        dx[:,:,0] = self.x_MM[0,:].reshape((N_P,1)) - self.x_MM[0,:].reshape((1,N_P))
        dx[:,:,1] = self.x_MM[1,:].reshape((N_P,1)) - self.x_MM[1,:].reshape((1,N_P))
        dx[:,:,2] = self.x_MM[2,:].reshape((N_P,1)) - self.x_MM[2,:].reshape((1,N_P))
        
        r = np.linalg.norm(dx,axis=2)
        np.einsum('ii->i',r)[:]=1.0
        if self.thole_a == None:
            f_E = np.ones((N_P,N_P))
            f_T = np.ones((N_P,N_P))
        else:
            s = self.thole_a * r / (np.outer(self.alpha_MM , self.alpha_MM)**(1.0/6.0))
            f_E = 1.0 - (1+s + 0.5*s *s)*np.exp(-s)
            f_T = f_E - ((s*s*s)/6.0) *np.exp(-s)
        
        dx_r = dx/(r.reshape((N_P,N_P,1)))
        r3inv = 1.0/(r*r*r)
        np.einsum('ii->i',r3inv)[:]=0.0
        #dx_r4 = dx_r/((r*r*r).reshape((N_P,N_P,1)))
        dx_r4 = dx_r * (r3inv.reshape((N_P,N_P,1)))
        T = (np.einsum('pqa,pqb->paqb',3.0*f_T.reshape((N_P,N_P,1))*dx_r,dx_r4).reshape((3*N_P,N_P,3))).reshape((3*N_P,3*N_P))
        T = T - np.kron(f_E * r3inv , np.eye(3))
        
        L = -T + 0
        np.einsum('ii->i',L)[:]=( (1./self.alpha_MM).repeat(3) )
        
        return T, L
    
    def calculateMMFixedChargeFieldsAndPot(self):
        """
        Calculates the fixed charge electric fields and potentials at each MM atom
        This excludes the self-interaction terms
        :return F_MM: 3 x N array external fields at points 
        :return Phi: N array of potential at points 
        """
        
        N = self.x_MM.shape[1] 
        F_MM = np.zeros((3,N))
        Phi_MM = np.zeros((N,))
        
        # loop over points and calculate fields and potentials
        for p in range(0,N):
            X_p = self.x_MM[:,p].reshape((3,1))

            inds = [n for n in range(0,N) if not n==p]
            Delta_x = X_p - self.x_MM[:,inds]
            r_sq = np.sum(Delta_x * Delta_x,axis=0)
            r = np.sqrt(r_sq)
            Phi_MM[p] = np.sum(self.q_MM[inds] / r)
            F_MM[:,p] = np.sum(((Delta_x*(self.q_MM[inds] / (r_sq*r)))),axis=1)
            #if self.scal_perm == None:
            #    Phi_MM[p] = np.sum(self.q_MM[inds] / r)
            #else:
            Phi_MM[p] = np.sum(self.scal_perm[p,inds]*self.q_MM[inds] / r)
            
            #if self.scal_indperm == None:
            #    F_MM[:,p] = np.sum(((Delta_x*(self.q_MM[inds] / (r_sq*r)))),axis=1)
            #else:
            F_MM[:,p] =  np.sum(self.scal_indperm[p,inds] *((Delta_x*(self.q_MM[inds] / (r_sq*r)))),axis=1)
            
            
        
        return F_MM, Phi_MM
    
    def calculateMMEnergy(self):
        """
        Calculates the MM electrostatic energy
        :return E_tot: total energy
        :return E_fixed: energy of fixed charges
        :return E_pol: polarisation energy E_tot = E_fixed + E_pol
        """
        
        # get fixed charge fields and potentials
        F_MM , Phi_MM = self.calculateMMFixedChargeFieldsAndPot()
        
        # get fixed charge energy
        E_fixed = 0.5 * np.sum(self.q_MM * Phi_MM)
        
        # get the poalrization energy
        E_pol, mu_ind = self.calculatePolEnergy(F_MM)
        
        E_tot = E_fixed + E_pol 
        
        return E_tot, E_fixed, E_pol
    
    def calculatePolEnergy(self,F_ext_in):
        """
        Calculates the Polarization energy and dipoles for the system
        :param F_ext: 3N x 1 or 3 x N array of external fields at each dipole
        :return E_pol: Polarization energy
        :return mu_ind: 3N x 1 Induced dipoles
        """
        
        N_P = self.alpha_MM.shape[0]
        if F_ext_in.shape[0] == 3:
            F_ext = F_ext_in.T.reshape((3*N_P,1))
        else:
            F_ext = F_ext_in
        
        # get the polarization matrix
        T, L = self.calculatePolMatrix()
        
        # solve the induction equations
        mu_ind = np.linalg.solve(L,F_ext)
        
        # get the energy
        E_pol = -0.5 * np.sum(mu_ind * F_ext) 
        
        return E_pol, mu_ind
    
    
    def calculateDRFVariables(self):
        """
        Calculates the variables required for evaluation of the DRF hamiltonian
        :return U_MM: F_MM.L^-1.F_MM
        :return U_QMMM: U_QMMM_A = F_MM.L^-1.f_QM_A
        :return U_QM: U_QMMM_AB = f_QM_A.L^-1.f_QM_B
        :return Phi_QM: electrostatic potentials due to MM atoms at each QM atom
        """
        
        
        N_QM = self.x_QM.shape[1] 
        N_MM = self.x_MM.shape[1]
        if self.l_max_Q == 0:
            N_Q = N_QM
        elif self.l_max_Q == 1:
            N_Q = 4 * N_QM
    
        if not self.damp_elec:
            self.R_damp = np.empty((N_QM,0))
        
        f_QM = np.zeros((3*N_MM,N_Q))
        Phi_QM = np.zeros((N_Q,))
        Phi_QM = [0.0 for n in range(0,N_Q)]
        
        # get the field, f_A, and potential, Phi_A, due to a unit charge at A at each of the MM atoms
        start = timer()
        for A in range(0,N_QM):
            x_A = 1.0*self.x_QM[:,A].reshape((3,1))
            f_A,Phi_A = calculateFieldAndPot(self.x_MM,1.0, x_A,R0=self.R_damp[A,:],n=self.n_damp)
            f_QM[:,A] = 1.*f_A.T.reshape(3*N_MM)
            # Phi_A is a list of potentials at each MM site due to unit charge at A
            # sum_k q_MM_k * Phi_A_k produces potential at A due to MM charges
            Phi_QM[A] = np.sum(Phi_A*self.q_MM)
            #if self.print_info : print("f_QM:" , f_A)
        end = timer()
        if self.print_info : print("QM Charge field+potential time:",end-start,"s")
        
        
        start = timer()
        if self.l_max_Q == 1 :
            for alpha in range(0,3):
                for A in range(0,N_QM):
                    unit_mu = np.zeros((3,))
                    unit_mu[alpha] = 1.0
                    x_A = 1.0*(self.x_QM[:,A]).reshape((3,1))
                    f_A_dip,Phi_A_dip = calculateDipoleFieldAndPot(self.x_MM,unit_mu,x_A,R0=self.R_damp[A,:],n=self.n_damp)
                    f_QM[:,A+(alpha+1)*N_QM] = f_A_dip.T.reshape((3*N_MM,))
                    # Phi_A is a list of potentials at each MM site due to unit charge at A
                    # sum_k q_MM_k * Phi_A_k produces potential at A due to MM charges
                    Phi_QM[(A+(alpha+1)*N_QM)] = (np.sum( Phi_A_dip*self.q_MM ))
                    

        end = timer()
        if self.print_info : print("QM Dipole field+potential time:",end-start,"s")
                    
            
        
        #if self.print_info : print(f_QM)
        
        Phi_QM = np.array(Phi_QM)
        #if self.print_info : print(Phi_QM)
            
        # generate the dipole interaction matrix
        T_mat, L = self.calculatePolMatrix()
        
        # get fixed charge fields and potentials
        start = timer()
        F_MM , Phi_MM = self.calculateMMFixedChargeFieldsAndPot()
        F_MM = F_MM.T.reshape((3*N_MM,1))
        end = timer()
        if self.print_info : print("MM charge fields + pot time:",end-start,"s")
        
        #mu_QM = np.linalg.solve(L,f_QM,assume_a="sym") 
        #mu_MM = np.linalg.solve(L,F_MM,assume_a="sym") 
        start = timer()
        if self.ind_mode == "linear solver":
            mu_QM = solve(L,f_QM,assume_a="sym") 
            mu_MM = solve(L,F_MM,assume_a="sym") 
        elif self.ind_mode == "simple":
            #mu_QM = solveSCFSimple(np.eye(3*N_MM)-L,f_QM,self.ind_tol, (self.alpha_MM).repeat(3).reshape((3*N_MM,1)) * f_QM)
            #mu_MM = solveSCFSimple(np.eye(3*N_MM)-L,F_MM,self.ind_tol,(self.alpha_MM).repeat(3).reshape((3*N_MM,1)) * F_MM)
            alpha_T = (self.alpha_MM).repeat(3).reshape((3*N_MM,1)) * T_mat
            mu_QM = solveSCFSimple(alpha_T,(self.alpha_MM).repeat(3).reshape((3*N_MM,1)) *f_QM,self.ind_tol, (self.alpha_MM).repeat(3).reshape((3*N_MM,1)) * f_QM)
            mu_MM = solveSCFSimple(alpha_T,(self.alpha_MM).repeat(3).reshape((3*N_MM,1)) *F_MM,self.ind_tol,(self.alpha_MM).repeat(3).reshape((3*N_MM,1)) * F_MM)
        elif self.ind_mode == "full inv":
            print("Full inverse calculated.")
            L_inv = inv(L)
            mu_QM = L_inv.dot(f_QM)
            mu_MM = L_inv.dot(F_MM)
            
            
        end = timer()
        if self.print_info : print("Linear equation solver time: ",end-start,"s")
        #mu_QM = np.linalg.solve(L,f_QM) 
        #mu_MM = np.linalg.solve(L,F_MM) 
        #alpha_half = (np.sqrt(self.alpha_MM).repeat((3,))).reshape((3*N_MM,1))
        #L_pre = (alpha_half*L) *alpha_half.T
        ##L_pre = np.einsum('i,ij,j->ij',alpha_half.reshape((3*N_MM,)),L,alpha_half.reshape((3*N_MM,)))
        #mu_QM = solve(L_pre,alpha_half*f_QM,assume_a="sym") *alpha_half
        #mu_MM = solve(L_pre,alpha_half*F_MM,assume_a="sym") *alpha_half

        
        # DRF variables
        start = timer()
        U_MM = np.sum(mu_MM * F_MM)
        U_QMMM = (F_MM.T.dot(mu_QM)).reshape((N_Q,))
        U_QM = (f_QM.T).dot(mu_QM)
        U_QM = 0.5 * (U_QM + U_QM.T)
        end = timer()
        if self.print_info : print("Contraction time: ",end-start,"s")
        
        #if self.print_info : print("N_QM:",N_QM, "N_Q:",N_Q)
        #if self.print_info : print(U_MM)
        #if self.print_info : print(U_QMMM)
        #if self.print_info : print(U_QM)
        #if self.print_info : print(Phi_QM)
        
        return U_MM, U_QMMM, U_QM, Phi_QM
    
    def calculateFQVariables(self):
        """
        Calculates variables in the FQ equations.
        """
        N_MM = len(self.q_MM)
        
        
        # construct the constrained charge groups
        # q_full = U.q_free + Q 
        N_con = len(self.fixed_Q_groups)
        N_free = N_MM - N_con 
        P = np.zeros((N_MM,N_free))
        S = np.zeros((N_MM,N_MM))
        Q = np.zeros((N_MM,))
        n_start = 0 
        for g,group in enumerate(self.fixed_Q_groups):       
            atoms = group["atoms"]
            N_g = len(atoms)
            Q[atoms[-1]] = group["charge"]
            for k in range(0,N_g-1):
                P[atoms[k],n_start+k] = 1.0 
            n_start = n_start + (N_g -1)
            S[atoms[-1],atoms[0:(N_g-1)]] = -1.0
        U = P + S.dot(P)
        
        # construct the interaction tensor
        eta = 0.5*(self.eta_MM.reshape((1,N_MM)) + self.eta_MM.reshape((N_MM,1)))
        eta_d = np.array(self.eta_MM).reshape((N_MM,))
        J = np.zeros((N_MM,N_MM))
        # distances
        dx = np.zeros((N_MM,N_MM,3))
        dx[:,:,0] = self.x_MM[0,:].reshape((N_MM,1)) - self.x_MM[0,:].reshape((1,N_MM))
        dx[:,:,1] = self.x_MM[1,:].reshape((N_MM,1)) - self.x_MM[1,:].reshape((1,N_MM))
        dx[:,:,2] = self.x_MM[2,:].reshape((N_MM,1)) - self.x_MM[2,:].reshape((1,N_MM))
        r = np.linalg.norm(dx,axis=2)
        J = eta/np.sqrt((1+eta*eta*r*r))
        np.einsum('ii->i',J)[:]=eta_d
        
        # transformed interaction tensor : q_free = -J_free^-1 b_0
        J_free = (U.T).dot(J.dot(U))

        #print(J)
        
        # Fixed charge part
        F_MM, Phi_MM = self.calculateMMFixedChargeFieldsAndPot()
        Phi_MM = np.array(Phi_MM).reshape((N_MM,))
        chi = np.array(self.chi_MM).reshape((N_MM,))
        
        # b_0 : q_free = -J_free^-1 b_0
        b_0 = (U.T).dot( chi + Phi_MM  + J.dot(Q) ) 
        J_free_inv_b_0 = np.linalg.solve(J_free,b_0)
        
        # U_FQ0 - the FQ energy in the abscence of QM potential
        U_FQ0 = Q.dot( chi + Phi_MM ) + 0.5*(Q.dot(J.dot(Q))) -0.5 * b_0.dot(J_free_inv_b_0)
        
        # q_0 - fluctuating charges in response to fixed MM charges
        q_0 = Q - U.dot(J_free_inv_b_0)
        #print(q_0)
        
        return U_FQ0, q_0, U, J_free
    
    def calculateDRFFQVariables(self):
        """
        Calculates the variables required for evaluation of the DRF hamiltonian
        :return U_MM: F_MM.L^-1.F_MM
        :return U_QMMM: U_QMMM_A = F_MM.L^-1.f_QM_A
        :return U_QM: U_QMMM_AB = f_QM_A.L^-1.f_QM_B
        :return Phi_QM: electrostatic potentials due to MM atoms at each QM atom
        """
        
        
        N_QM = self.x_QM.shape[1] 
        N_MM = self.x_MM.shape[1]
        if self.l_max_Q == 0:
            N_Q = N_QM
        elif self.l_max_Q == 1:
            N_Q = 4 * N_QM
    
        if not self.damp_elec:
            self.R_damp = np.empty((N_QM,0))
        
        phi_QM = np.zeros((N_MM,N_Q))
        Phi_QM = np.zeros((N_Q,))
        
        # get the field, f_A, and potential, Phi_A, due to a unit charge at A at each of the MM atoms
        start = timer()
        for A in range(0,N_QM):
            x_A = 1.0*self.x_QM[:,A].reshape((3,1))
            f_A,Phi_A = calculateFieldAndPot(self.x_MM,1.0, x_A,R0=self.R_damp[A,:])
            phi_QM[:,A] = 1.*Phi_A.reshape((N_MM,))
            # Phi_A is a list of potentials at each MM site due to unit charge at A
            # sum_k q_MM_k * Phi_A_k produces potential at A due to MM charges
            Phi_QM[A] = np.sum(Phi_A*self.q_MM)
            #if self.print_info : print("f_QM:" , f_A)
        end = timer()
        if self.print_info : print("QM Charge field+potential time:",end-start,"s")
        
        
        start = timer()
        if self.l_max_Q == 1 :
            for alpha in range(0,3):
                for A in range(0,N_QM):
                    unit_mu = np.zeros((3,))
                    unit_mu[alpha] = 1.0
                    x_A = 1.0*(self.x_QM[:,A]).reshape((3,1))
                    f_A_dip,Phi_A_dip = calculateDipoleFieldAndPot(self.x_MM,unit_mu,x_A,R0=self.R_damp[A,:])
                    phi_QM[:,A+(alpha+1)*N_QM] = 1.*Phi_A_dip.reshape((N_MM,))
                    # Phi_A is a list of potentials at each MM site due to unit charge at A
                    # sum_k q_MM_k * Phi_A_k produces potential at A due to MM charges
                    Phi_QM[(A+(alpha+1)*N_QM)] = (np.sum( Phi_A_dip*self.q_MM ))
                    

        end = timer()
        if self.print_info : print("QM Dipole field+potential time:",end-start,"s")
                    
        #Phi_QM = np.array(Phi_QM)
        #if self.print_info : print(Phi_QM)
            
        # generate the dipole interaction matrix
        U_FQ0, q_0, U, J_free = self.calculateFQVariables()
        
        # calculate the delta_qs
        start = timer()
        dq_QM = U.dot(solve(J_free,(U.T).dot(phi_QM),assume_a="sym"))
        end = timer()
        if self.print_info : print("Linear equation solver time: ",end-start,"s")

        
        # DRF variables
        start = timer()
        U_MM = -2.0*U_FQ0
        U_QMMM = -(phi_QM.T).dot(q_0)
        U_QM = (phi_QM.T).dot(dq_QM)
        end = timer()
        if self.print_info : print("Contraction time: ",end-start,"s")
        
        #if self.print_info : print("N_QM:",N_QM, "N_Q:",N_Q)
        #if self.print_info : print(U_MM)
        #if self.print_info : print(U_QMMM)
        #if self.print_info : print(U_QM)
        #if self.print_info : print(Phi_QM)
        
        return U_MM, U_QMMM, U_QM, Phi_QM
    
    def generateMullikenChargeOperators(self,overlaps,ao_atom_inds,mol=None):
        """
        Generates the Lulliken type charge operator:
        Q_A,nm = S_nm (delta_n,A + delta_m,B)/2 
        or alternatively
        Q_A = (1/2)(P_A * S + S * P_A)
        :param overlaps: Array of AO overlaps
        :param ao_atom_inds: N_AO list or array of integers specifying which atom each AO belongs to.
        :return Q: A list [Q_0, Q_1, ...] of charge operators in the AO basis
        """
        
        N_QM = self.x_QM.shape[1] 
        N_AO = overlaps.shape[0]
        Q = [np.zeros((N_AO,N_AO))] * N_QM
        for A in range(0,N_QM):
            is_A = np.array([int(k == A) for k in ao_atom_inds])
            P_A_mull = is_A.reshape((N_AO,1)) + is_A.reshape((1,N_AO))
            Q[A] = -0.5 * P_A_mull * overlaps
            
        if self.l_max_Q > 0 :
            r_ints = mol.intor('cint1e_r_sph',comp=3)
            mu = self.generateMullikenDipoleOperators(r_ints,Q,ao_atom_inds)
            for alpha in range(0,3):
                for A in range(0,N_QM):
                    Q.append(mu[A][alpha,:,:])
            
        return Q
    
    def generateLowdinChargeOperators(self,overlaps,ao_atom_inds): 
        """
        Generates the Lowdin type charge operator:
        Q_A = S^(1/2) P_A S^(1/2)
        :param overlaps: Array of AO overlaps
        :param ao_atom_inds: N_AO list or array of integers specifying which atom each AO belongs to.
        :return Q: A list [Q_0, Q_1, ...] of charge operators in the AO basis
        """
        
        N_QM = self.x_QM.shape[1] 
        N_AO = overlaps.shape[0]
        Q = [np.zeros((N_AO,N_AO))] * N_QM
        # calculate S^1/2
        S_1_2 = sqrtm(overlaps)
        for A in range(0,N_QM):
            is_A = np.array([(k == A) for k in ao_atom_inds])
            # equivalent to (S^1/2*P_A)*S_1_2 but avoiding extra matrix formation and multiplication
            Q[A] = -(S_1_2* is_A).dot(S_1_2)
        
        
        
        return Q
    
    def generateChargeOperators(self,method="mulliken",overlaps=None,ao_atom_inds=None,mol=None):
        """
        Generates a set of charge operators as a list [Q_0, Q_1, ...] in the AO basis using the designated method.
        :param method: str defining the method: mulliken, lowdin
        :param overlaps: N_AO x N_AO numpy array of AO overlaps
        :param ao_atom_inds: N_AO list or array of integers specifying which atom each AO belongs to.
        :return Q: a list [Q_0, Q_1, ...] of charge operators in the AO basis
        """
        
        if method=="mulliken":
            return self.generateMullikenChargeOperators(overlaps=overlaps,ao_atom_inds=ao_atom_inds,mol=mol)
        elif method=="lowdin":
            return self.generateLowdinChargeOperators(overlaps=overlaps,ao_atom_inds=ao_atom_inds)
        elif method=="esp":
            return self.generateESPChargeOperators(overlaps=overlaps,mol=mol)
        else:
            raise Exception("Charge operator method,", method," not recognised.")
            
    def generateMullikenDipoleOperators(self,r_ints,Q,ao_atom_inds):
        """
        Generates the Lulliken type dipole operator:
        mu_A = -(1/2)(P_A * mu_A + S * mu_A) - R_A Q_A
        :param overlaps: Array of AO overlaps
        :param ao_atom_inds: N_AO list or array of integers specifying which atom each AO belongs to.
        :return mu: A list [mu_0, mu_1, ...] of dipole operators in the AO basis
        """
        
        N_QM = self.x_QM.shape[1] 
        N_AO = r_ints.shape[1]
        mu = [np.zeros((3,N_AO,N_AO)) for n in range(0,N_QM)] 
        for A in range(0,N_QM):
            is_A = np.array([int(k == A) for k in ao_atom_inds])
            P_A_mull = is_A.reshape((N_AO,1)) + is_A.reshape((1,N_AO))
            for alpha in range(0,3):
                mu[A][alpha,:,:] = -0.5* P_A_mull * r_ints[alpha,:,:] - self.x_QM[alpha,A] * Q[A]
            
        
        return mu
    
    def generateLowdinDipoleOperators(self,dip_ints,Q,ao_atom_inds): 
        """
        Generates the Lowdin type dipole operator:
        Q_A = mu^(1/2) P_A mu^(1/2)
        :param overlaps: Array of AO overlaps
        :param ao_atom_inds: N_AO list or array of integers specifying which atom each AO belongs to.
        :return mu: A list [mu_0, mu_1, ...] of dipole operators in the AO basis
        """
        
        N_QM = self.x_QM.shape[1] 
        N_AO = rints.shape[1]
        mu = [np.zeros((3,N_AO,N_AO))] * N_QM
        # calculate r_alpha^1/2 matrices
        r_ints_1_2 = np.zeros((3,N_AO,N_AO)) 
        for alpha in range(0,3):
            r_ints_1_2 = sqrtm(r_ints[alpha,:,:])
            
        for A in range(0,N_QM):
            is_A = np.array([(k == A) for k in ao_atom_inds])
            # equivalent to (S^1/2*P_A)*S_1_2 but avoiding extra matrix formation and multiplication
            for alpha in range(0,3):
                mu[A][alpha,:,:] =-(r_ints_1_2[alpha,:,:]* is_A).dot(r_ints_1_2[alpha,:,:]) - x_QM[alpha,A] * Q[A]
        
        return mu
            
    def setupElectrostaticEmbeddingPySCF(self,mf):
        """
        This modifies an input pyscf scf object to add 
        """
        
        # first get the ao_info from the mf.mol object
        ao_info = getPySCFAOInfo(mf.mol)
        
        # get charge operators
        Q = self.generateChargeOperators(method=self.charge_op_method,overlaps=ao_info["overlaps"],ao_atom_inds=ao_info["ao_atom_inds"],mol=mf.mol)
        
        # generate QM/MM / DRF variables This is overkill for fixed charge embedding - needs modifying
        U_MM,U_QMMM, U_QM, Phi_QM = self.calculateDRFVariables()
        
        # construct the modified 1e interaction term
        N_e = mf.mol.nelectron
        N_QM = self.x_QM.shape[1]
        if not self.use_exact_qmee:
            V = np.sum(Phi_QM.reshape((N_QM,1,1)) * np.array(Q), axis=0) 
        else:
            N_AO = mf.mol.nao
            V = np.zeros((N_AO,N_AO))
            
            for A in range(0,N_MM):
                grid = (self.x_MM[:,A]).reshape((1,3))
                V_A = mf.mol.intor('cint1e_grids_sph',grids=grid)
                V = V + (-self.q_MM[A]) * V_A[0]
                
        V = V + (1.0/N_e)*ao_info["overlaps"] * np.sum(np.array(ao_info["nuc_charges"]) * Phi_QM[0:N_QM])
        
        # generate the new QM hamiltonian with just fixed charge interactions
        h_1e =  (mf.mol.intor('int1e_kin') + mf.mol.intor('int1e_nuc') + V) 
        
        # modify the mf object 1e hamiltonian routine
        mf.get_hcore = lambda *args: h_1e
        
        return mf
    
    def setupDRFPySCF(self,mf,return_U_MM=False):
        """
        This modifies an input pyscf scf object to add the DRF terms to the 1e hamiltonian and 2e integrals.
        :param mf: PySCF mf object to be modified
        :return mf: returns the same mf object
        """
        #print("passed to qmmmpol omega,alpha,hyb",mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin))
        is_hyb_dft = False
        is_lr_dft = False
        if mf.__class__.__name__ == "RHF":
            self.mf_drf = scf.RHF(mf.mol)
            self.mf_copy = scf.RHF(mf.mol)
            print("RHF DRF")
        elif mf.__class__.__name__ == "UHF":
            self.mf_drf = scf.UHF(mf.mol)
            self.mf_copy = scf.UHF(mf.mol)
            print("UHF DRF")
            
        elif mf.__class__.__name__ == "RKS" or mf.__class__.__name__ == "UKS":
            if mf.__class__.__name__ == "RKS":
                #self.mf_drf = mf.to_hf() 
                self.mf_drf = dft.RKS(mf.mol,xc="HF")
                #self.mf_drf = scf.RHF(mf.mol)
                self.mf_copy = dft.RKS(mf.mol,xc=mf.xc)
                print("RKS DRF")
            elif mf.__class__.__name__ == "UKS":
                self.mf_drf = dft.UKS(mf.mol,xc="HF")
                #self.mf_drf = scf.UHF(mf.mol)
                self.mf_copy = dft.UKS(mf.mol,xc=mf.xc)
                print("UKS DRF")
            
            
            
            #self.mf_drf.build()
            self.mf_drf.omega = 0.0
            if (mf._numint.hybrid_coeff(mf.xc))>np.finfo(float).eps:
                is_hyb_dft = True
            if (mf._numint.rsh_coeff(mf.xc)[0])>np.finfo(float).eps:
                is_lr_dft = True
                is_hyb_dft = True
            #if self.print_info : print(self.mf_drf._eri)
        
        if mf.__class__.__name__ == "DFRHF":
            aux_basis = mf.auxbasis
            self.mf_drf = scf.RHF(mf.mol).density_fit(auxbasis=aux_basis)
            self.mf_copy = scf.RHF(mf.mol).density_fit(auxbasis=aux_basis)
            print("DFRHF DRF")
        elif mf.__class__.__name__ == "DFUHF":
            aux_basis = mf.auxbasis
            self.mf_drf = scf.UHF(mf.mol).density_fit(auxbasis=aux_basis)
            self.mf_copy = scf.UHF(mf.mol).density_fit(auxbasis=aux_basis)
            print("DFUHF DRF")
            
        elif mf.__class__.__name__ == "DFRKS" or mf.__class__.__name__ == "DFUKS":
            if mf.__class__.__name__ == "DFRKS":
                #self.mf_drf = mf.to_hf() 
                aux_basis = mf.auxbasis
                self.mf_drf = dft.RKS(mf.mol).density_fit(auxbasis=aux_basis)
                self.mf_copy = dft.RKS(mf.mol,xc=mf.xc).density_fit(auxbasis=aux_basis)
                print("DFRKS DRF")
            elif mf.__class__.__name__ == "DFUKS":
                aux_basis = mf.auxbasis
                self.mf_drf = dft.UKS(mf.mol).density_fit(auxbasis=aux_basis)
                self.mf_copy = dft.UKS(mf.mol,xc=mf.xc).density_fit(auxbasis=aux_basis)
                print("DFUKS DRF")
            
            self.mf_drf.xc = 'HF*1.0' 
            
            #self.mf_drf.build()
            self.mf_drf.omega = 0.0
            if (mf._numint.hybrid_coeff(mf.xc))>np.finfo(float).eps:
                is_hyb_dft = True
            if (mf._numint.rsh_coeff(mf.xc)[0])>np.finfo(float).eps:
                is_lr_dft = True
                is_hyb_dft = True
        #else:
        #    raise Exception("SCF type not recognised: "+mf.__class__.__name__)
            
            
        
        #self.mf_copy = mf.copy()
        #print("after copy mf omega,alpha,hyb",mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin))
        #print("copy of mf omega,alpha,hyb",self.mf_copy._numint.rsh_and_hybrid_coeff(self.mf_copy.xc, spin=self.mf_copy.mol.spin))
        
        # first get the ao_info from the mf.mol object
        ao_info = getPySCFAOInfo(mf.mol)
        N_AO = len(ao_info["ao_atom_inds"])
        
        # get charge operators
        if type(self.Q) == type(np.array([[]])):
            Q = self.Q 
        elif self.Q == None:
            start = timer()
            Q = self.generateChargeOperators(method=self.charge_op_method,overlaps=ao_info["overlaps"],ao_atom_inds=ao_info["ao_atom_inds"],mol=mf.mol)
            end = timer()
            if self.print_info : print("ESP operator set-up time:",end-start,"s")
            Q = np.array(Q)
            self.Q = Q 
        
        #if self.print_info : print(Q.shape)
        #if self.print_info : print(Q)
        
        # generate QM/MM / DRF variables This is overkill for fixed charge embedding - needs modifying
        start = timer()
        if self.drf_mode == "dipole":
            U_MM,U_QMMM, U_QM, Phi_QM = self.calculateDRFVariables()
        elif self.drf_mode == "FQ" :
            U_MM,U_QMMM, U_QM, Phi_QM = self.calculateDRFFQVariables()
        else:
            raise Exception("Need to be in either dipole or fluctuating charge mode")
        end = timer()
        if self.print_info : print("DRF variable set-up time:",end-start,"s")
        # construct the modified 1e hamiltonian for the static multipoles in the MM environment
        N_e = mf.mol.nelectron
        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]
        N_Q = Q.shape[0]
        Z_QM = np.array(ao_info["nuc_charges"])
        start = timer()
        #V = np.sum(Phi_QM.reshape((N_Q,1,1)) * (Q), axis=0) 
        if not self.use_exact_qmee:
            V = np.einsum('A,Aij->ij',Phi_QM,Q)
            V = V + ao_info["overlaps"] * ((1.0/float(N_e))*np.sum(Z_QM * Phi_QM[0:N_QM]))
        else:
            N_AO = mf.mol.nao
            V = np.zeros((N_AO,N_AO))
            
            for A in range(0,N_MM):
                grid = (self.x_MM[:,A]).reshape((1,3))
                V_A = mf.mol.intor('cint1e_grids_sph',grids=grid)
                V = V + (-self.q_MM[A]) * V_A[0]
            U_nuc = 0.0
            
            for A in range(0,N_MM):
                x_0 =  (self.x_MM[:,A]).reshape((3,1))
                f,phi = calculateFieldAndPot(self.x_QM,self.q_MM[A],x_0)
                U_nuc = U_nuc + np.einsum('A,A',phi,Z_QM)
            V = V + ao_info["overlaps"] * ((1.0/float(N_e))*U_nuc)
        
        end = timer()
        if self.print_info : print("Fixed charge ESP term time:",end-start,"s")
        #if self.print_info : print(V)
        
        #if self.print_info : print(V)
        
        # generate the new QM hamiltonian with just fixed charge interactions
        #h_1e =  (mf.mol.intor('cint1e_kin_sph') + mf.mol.intor('cint1e_nuc_sph') + V) 
        ##h_1e =  (mf.mol.intor('int1e_kin') + mf.mol.intor('int1e_nuc') + V) 
        ## modify the mf object 1e hamiltonian routine
        #mf.get_hcore = lambda *args: h_1e
        #return mf
    
        # add the DRF 1e correction terms
        # the "induction" term - interaction of electron with dipoles induced by static charges
        start = timer()
        U_ind = U_QMMM.reshape((N_Q)) + (U_QM[:,0:N_QM].dot(Z_QM)).reshape((N_Q))
        #if self.print_info : print(U_ind,U_QM.dot(Z_QM))
        #V = V -  np.sum(U_ind.reshape((N_Q,1,1)) * Q, axis=0) 
        V = V - np.einsum('A,Aij->ij',U_ind,Q)
        end = timer()
        if self.print_info : print("Induction term time:",end-start,"s")
        #U_QMMM = U_QMMM.reshape((N_QM,)) 
        #if self.print_info : print(U_QMMM.shape)
        #for A in range(0,N_QM):
        #    s = 0.
        #    for B in range(0,N_QM):
        #        s = s + U_QM[A,B] * Z_QM[B] 
        #    if self.print_info : print(U_QMMM[A],s)
        #    V = V - (U_QMMM[A] + s) * Q[A,:,:]
        
        # add the self term - interaction of electron with its own response in the polarizable env
        Sinv = inv(ao_info["overlaps"])
        start = timer()
        #for A in range(0,N_Q):
        #    V = V - 0.5* Q[A,:,:].dot(Sinv).dot( np.sum(U_QM[A,:].reshape((N_Q,1,1)) * Q , axis=0) )
        UQ = np.einsum('AB,Bij->Aij',-U_QM,Q)
        if not self.use_mf_pol:
            V = V +0.5 * np.einsum('Aij,Ajk->ik',Q.dot(Sinv),UQ)
        end = timer()
        if self.print_info : print("DRF self int term:",end-start,"s")
        #for A in range(0,N_QM):
        #    for B in range(0,N_QM):
        #        V = V - 0.5* Q[A,:,:].dot( U_QM[A,B] * Q[B,:,:]  )
        
        V_0_DRF = -0.5*U_MM - np.sum(Z_QM * U_QMMM[0:N_QM]) - 0.5 * Z_QM.dot(U_QM[0:N_QM,0:N_QM].dot(Z_QM))
        
        V = V + (1.0/N_e)*ao_info["overlaps"] * V_0_DRF
        
        self.UQ = UQ 
        
        # the electron repulsion integrals with 's1' symmetry are stored as a 4-index array with elements (ij|kl) in chemist notation
        # eri[i,j,k,l] = int phi_i*(x1) phi_j(x1) h2 phi_k*(x2) phi_l(x2)
        #eri_0 = self.mf_drf.mol.intor('cint2e_sph', aosym='s1')
        
        start = timer()

        #eri = np.einsum('Aij,Akl->ijkl',Q,UQ)
        end = timer()
        if self.print_info : print("Modified ERI set-up time:",end-start,"s")

        if self.add_rep1e:
            U_reps = self.calculateBuckRepulsion(pairwise=True)
            for A in range(0,N_QM):
                V = V + (-Q[A,:,:]) * (np.sum(U_reps[A,:])/(Z_QM[A]))
            if self.l_max_Q > 0:
                for alpha in range(0,3):
                    for A in range(0,N_QM):
                        V = V + (-Q[A+(alpha+1)*N_QM,:,:]) * (np.sum(U_reps[A+(alpha+1)*N_QM,:])/(Z_QM[A]))
        start = timer()
        if self.add_exrep:
            V = V + self.calculateHeffExRep(mf.mol)
        end = timer()
        if self.print_info : print("Exch-Rep H_eff construction time:", end-start,"s")
        
        start
        # generate the new QM hamiltonian with just fixed charge interactions
        h_1e =  (mf.mol.intor('cint1e_kin_sph') + mf.mol.intor('cint1e_nuc_sph') + V) 
        h_1e_0 = mf.mol.intor('cint1e_kin_sph') + mf.mol.intor('cint1e_nuc_sph')
        #h_1e =  (mf.mol.intor('int1e_kin') + mf.mol.intor('int1e_nuc') + V) 
        
        #def get_jk_new(*args):
        #    #if self.print_info : print([arg for arg in args])
        #    return (self.mf_copy.get_j(*args)+get_j_drf(args[1])-0.5*get_k_drf(args[1]),self.mf_copy.get_k(*args))
            #return (self.mf_copy.get_j(*args)+self.mf_drf.get_j(*args)-0.5*self.mf_drf.get_k(*args),self.mf_copy.get_k(*args))

        #if self.print_info : print(h_1e)
        
        # modify the mf object 1e hamiltonian routine
        mf.get_hcore = lambda *args: h_1e
        
        # modify the two-electron integrals
        #self.mf_drf._eri = ao2mo.restore(8, eri, N_AO)
        self.mf_drf.get_j = lambda *args,**kwargs  : self.get_j_drf(args[1])
        self.mf_drf.get_k = lambda *args,**kwargs  : self.get_k_drf(args[1])
        self.mf_drf.get_jk = lambda *args,**kwargs: (self.get_j_drf(args[1]),self.get_k_drf(args[1]))
        self.mf_drf.get_hcore = lambda *args: V
        #self.mf_copy.mo_occ = np.array([N_e/2,N_e/2])
        
        end = timer()
        if self.print_info : print("Modification time:",end-start,"s")
        #self.mf_drf.get_hcore = lambda *args: np.zeros(h_1e.shape)
       
        if not self.use_mf_pol:
            def get_veff(mf_obj,dm):
                if dm is None:
                    dm = mf_obj.make_rdm1()
                vxc = self.mf_copy.get_veff(dm=dm)
                vxc_drf = self.mf_drf.get_veff(dm=dm)
                #print(dir(vxc_drf))
                ecoul = vxc.ecoul
                exc = vxc.exc 

                vxc = lib.tag_array(vxc+vxc_drf, ecoul=(ecoul+vxc_drf.ecoul), exc=(exc+vxc_drf.exc), vj=None, vk=None)

                return vxc
        
        # mean-field polarization version
        if self.use_mf_pol:
            
            # modified get_veff for mean-field polarization
            def get_veff(mf_obj,dm):
                if dm is None:
                    dm = mf_obj.make_rdm1()
                vxc = self.mf_copy.get_veff(dm=dm)

                q = np.einsum('Aij,ij->A',self.Q,dm)
                vxc_mf = -np.einsum('A,Aij->ij',q,self.UQ)
                e_mf = -0.5 * q.dot(U_QM.dot(q))
                #print(dir(vxc_drf))
                ecoul = vxc.ecoul + e_mf
                exc = vxc.exc 

                vxc = lib.tag_array(vxc+vxc_mf, ecoul=(ecoul), exc=(exc), vj=None, vk=None)

                return vxc
            
        
        #if mf.__class__.__name__ == "RHF" or mf.__class__.__name__ == "UHF" :
        #    mf._eri = ao2mo.restore(8, eri, N_AO)
        
        # modify the output Veff
        #mf.get_veff = lambda *args: self.mf_copy.get_veff(*args) + self.mf_drf.get_veff(*args)
        #mf.get_veff = lambda *args: self.mf_copy.get_veff(*args) + self.mf_drf.get_veff(dm=mf.make_rdm1())
        if mf.__class__.__name__ == "RKS" :
            if is_hyb_dft and not is_lr_dft:
                #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)-0.5*self.mf_drf.get_k(*args),self.mf_copy.get_k(*args))
                #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)-0.5*self.mf_drf.get_k(*args)
                #U_MM
                #mf.get_jk = lambda *args:(self.mf_copy.get_j(*args)+get_j_drf(args[1])-0.5*get_k_drf(args[1]),self.mf_copy.get_k(*args))
                #mf.get_jk =(self.mf_copy.get_j(*args)+get_j_drf(args[1])-0.5*get_k_drf(args[1]),self.mf_copy.get_k(*args))
                mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
                
            elif not is_hyb_dft : 
                #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)-0.5*self.mf_drf.get_k(*args)
                mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
            if is_lr_dft:
                #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args)+self.get_j_drf(args[1])-0.5*self.get_k_drf(args[1]),self.mf_copy.get_k(*args))
                #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args,**kwargs),self.mf_copy.get_k(*args)+self.mf_drf.get_k(*args,**kwargs))
                
                #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args)+self.get_j_drf(args[1])-0.5*self.get_k_drf(args[1])
                mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
                #raise Exception("Range separated hybrids not supported for DRF.")
                #mf.energy_elec = lambda *args,**kwargs:  self.mf_drf.energy_elec(dm=mf.make_rdm1())
            
            #mf.get_jk = lambda *args,hermi=0: (self.mf_copy.get_j(*args)+get_j_drf(args[1])-0.5*self.mf_drf.get_k(*args),self.mf_copy.get_k(*args))
            #mf.get_jk = lambda *args : ([],0)  
            
        elif mf.__class__.__name__ == "UKS" and is_hyb_dft:
            if is_hyb_dft :
                #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)-self.mf_drf.get_k(*args,**kwargs),self.mf_copy.get_k(*args))
                mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
            elif not is_hyb_dft : 
                #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)-self.mf_drf.get_k(*args)
                mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
            #if is_lr_dft:
            #    mf.get_j = lambda *args: self.mf_copy.get_j(*args)+self.mf_drf.get_j(*args)
            #    mf.get_k = lambda *args: self.mf_copy.get_k(*args)+self.mf_drf.get_k(*args)
            if is_lr_dft:
                #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args,**kwargs)-self.mf_drf.get_k(*args,**kwargs),self.mf_copy.get_k(*args))
                #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args,**kwargs)-self.mf_drf.get_k(*args,**kwargs)
                mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
                
        if mf.__class__.__name__ in ["DFRKS","DFUKS","DFRHF","DFUHF"] :
            mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
            
        if mf.__class__.__name__ == "RHF" :
            #mf.get_j = lambda *args: self.mf_copy.get_j(*args)+self.mf_drf.get_j(*args)-0.5*self.mf_drf.get_k(*args)
            #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args),self.mf_copy.get_k(*args)+self.mf_drf.get_k(*args))
            #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)
            #mf.get_k = lambda *args,**kwargs: self.mf_copy.get_k(*args,**kwargs)+self.mf_drf.get_k(*args)
            mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
        elif mf.__class__.__name__ == "UHF":
            #mf.get_j = lambda *args: self.mf_copy.get_j(*args)+self.mf_drf.get_j(*args)-self.mf_drf.get_k(*args)
            #mf.get_jk = lambda *args,**kwargs: (self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args),self.mf_copy.get_k(*args)+self.mf_drf.get_k(*args))
            #mf.get_j = lambda *args,**kwargs: self.mf_copy.get_j(*args,**kwargs)+self.mf_drf.get_j(*args)
            #mf.get_k = lambda *args,**kwargs: self.mf_copy.get_k(*args,**kwargs)+self.mf_drf.get_k(*args)
            mf.get_veff = lambda *args,**kwargs: get_veff(mf,args[1])
        
        
        
        #mf.energy_elec = lambda *args: self.mf_copy.energy_elec(*args) + self.mf_drf.energy_elec(*args)
        if return_U_MM:
            return mf, -0.5*U_MM
        else:
            return mf
    
     #alternative formulation to just obtain J and K matrices
    def get_j_drf(self,dm):
        if len(dm.shape)==2:
            q = np.einsum('Aij,ij->A',self.Q,dm)
            return np.einsum('A,Aij->ij',q,self.UQ)
        elif len(dm.shape)==3:
            q = np.einsum('Aij,nij->nA',self.Q,dm)
            return np.einsum('nA,Aij->nij',q,self.UQ)

    def get_k_drf(self,dm):
        if len(dm.shape)==2:
            dmQ = np.einsum('jk,Bkl->Bjl',dm,self.Q)
            return np.einsum('Aij,Ajl->il',self.UQ,dmQ)
        elif len(dm.shape)==3:
            dmQ = np.einsum('njk,Bkl->nBjl',dm,self.Q)
            return np.einsum('Aij,nAjl->nil',self.UQ,dmQ)
    
    def genESPGrid(self,mol):
        """
        Generates a grid of points for ESP charge evaluation for the molecule
        :param mol: a pyscf.gto mol object
        :return grid_coords: an N x 3 array of coordinates of the grid points after pruning
        """
        Z = mol.atom_charges()
        r_vdw = radii.VDW[Z]
        if self.grid_method == "lebedev":
            grids = dft.gen_grid.Grids(mol)
            grids.atom_grid = self.esp_grid
            grids.radi_method = dft.gauss_chebyshev
            grids.atomic_radii = radii.VDW
            grids.kernel()

            grid_coords = grids.coords
            
            
        elif self.grid_method == "cartesian":
            box_lims = np.zeros((3,2))
            grids = []
            for alpha in range(0,3):
                x_min = np.min(self.x_QM[alpha,:])
                x_max = np.min(self.x_QM[alpha,:])
                box_lims[alpha,0] = x_min-self.box_pad 
                box_lims[alpha,1] = x_max+self.box_pad 
                grids.append(np.linspace(box_lims[alpha,0],box_lims[alpha,1],num=self.N_cart))
            
            
            grid_coords = lib.cartesian_prod(grids)
                
                
        
        for A in range( self.x_QM.shape[1]):
                x_A = self.x_QM[:,A]
                inc = np.linalg.norm(grid_coords-x_A,axis=1)>1.5*r_vdw[A]
                grid_coords = grid_coords[inc,:]
        
        return grid_coords
    
    def generateESPChargeOperators(self,overlaps,mol):
        """
        Generates ESP fitted charge operators
        """
        
        N_QM = self.x_QM.shape[1] 
        N_AO = overlaps.shape[0]
        if self.l_max_Q == 0:
            N_Q = N_QM 
        elif self.l_max_Q == 1:
            N_Q = 4*N_QM 
            
        #get r ints if dipoles requested
        if self.l_max_Q>0:
            rints = mol.intor('cint1e_r_sph',comp=3)
            
        Q = [np.zeros((N_AO,N_AO)) for n in range(0,N_Q)] 
        
        grid_coords = self.genESPGrid(mol)
        N_grid = grid_coords.shape[0]
        D = np.zeros((N_grid,N_Q))
        for A in range(0,N_QM):
            D[:,A] = 1./np.linalg.norm(grid_coords - self.x_QM[:,A],axis=1)
        
        # add dipole potential terms
        if self.l_max_Q>0:
            for alpha in range(0,3):
                for A in range(0,N_QM):
                    d = grid_coords - self.x_QM[:,A]
                    D[:,N_QM*(1+alpha)+A] = d[:,alpha]/(np.linalg.norm(d,axis=1)**3)
        
        #if self.print_info : print(D)
        mol_A = mol.copy() 
        mol_B = mol.copy() 
        for A in range(0,N_QM):
            for B in range(A,N_QM):
                #if self.print_info : print("A:",A,"B:",B)
                bas_start_A, bas_end_A, ao_start_A, ao_end_A = mol.aoslice_by_atom()[A]
                bas_start_B, bas_end_B, ao_start_B, ao_end_B = mol.aoslice_by_atom()[B]
                mol_A = mol.copy() 
                mol_B = mol.copy() 
                #if self.print_info : print(bas_start_A, bas_end_A, ao_start_A, ao_end_A)
                #if self.print_info : print(bas_start_B, bas_end_B, ao_start_B, ao_end_B)
                mol_A._bas = mol._bas[bas_start_A:bas_end_A]
                mol_B._bas = mol._bas[bas_start_B:bas_end_B]
                # generates the ESP integrals for AOs on A and B
                N_A =  ao_end_A -  ao_start_A
                N_B =  ao_end_B -  ao_start_B
                esp_AB = np.zeros((N_grid,N_A,N_B))
                #for k in range(0,N_grid):
                #    coords = grid_coords[k,:].reshape((1,3))
                #    esp_AB[k,:,:]  = gto.intor_cross('cint1e_grids_sph',mol_A,mol_B,grids=coords)
                
                block_size = self.grid_block_size 
                N_block = int(np.ceil(N_grid/block_size))
                for k in range(0,N_block):
                    start = k*block_size
                    end = min((k+1)*block_size,N_grid)
                    coords = grid_coords[start:end,:]
                    esp_AB[start:end,:,:]  = gto.intor_cross('cint1e_grids_sph',mol_A,mol_B,grids=coords)
                    
                #esp_AB = gto.intor_cross('cint1e_grids_sph',mol_A,mol_B,grids=grid_coords)
                #if self.print_info : print(esp_AB.shape)
                #N_A = esp_AB.shape[1]
                #N_B = esp_AB.shape[2]
                #Q_AB = [np.zeros(esp[0,:,:].shape)] * N_QM
                for n in range(0,N_A):
                    for m in range(0,N_B):
                        esp = -1.0*esp_AB[:,n,m] 
                        nu = ao_start_A+n
                        mu = ao_start_B+m
                        if self.l_max_Q ==0 :
                            Q_nm_fit = np.linalg.solve(D.T.dot(D),D.T.dot(esp))
                            Q_nm_fit = Q_nm_fit - (1./N_QM) * (overlaps[nu,mu] + np.sum(Q_nm_fit))
                        
                        elif self.l_max_Q == 1:
                            dim = N_Q + 4
                            A_fit = D.T.dot(D) 
                            b_fit = D.T.dot(esp)
                            A_con = np.zeros((dim,dim))
                            b_con = np.zeros((dim,))
                            A_con[0:N_Q,0:N_Q] = A_fit 
                            b_con[0:N_Q] = b_fit
                            # charge constraint
                            b_con[N_QM] = -overlaps[nu,mu]
                            A_con[N_QM,0:N_QM] = 1.
                            A_con[0:N_QM,N_QM] = 1.
                            # dipole constraints
                            for alpha in range(0,3):
                                A_con[N_QM+1+alpha,0:N_QM] = self.x_QM[alpha,:]
                                A_con[0:N_QM,N_QM+1+alpha] = self.x_QM[alpha,:]
                                A_con[N_QM+1+alpha,(N_QM*alpha+1):(N_QM*(alpha+2))] = 1.
                                A_con[(N_QM*alpha+1):(N_QM*(alpha+2)),N_QM+1+alpha] = 1.
                                b_con[N_QM+1+alpha] = -rints[alpha,nu,mu]
                            #if self.print_info : print(A_con)
                            #if self.print_info : print(b_con)
                            #A_con[N_Q:,N_Q:] = A_con[N_Q:,N_Q:] + np.eye(4)*0.e-0
                            #Q_nm_fit = np.linalg.solve(A_con,b_con)
                            Q_nm_fit = np.zeros((N_Q,))
                            if self.no_charge:
                                Q_nm_fit[N_QM:] = np.linalg.solve(A_fit[N_QM:,N_QM:],b_fit[N_QM:])
                                #Q_nm_fit[N_QM:] = Q_nm_fit[N_QM:] - (1./N_QM) * (overlaps[nu,mu] + np.sum(Q_nm_fit[N_QM:]))
                            else:
                                Q_nm_fit = np.linalg.solve(A_fit,b_fit)
                                Q_nm_fit[0:N_QM] = Q_nm_fit[0:N_QM] - (1./N_QM) * (overlaps[nu,mu] + np.sum(Q_nm_fit[0:N_QM]))
                            
                            for alpha in range(0,3):
                                Q_nm_fit[(N_QM*(alpha+1)):(N_QM*(alpha+2))] = Q_nm_fit[ (N_QM*(alpha+1)):(N_QM*(alpha+2)) ] \
                                    - (1./N_QM) * ( rints[alpha,nu,mu] + (np.sum( Q_nm_fit[(N_QM*(alpha+1)):(N_QM*(alpha+2))]) + np.sum( self.x_QM[alpha,:]*Q_nm_fit[0:N_QM] )) ) 
                        
                        
                        for C,Q_C in enumerate(Q):
                            q_C = Q_nm_fit[C]
                            Q_C[nu,mu] = q_C+0.
                            Q_C[mu,nu] = q_C+0.

        
        return Q
    
    def setupRepulsion(self,Z_QM,Z_MM):
        """
        Sets up repulsion parameters
        U = A/r exp(-(r/r_0))
        where A = Z_eff,X Z_eff,Y 1/r_0 = 1/2 (1/R_X + 1/R_Y)
        """

        N_QM = len(Z_QM)
        N_MM = len(Z_MM)
        # setup Z_eff
        self.Z_eff_QM = np.zeros((N_QM,))
        self.Z_eff_MM = np.zeros((N_MM,))
        for A,Z in enumerate(Z_QM):
            if Z<=2 :
                self.Z_eff_QM[A] = Z
            elif Z>2 and Z<=10:
                self.Z_eff_QM[A] = Z - 2 
            elif Z>10 and Z<=18:
                self.Z_eff_QM[A] = Z - 10
        
        for A,Z in enumerate(Z_MM):
            if Z<=2 :
                self.Z_eff_MM[A] = Z
            elif Z>2 and Z<=10:
                self.Z_eff_MM[A] = Z - 2 
            elif Z>10 and Z<=18:
                self.Z_eff_MM[A] = Z - 10

        self.R_QM = radii.VDW[Z_QM]
        self.R_MM = radii.VDW[Z_MM] 

        self.r_0_inv = np.zeros((N_QM,N_MM))
        self.A_pauli = np.zeros((N_QM,N_MM))
        

        for A in range(0,N_QM):
            #self.r_0_inv[A,:] = 3.0 * (1./self.R_MM + 1./self.R_QM[A])
            #self.A_pauli[A,:] = (self.Z_eff_QM[A] * self.Z_eff_MM)
            
            self.r_0_inv[A,:] = (16./(self.R_MM + self.R_QM[A]))
            self.r_0_inv[A,:] = 8.*(1./self.R_MM + 1./self.R_QM[A])
            self.A_pauli[A,:] = (self.Z_eff_QM[A] * self.Z_eff_MM)**2
            
            #self.r_0_inv[A,:] = (16./(self.R_MM + self.R_QM[A]))
            #self.r_0_inv[A,:] = 6.*(1./self.R_MM + 1./self.R_QM[A])
            #self.A_pauli[A,:] = (0.5*(self.Z_eff_QM[A] * self.Z_eff_MM)) * (((self.R_MM + self.R_QM[A])**2 /((self.R_MM *self.R_QM[A]))))
            self.r_0_inv[A,:] = (16./(self.R_MM + self.R_QM[A]))
            self.A_pauli[A,:] = 1.0*(self.Z_eff_QM[A] * self.Z_eff_MM)# / (self.R_MM + self.R_QM[A])


        return

    def calculateRepulsion(self):

        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]

        U_rep = 0.0
        for A in range(0,N_QM):
            r = np.linalg.norm(self.x_MM - self.x_QM[:,A].reshape((3,1)),axis=0)
            U_rep = U_rep + np.sum( (self.A_pauli[A,:] / r) * np.exp(-self.r_0_inv[A,:] * r)   )
            #U_rep = U_rep + np.sum( (self.A_pauli[A,:] ) * np.exp(-self.r_0_inv[A,:] * r)   )

        return U_rep
    
    def setupLJRep(self,sigma_QM,epsilon_QM,sigma_MM,epsilon_MM,n_LJ=12,xi=13.772,Z_QM=[],Z_MM=[]):
        """
        Sets up LJ type 1/r^12 repulsion
        
        """
        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]
        self.C_12 = np.zeros((N_QM,N_MM))
        self.epsilon = np.zeros((N_QM,N_MM))
        self.C_6 = np.zeros((N_QM,N_MM))
        self.sigma = np.zeros((N_QM,N_MM))
        self.n_LJ = n_LJ
        self.xi = xi 
        for A in range(0,N_QM):
            epsilons = np.sqrt(epsilon_QM[A] * epsilon_MM)
            sigmas = 0.5 * (sigma_MM + sigma_QM[A])
            #sigmas = np.sqrt((sigma_MM * sigma_QM[A]))
            self.C_12[A,:] = 4.0*epsilons * (sigmas**self.n_LJ)
            self.C_6[A,:] = 4.0*epsilons * (sigmas**6)
            self.epsilon[A,:] = 1.0*epsilons
            self.sigma[A,:] = 1.0*sigmas
        
        if self.damp_elec:
            self.Z_QM = Z_QM 
            self.Z_MM = Z_MM 
            self.R_MM = radii.COVALENT[self.Z_MM]
            self.R_QM = radii.COVALENT[self.Z_QM]
            #self.R_MM = radii.VDW[self.Z_MM]
            #self.R_QM = radii.VDW[self.Z_QM]
            self.R_damp = np.zeros((N_QM,N_MM))
            for A in range(0,N_QM):
                self.R_damp[A,:] = self.R_QM[A] + self.R_MM
            #self.R_damp = self.sigma 
        else:
            self.R_damp = np.empty((N_QM,0))
                
        
        
        return
    
    def setDampingParameters(self,mode="cov",Z_MM=None,Z_QM=None,R_damp=None):
        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]
        self.Z_QM = Z_QM 
        self.Z_MM = Z_MM 
        if self.damp_elec:
            
            if mode == "cov" or mode == "vdw":
                N_QM = len(Z_QM)
                N_MM = len(Z_MM)
                self.Z_QM = Z_QM 
                self.Z_MM = Z_MM 
                if mode =="cov":
                    self.R_MM = radii.COVALENT[self.Z_MM]
                    self.R_QM = radii.COVALENT[self.Z_QM]
                elif mode == "vdw":
                    self.R_MM = radii.VDW[self.Z_MM]
                    self.R_QM = radii.VDW[self.Z_QM]
                
                self.R_damp = np.zeros((N_QM,N_MM))
                for A in range(0,N_QM):
                    self.R_damp[A,:] = self.R_QM[A] + self.R_MM
            elif mode == "custom":
                if R_damp is None:
                    self.R_damp = np.empty((N_QM,0))
                else:
                    self.R_damp = R_damp

        else:
            self.R_damp = np.empty((N_QM,0))
        
        return
    
    def calculateLJRepulsion(self):
        
        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]

        U_rep = 0.0
        for A in range(0,N_QM):
            r_12_inv = np.linalg.norm(self.x_MM - self.x_QM[:,A].reshape((3,1)),axis=0)**(-self.n_LJ)
            U_rep = U_rep + np.sum(  self.C_12[A,:] * r_12_inv  )

        return U_rep
    
    def calculateBuckRepulsion(self,pairwise=False):
        
        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]
        if pairwise and self.l_max_Q>0:
            U_reps = np.zeros((4*N_QM,N_MM))
        else:
            U_reps = np.zeros((N_QM,N_MM))
        y = 2.0**(1.0/6.0)
        
        for A in range(0,N_QM):
            r = np.linalg.norm(self.x_MM - self.x_QM[:,A].reshape((3,1)),axis=0)
            U_reps[A,:] = (  2.0*self.epsilon[A,:] * (6.0/(self.xi-6.0) * np.exp(self.xi * (1.0-r/(y*self.sigma[A,:])) )))
            #U_reps[A,:] = (y*self.sigma[A,:]/r)*(  2.0*self.epsilon[A,:] * (6.0/(self.xi-6.0) * np.exp(self.xi * (1.0-r/(y*self.sigma[A,:])) )))
        
        if pairwise and self.l_max_Q>0:
            for A in range(0,N_QM):
                r = np.linalg.norm(self.x_MM - self.x_QM[:,A].reshape((3,1)),axis=0)
                dx = self.x_MM - self.x_QM[:,A].reshape((3,1))
                dU_dr = (  2.0*self.epsilon[A,:] * (6.0/(self.xi-6.0) *(-self.xi/(y*self.sigma[A,:])) * np.exp(self.xi * (1.0-r/(y*self.sigma[A,:])) )))
                U_reps[A+N_QM,:] = -dx[0,:]/r * dU_dr
                U_reps[A+N_QM*2,:] = -dx[1,:]/r * dU_dr
                U_reps[A+N_QM*3,:] = -dx[2,:]/r * dU_dr
                #U_reps[A,:] = (y*self.sigma[A,:]/r)*(  2.0*self.epsilon[A,:] * (6.0/(self.xi-6.0) * np.exp(self.xi * (1.0-r/(y*self.sigma[A,:])) )))
        
        
        if pairwise:
            return (U_reps)
        else:
            return np.sum(U_reps)
    
    def setupExRepParams(self,N_val,R_exrep):
        
        self.N_val_MM = np.array(N_val)
        self.beta_MM = np.zeros(self.N_val_MM.shape)
        N_MM = len(N_val)
        rho_0 = 1e-3
        if self.single_gto_exrep:
            for B in range(0,N_MM):
                R_0 = 1.0*R_exrep[B]
                #func = lambda beta : (np.log(rho_0) -  np.log(self.N_val_MM[B]*(np.abs(beta/np.pi)**(3/2) ))+(-np.abs(beta)*R_0*R_0))
                func = lambda beta : ((rho_0) -  (self.N_val_MM[B]*(np.abs(beta/np.pi)**(3/2) ))*np.exp(-np.abs(beta)*R_0*R_0))
                beta_sol = fsolve(func,np.array([1.0]),xtol=1e-8)
                self.beta_MM = np.abs(beta_sol)
        else:
            
            mol_H = gto.M("H 0 0 0",basis=self.ex_rep_bas,spin=1)
            sto_nG_info = np.array(mol_H._basis["H"][0][1:])
            beta_0 = sto_nG_info[:,0]
            c_0 = sto_nG_info[:,1]
            self.beta_MM = np.zeros((N_MM,len(c_0)))
            self.c_MM = np.zeros((N_MM,len(c_0)))
            norm = 0.
            for i in range(0,len(c_0)):
                norm = norm + np.sum( (4.0*beta_0[i]*beta_0/((beta_0[i]+beta_0)**2))**(0.75) * c_0[i]*c_0)
            #if self.print_info : print(norm)
            c_0 = c_0 / np.sqrt(norm)
            phi_nG = lambda r,gamma : (gamma**(1.5)) * np.sum(c_0 * (2.0*beta_0/np.pi)**(0.75) *np.exp(-(gamma * gamma) * beta_0 * r * r))
            
            for B in range(0,N_MM):
                R_0 = 0.75*R_exrep[B]
                #func = lambda beta : (np.log(rho_0) -  np.log(self.N_val_MM[B]*(np.abs(beta/np.pi)**(3/2) ))+(-np.abs(beta)*R_0*R_0))
                R_exp = 1.0*R_exrep[B] 
                func_exp = lambda alpha : np.log(rho_0) - np.log((self.N_val_MM[B]/(8.0*np.pi))*(np.abs(alpha)**3) * np.exp(-np.abs(alpha) * R_exp))
                alpha_sol = np.abs(fsolve(func_exp,1.0,xtol=1e-8))
                rho_exp = (self.N_val_MM[B]/(8.0*np.pi))*(alpha_sol**3) * np.exp(-alpha_sol * R_0)
                func = lambda gamma : rho_exp - self.N_val_MM[B]*(phi_nG(R_0,np.abs(gamma))**2)
                gamma_sol = fsolve(func,1.0,xtol=1e-8)
                self.N_val_MM[B] = 1.0 * self.N_val_MM[B]
                
                #R_0 = 1.0*R_exrep[B]
                #func = lambda gamma : rho_0 - self.N_val_MM[B]*(phi_nG(R_0,np.abs(gamma))**2)
                #gamma_sol = fsolve(func,1.0,xtol=1e-8)
                #if self.print_info : print(gamma_sol)
                self.beta_MM[B,:] = gamma_sol*gamma_sol * beta_0
                self.c_MM[B,:] = gamma_sol**(1.5) * c_0
                

        return
    
    def calculateHeffExRep(self,mol):
        
        N_AO = mol.nao
        N_QM = self.x_QM.shape[1]
        N_MM = self.x_MM.shape[1]
        h_rep = np.zeros((N_AO,N_AO))
        R_vdw_QM = radii.VDW[self.Z_QM]
        R_vdw_MM = radii.VDW[self.Z_MM]
        if self.single_gto_exrep:
            for B in range(0,N_MM):
                Delta_x = self.x_QM - self.x_MM[:,B].reshape((3,1))
                r_sq = np.sum(Delta_x * Delta_x,axis=0)
                r = np.sqrt(r_sq)
                r_thresh = self.exrep_scal * (R_vdw_QM + R_vdw_MM[B]) 
                dr = r - r_thresh
                if np.min(dr) < 0.0 :
                #if np.min(r)<self.exrep_cut:
                    basis = {"GHOST":[[0,[0.5*self.beta_MM[B],1.0]]]}
                    mol_B = gto.M(atom=[["GHOST",self.x_MM[:,B]]],basis=basis,unit="Bohr")
                    h_rep = h_rep + (0.5*self.scal_N_val*self.N_val_MM[B]) * gto.intor_cross("cint2e_sph",mol,mol_B)[:,0,:N_AO,N_AO]
        else:
            n_added = 0
            for B in range(0,N_MM):
                Delta_x = self.x_QM - self.x_MM[:,B].reshape((3,1))
                r_sq = np.sum(Delta_x * Delta_x,axis=0)
                r = np.sqrt(r_sq)
                r_thresh = self.exrep_scal * (R_vdw_QM + R_vdw_MM[B]) 
                dr = r - r_thresh
                if np.min(dr) < 0.0 :
                    n_added = n_added + 1
                #if np.min(r)<self.exrep_cut:
                    data = [ [self.beta_MM[B,n],self.c_MM[B,n]] for n in range(0,len(self.c_MM[B,:]))]
                    basis = {"GHOST":[[0]+data]}
                    mol_B = gto.M(atom=[["GHOST",self.x_MM[:,B]]],basis=basis,unit="Bohr")
                    mol_comb = mol+mol_B
                    N = mol.nbas
                    M = mol_comb.nbas
                    h_rep = h_rep + (0.5*self.scal_N_val*self.N_val_MM[B]) * mol_comb.intor("cint2e_sph",shls_slice=(0,N,N,M,N,M,0,N)).reshape((N_AO,N_AO))
                    #h_rep = h_rep + (0.5*self.N_val_MM[B]) * gto.intor_cross("cint2e_sph",mol,mol_B)[:,0,:N_AO,N_AO]
            if self.print_info : print("Number of repulsion terms: ",n_added)
        
            
                
        
        return h_rep
        
        
        
    def setupDRFPySCFTD(self,mytd):
        if not self.use_mf_pol:
            singlet = mytd.singlet 
            #V = self.mf_drf.get_hcore() 
            #self.mf_drf.get_hcore = lambda *args,**kwargs : np.zeros(V.shape)
            #h_1e = self.mf_copy.get_hcore()
            #self.mf_copy.get_hcore = lambda *args,**kwargs : (h_1e + V)
            vresp = self.mf_copy.gen_response(singlet=singlet,mo_coeff=mytd._scf.mo_coeff, mo_occ=mytd._scf.mo_occ,hermi=0)
            vresp_drf = self.mf_drf.gen_response(singlet=singlet,mo_coeff=mytd._scf.mo_coeff, mo_occ=mytd._scf.mo_occ,hermi=0)
            mytd._scf.gen_response = lambda singlet=singlet,hermi=0: (lambda *args2 : vresp(*args2)+vresp_drf(*args2))
        
        
        return mytd
    
    def setupQMEEPySCF(self,mf,use_exact_emb=False,return_V=False):
        
        if use_exact_emb:
            N_MM = self.x_MM.shape[1]
            mol = mf.mol
            N_AO = mol.nao
            V = np.zeros((N_AO,N_AO))
            
            for A in range(0,N_MM):
                grid = (self.x_MM[:,A]).reshape((1,3))
                V_A = mol.intor('cint1e_grids_sph',grids=grid)
                V = V + (-self.q_MM[A]) * V_A[0]
            U_nuc = 0.0
            Z_QM = mol.atom_charges()
            for A in range(0,N_MM):
                x_0 =  (self.x_MM[:,A]).reshape((3,1))
                f,phi = calculateFieldAndPot(self.x_QM,self.q_MM[A],x_0)
                U_nuc = U_nuc + np.einsum('A,A',phi,Z_QM)
                
            
        else:
            mol = mf.mol
            N_MM = self.x_MM.shape[1]
            N_QM = self.x_QM.shape[1]
            Z_QM = mol.atom_charges()
            N_AO = mol.nao
            # first get the ao_info from the mf.mol object
            ao_info = getPySCFAOInfo(mf.mol)

            # get charge operators
            Q = self.generateChargeOperators(method=self.charge_op_method,overlaps=ao_info["overlaps"],ao_atom_inds=ao_info["ao_atom_inds"],mol=mf.mol)
            Q = np.array(Q)
            # generate QM/MM / DRF variables This is overkill for fixed charge embedding - needs modifying
            U_MM,U_QMMM, U_QM, Phi_QM = self.calculateDRFVariables()
            
            U_nuc = np.einsum('A,A',Phi_QM[0:N_QM],Z_QM)
            V = np.einsum('A,Anm->nm',Phi_QM,Q)
        
        if self.add_exrep:
            V = V + self.calculateHeffExRep(mf.mol)
        
        V = V + (U_nuc/mol.nelectron) * mol.intor('cint1e_ovlp_sph')
        h_1e = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + V 
        mf.get_hcore = lambda *args : (h_1e)
            
        
        
        if return_V:    
            return mf, V
        else:
            return mf
    
    def setupBondsScalings(self,pairs_12=[], pairs_13=[],scal_12_perm=0.,scal_13_perm=0,scal_12_indperm=0.,scal_13_indperm=0.):
        N_MM = len(self.q_MM)
        
        self.pairs_12 = pairs_12 
        
        # set up 1-3 bonded pairs
        if pairs_13 == [] :
            
            #self.pairs_13 = []
            #for p1 in self.pairs_12:
            #    A,B = p1
            #    # loop over all other atoms
            #    for C in range(0,N_MM):
            #        if C ==A or C==B:
            #            continue
            #        # if A-C bonded and B-C not bonded add
            #        elif [A,C] in self.pairs_12 or [C,A] in pairs_12:
            #            if not ([B,C] in self.pairs_13 or [C,B] in self.pairs_13):
            #                self.pairs_13.append([B,C])
            #        # if B-C bonded and A-C not bonded add
            #        elif [B,C] in self.pairs_12 or [C,B] in pairs_12:
            #            if not ([A,C] in self.pairs_13 or [C,A] in self.pairs_13):
            #                self.pairs_13.append([A,C])
            
            self.pairs_13 = []
            n_pairs = len(self.pairs_12)
            for n in range(0,n_pairs):
                p1 = self.pairs_12[n]
                A,B = p1
                
                for m in range(n+1,n_pairs):
                    p2 = self.pairs_12[m]
                    C,D = p2
                    if A==C:
                        self.pairs_13.append([B,D])
                    elif A==D:
                        self.pairs_13.append([B,C])
                    elif B==C:
                        self.pairs_13.append([A,D])
                    elif B==D:
                        self.pairs_13.append([A,C])
                            
        else:
            self.pairs_13 = pairs_13
        
        new_13 = [p for p in self.pairs_13 if p not in self.pairs_12]
        self.pairs_13 = new_13       
            
        self.scal_perm = np.ones((N_MM,N_MM))
        for p in self.pairs_13:
            A,B = p 
            self.scal_perm[A,B] = scal_13_perm
            self.scal_perm[B,A] = scal_13_perm
        for p in self.pairs_12:
            A,B = p 
            self.scal_perm[A,B] = scal_12_perm
            self.scal_perm[B,A] = scal_12_perm
        
        self.scal_indperm = np.ones((N_MM,N_MM))
        for p in self.pairs_13:
            A,B = p 
            self.scal_indperm[A,B] = scal_13_indperm
            self.scal_indperm[B,A] = scal_13_indperm
        for p in self.pairs_12:
            A,B = p 
            self.scal_indperm[A,B] = scal_12_indperm
            self.scal_indperm[B,A] = scal_12_indperm
            
            
        return
            
    
            
        
    
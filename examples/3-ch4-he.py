

import sys,os
sys.path.append('../pyespf/')

import numpy as np
from QMMMPol import QMMMPol, getPySCFAOInfo
from pyscf import gto, scf, ao2mo, data, dft, tddft
from pyscf.data import radii
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.style as pltstyle
#pltstyle.use('classic')


"""
This example runs the CH4 + He calcualtion
"""
#  separations in Bohr
N_r = 30

r_vals = np.linspace(8.,3.5,num=N_r) / radii.BOHR
nstates = 0
energies = np.zeros((r_vals.shape[0],nstates+1))

# basic info for the QM/MM system  (all coordinates in Bohr)
r_CH = 2.0 

# basic info for the QM/MM system  (all coordinates in Bohr)
QM_atoms = ["C","H","H","H","H"]
x_QM = r_CH*np.array([[0.,0.,0.],[1.,0.,0.],[-1./3.,2.*np.sqrt(2.)/3.,0.],
                 [-1./3.,-np.sqrt(2.)/3.,np.sqrt(6.)/3.],[-1./3.,-np.sqrt(2.)/3.,-np.sqrt(6.)/3.]]).T

x_MM = (np.array([[100.,0.,0.]])).T
alpha_MM = np.array([0.0]) 


q_MM = np.array([0.0e0])
charge_QM = 0
spin_QM = 0
basis_set = "cc-pVQZ-F12"
# cc-pVQZ-F12 from www.basissetexchange.org
basis_set = {"H":gto.parse('''
#BASIS SET: (8s,4p,2d,1f) -> [5s,4p,2d,1f]
H    S
    402.0000000              0.0002790              0.0000000              0.0000000              0.0000000              0.0000000
     60.2400000              0.0021650              0.0000000              0.0000000              0.0000000              0.0000000
     13.7300000              0.0112010              0.0000000              0.0000000              0.0000000              0.0000000
      3.9050000              0.0448780              0.0000000              0.0000000              0.0000000              0.0000000
      1.2830000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.4655000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.1811000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0727900              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
H    P
      3.2599000              1.0000000              0.0000000              0.0000000              0.0000000
      1.2389000              0.0000000              1.0000000              0.0000000              0.0000000
      0.4708000              0.0000000              0.0000000              1.0000000              0.0000000
      0.1789000              0.0000000              0.0000000              0.0000000              1.0000000
H    D
      1.1111000              1.0000000              0.0000000
      0.3501000              0.0000000              1.0000000
H    F
      0.4796000              1.0000000
      '''),
            "C":gto.parse('''
            #BASIS SET: (15s,9p,4d,3f,2g) -> [7s,7p,4d,3f,2g]
C    S
  96770.0000000              0.0000250             -0.0000050              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
  14500.0000000              0.0001900             -0.0000410              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   3300.0000000              0.0010000             -0.0002130              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    935.8000000              0.0041830             -0.0008970              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    306.2000000              0.0148590             -0.0031870              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    111.3000000              0.0453010             -0.0099610              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     43.9000000              0.1165040             -0.0263750              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     18.4000000              0.2402490             -0.0600010              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      8.0540000              0.3587990             -0.1068250              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      3.6370000              0.2939410             -0.1441660              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.6560000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.6333000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.2545000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.1019000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0394000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
C    P
    101.8000000              0.0008910              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     24.0400000              0.0069760              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      7.5710000              0.0316690              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      2.7320000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.0850000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.4496000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.1876000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.0760600              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0272000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
C    D
      3.7056000              1.0000000              0.0000000              0.0000000              0.0000000
      1.4212000              0.0000000              1.0000000              0.0000000              0.0000000
      0.5451000              0.0000000              0.0000000              1.0000000              0.0000000
      0.2091000              0.0000000              0.0000000              0.0000000              1.0000000
C    F
      1.4438000              1.0000000              0.0000000              0.0000000
      0.5931000              0.0000000              1.0000000              0.0000000
      0.2436000              0.0000000              0.0000000              1.0000000
C    G
      1.1825000              1.0000000              0.0000000
      0.4685000              0.0000000              1.0000000
            ''')}
# quick calculation
basis_set = "6-31G"
qmmmpol = QMMMPol()

#qmmmpol.esp_grid = (16,38)
qmmmpol.charge_op_method = "esp"
#qmmmpol.charge_op_method = "mulliken"
#qmmmpol.grid_method = "lebedev"
qmmmpol.setupQMMMPol(x_MM,q_MM,alpha_MM,x_QM)
qmmmpol.damp_elec = True

# set-up the pyscf mol and scf object for calculation without X-
atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))]
mol = gto.M(atom=atom_info, unit="Bohr", basis=basis_set, charge=charge_QM, spin=spin_QM)
Z_QM = mol.atom_charges()
mf = scf.RHF(mol)
mf = scf.RKS(mol)
mf = dft.RKS(mol).apply(scf.addons.remove_linear_dep_)
xc_name = "PBE0"
mf.xc = xc_name
mf.tol_conv = 1e-12
mf.kernel()
mu_gs = mf.dip_moment(unit="AU")
mu_sq = np.linalg.norm(mu_gs)**2
print(mu_sq)

dm = mf.make_rdm1()
print("Reference energy:")
E_ref = mf.energy_tot()


q_MM = np.array([0.0e0])
alpha_MM = np.array([1.20409]) # cc-pVQZ-F12
Z_MM = [2]
R_dens = np.array([1.34])/radii.BOHR
N_val = np.array([2.0])

# DRF calculations
print("DRF calculations:")
for n,r in enumerate(r_vals):

    # set-up separation for MM He atom
    x_MM = (np.array([[r,0.,0.]])).T
    #qmmmpol.Q = None
    qmmmpol = QMMMPol()
    qmmmpol.setupQMMMPol(x_MM,q_MM,alpha_MM,x_QM)
    
    # set-up the pyscf mol and scf objects
    atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))]
    mol = gto.M(atom=atom_info, unit="Bohr", basis=basis_set, charge=charge_QM, spin=spin_QM)
    mf = dft.RKS(mol)
    mf.xc = xc_name
    # set-up the QM/MM calculation
    # use mean-field instead of DRF (default false)
    qmmmpol.use_mf_pol = False
    # Charge only: l_max_Q = 0 , with dipoles: l_max_Q = 1
    qmmmpol.l_max_Q = 1 
    # Use ESPF method for charge operators (default)
    qmmmpol.charge_op_method = "espf"
    # Grid for ESPF
    qmmmpol.esp_grid = (32,38)
    # Use ESPF or exact method for fixed charges
    qmmmpol.use_exact_qmee = False
    # Turn on damping for QM-MM interactions
    qmmmpol.damp_elec = True
    # default using covalent radii
    qmmmpol.setDampingParameters(mode="cov",Z_QM=Z_QM,Z_MM=Z_MM)
    # add exchange repulsion term
    qmmmpol.add_exrep = True
    # sets cut off for ex-rep in terms of covalent radii
    qmmmpol.exrep_scal = 10.0
    qmmmpol.ex_rep_bas = "STO-3G"
    qmmmpol.setupExRepParams(N_val,R_dens)

    # set up modified PySCF scf/dft object
    mf = qmmmpol.setupDRFPySCF(mf)
    #print(mf.get_hcore())
    mf.tol_conv = 1e-12
    # get energy
    mf.kernel(dm=dm) 
    energies[n,0] = mf.energy_tot()
    mf.dip_moment(unit="AU")
    dm = mf.make_rdm1()

    

print("E(r)-E_ref [Hartree]")
print(energies - E_ref)
scal = lambda x : x
plt.plot(r_vals*data.radii.BOHR,scal((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")

plt.ylabel("Energy [Hartrees]")
plt.xlabel("Separation [Angstrom]")

plt.legend()
plt.show()


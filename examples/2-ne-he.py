

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
This example calculates the energy for the Ne He system 
"""

#  separations in Bohr
N_r = 101
r_vals = np.linspace(10.,2.5,num=N_r) / radii.BOHR
nstates = 0
energies = np.zeros((r_vals.shape[0],nstates+1))
energies_l0 = np.zeros(r_vals.shape)

# basic info for the QM/MM system  (all coordinates in Bohr)
QM_atoms = ["Ne"]
x_QM = np.array([[0.,0.,0.]]).T

x_MM = (np.array([[1000.,0.,0.]])).T
alpha_MM = np.array([0.0]) 


q_MM = np.array([0.0e0])
charge_QM = 0
spin_QM = 0
# cc-pVQZ-F12 from basis set exchange (www.basissetexchange.org)
basis_set = {"Ne":'''
#BASIS SET: (15s,9p,4d,3f,2g) -> [7s,7p,4d,3f,2g]
Ne    S
 262700.0000000              0.0000260             -0.0000060              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
  39350.0000000              0.0002000             -0.0000470              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   8955.0000000              0.0010500             -0.0002470              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   2538.0000000              0.0044000             -0.0010380              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    829.9000000              0.0156490             -0.0037110              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    301.5000000              0.0477580             -0.0115930              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    119.0000000              0.1229430             -0.0310860              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     50.0000000              0.2524830             -0.0709720              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     21.9800000              0.3663140             -0.1272660              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      9.8910000              0.2796170             -0.1512310              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      4.3270000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.8040000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.7288000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.2867000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0957000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Ne    P
    299.1000000              0.0010380              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     70.7300000              0.0083750              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     22.4800000              0.0396930              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      8.2460000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      3.2690000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.3150000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.5158000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.1918000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0654000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Ne    D
     11.9220000              1.0000000              0.0000000              0.0000000              0.0000000
      4.0799000              0.0000000              1.0000000              0.0000000              0.0000000
      1.3963000              0.0000000              0.0000000              1.0000000              0.0000000
      0.4778000              0.0000000              0.0000000              0.0000000              1.0000000
Ne    F
      7.1249000              1.0000000              0.0000000              0.0000000
      2.3230000              0.0000000              1.0000000              0.0000000
      0.7574000              0.0000000              0.0000000              1.0000000
Ne    G
      4.4685000              1.0000000              0.0000000
      0.9718000              0.0000000              1.0000000
      '''}
# quick calcuation with 6-31G
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
xc_name = "PBE0"
mf.xc = xc_name
mf.tol_conv = 1e-12
mf.kernel()
mu_gs = mf.dip_moment(unit="AU")

dm = mf.make_rdm1()
print("Reference energy:")
E_ref = mf.energy_tot()

# MM region parameters
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

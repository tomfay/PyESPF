

import sys,os
sys.path.append('../pyespf')

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
This example calculates the energy for the Na+ He system 
"""

#  separations in Bohr
N_r = 101
r_vals = np.linspace(10.,2.0,num=N_r) / radii.BOHR
nstates = 0 
energies = np.zeros((r_vals.shape[0],nstates+1))

# basic info for the QM/MM system  (all coordinates in Bohr)

# the QM system
QM_atoms = ["Na"]
x_QM = np.array([[0.,0.,0.]]).T

q_MM = np.array([0.0e0])
charge_QM = 1
spin_QM = 0
# cc-pVQZ-F12 from basis set exchange (www.basissetexchange.org)
basis_set = {"Na":'''#BASIS SET: (21s,15p,5d,3f,2g) -> [8s,8p,5d,3f,2g]
Na    S
2185572.0000000              0.0000020             -0.0000010              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
 327228.4000000              0.0000180             -0.0000040              0.0000010              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
  74466.8400000              0.0000950             -0.0000230              0.0000030              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
  21093.1500000              0.0004010             -0.0000980              0.0000150              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   6881.8980000              0.0014590             -0.0003570              0.0000540              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   2484.6960000              0.0047460             -0.0011650              0.0001750              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    969.2232000              0.0140310             -0.0034640              0.0005200              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    402.0643000              0.0377330             -0.0094950              0.0014310              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    175.3545000              0.0907020             -0.0235870              0.0035540              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     79.6519900              0.1864660             -0.0523940              0.0079540              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     37.3867200              0.3018370             -0.0980280              0.0149550              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     18.0019400              0.3238300             -0.1436730              0.0224720              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      8.7243710              0.1687000             -0.1022980              0.0162050              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      3.8577150              0.0230830              0.1380290             -0.0235500              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.8156860             -0.0001470              0.4290060             -0.0801200              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.8382540              0.0010990              0.4478130             -0.1129210              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.3819350              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.0716790              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.0339160              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.0165250              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0051000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Na    P
   1119.5780000              0.0001620             -0.0000160              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    265.3239000              0.0014080             -0.0001400              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     85.9955300              0.0075860             -0.0007540              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     32.5375900              0.0296150             -0.0029680              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     13.5156500              0.0884770             -0.0089190              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      5.9668560              0.1955190             -0.0200290              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      2.7000450              0.3066210             -0.0313800              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.2185120              0.3412420             -0.0364890              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.5421870              0.2296900             -0.0296250              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.2274130              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.1330400              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.0575770              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.0259710              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.0119010              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0055000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Na    D
      1.5223140              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.2353140              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.1361410              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.0787640              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0455690              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Na    F
      0.1811270              1.0000000              0.0000000              0.0000000
      0.1092000              0.0000000              1.0000000              0.0000000
      0.0658360              0.0000000              0.0000000              1.0000000
Na    G
      0.1578580              1.0000000              0.0000000
      0.0916070              0.0000000              1.0000000
      '''}

# set to STO-3G for a quick calculation
basis_set = "STO-3G"


#MM region parameters
q_MM = np.array([0.0e0])
alpha_MM = np.array([1.20409]) # cc-pVQZ-F12
Z_MM = [2]
R_dens = np.array([1.34])/radii.BOHR
N_val = np.array([2.0])
x_MM = (np.array([[100.,0.,0.]])).T

# set-up QMMMPol object - handles basis functions
qmmmpol = QMMMPol()

qmmmpol.charge_op_method = "esp"
#qmmmpol.charge_op_method = "mulliken"
#qmmmpol.grid_method = "lebedev"
qmmmpol.setupQMMMPol(x_MM,q_MM,alpha_MM,x_QM)
qmmmpol.damp_elec = True

# set-up the pyscf mol and scf object for calculation 
atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))]
mol = gto.M(atom=atom_info, unit="Bohr", basis=basis_set, charge=charge_QM, spin=spin_QM)
Z_QM = mol.atom_charges()
mf = scf.RKS(mol)
xc_name = "PBE0"
mf.xc = xc_name
mf.tol_conv = 1e-12

# Run reference energy calculation
mf.kernel()
mu_gs = mf.dip_moment(unit="AU")
mu_sq = np.linalg.norm(mu_gs)**2
dm = mf.make_rdm1()
print("Reference energy:")
E_ref = mf.energy_tot()



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
    # use custom (in this case used because radius for Na+ is much smaller than Na)
    R_damp = np.array([[1.02+0.28]])/radii.BOHR
    #R_damp = np.array([[1.34+1.33]])/radii.BOHR
    qmmmpol.setDampingParameters(mode="custom",R_damp=R_damp)
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




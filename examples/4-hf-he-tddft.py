

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
This example runs the HF + He interaction energy calculation for ground and excited states using TDA-TDDFT
"""

#  separations in Bohr
N_r = 30

r_vals = np.linspace(9.,2.5,num=N_r) / radii.BOHR
nstates = 3
singlet = True
energies = np.zeros((r_vals.shape[0],nstates+1))

# basic info for the QM/MM system  (all coordinates in Bohr)
QM_atoms = ["H","F"]
r_bond = 0.910 / radii.BOHR
x_QM = r_bond*np.array([[-1.0,0.,0.],[0.0,0.,0.0]]).T

x_MM = (np.array([[100.,0.,0.]])).T
alpha_MM = np.array([0.0]) 

q_MM = np.array([0.0e0])
charge_QM = 0
spin_QM = 0
#cc-pVQZ-F12 from www.basissetexchange.org
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
            "F":gto.parse('''
            #BASIS SET: (15s,9p,4d,3f,2g) -> [7s,7p,4d,3f,2g]
F    S
 211400.0000000              0.0000260             -0.0000060              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
  31660.0000000              0.0002010             -0.0000470              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   7202.0000000              0.0010560             -0.0002440              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
   2040.0000000              0.0044320             -0.0010310              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    666.4000000              0.0157660             -0.0036830              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
    242.0000000              0.0481120             -0.0115130              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     95.5300000              0.1232320             -0.0306630              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     40.2300000              0.2515190             -0.0695720              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     17.7200000              0.3645250             -0.1239920              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      8.0050000              0.2797660             -0.1502140              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      3.5380000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.4580000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.5887000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.2324000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0806000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
F    P
    241.9000000              0.0010020              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     57.1700000              0.0080540              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     18.1300000              0.0380480              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      6.6240000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      2.6220000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      1.0570000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.4176000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.1574000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0550000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
F    D
      9.2778000              1.0000000              0.0000000              0.0000000              0.0000000
      3.2485000              0.0000000              1.0000000              0.0000000              0.0000000
      1.1375000              0.0000000              0.0000000              1.0000000              0.0000000
      0.3983000              0.0000000              0.0000000              0.0000000              1.0000000
F    F
      4.5969000              1.0000000              0.0000000              0.0000000
      1.6112000              0.0000000              1.0000000              0.0000000
      0.5647000              0.0000000              0.0000000              1.0000000
F    G
      2.1149000              1.0000000              0.0000000
      0.7640000              0.0000000              1.0000000
            ''')}
# for quick calculation
#basis_set = "STO-3G"  

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

mf = scf.RKS(mol)
xc_name = "PBE0"
mf.xc = xc_name
mf.tol_conv = 1e-12
mf.kernel()
mu_gs = mf.dip_moment(unit="AU")

dm = mf.make_rdm1()
print("Reference energy:")
E_ref = mf.energy_tot()

mytd = tddft.TDA(mf)
mytd.singlet = singlet
mytd.nstates = nstates
mytd.kernel()
mytd.analyze(verbose=4)

xy = mytd.xy
x0 = np.array([np.hstack( (xy[n][0].flatten()) ) for n in range(0,nstates)]) 

E_refs = mytd.e 

# MM parmaeters
q_MM = np.array([0.0e0])
#alpha_MM = np.array([1.12648]) # cc-pVTZ-F12
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
    
    # TDA part
    mytd = tddft.TDA(mf)
    mytd.singlet = singlet
    mytd.nstates = nstates
    #mytd.max_cyc = 1000
    
    # modify TDA object to include ESPF-DRF terms
    mytd = qmmmpol.setupDRFPySCFTD(mytd)
    
    mytd.kernel(x0=x0)
    #mytd.kernel()
    #mytd.analyze(verbose=1)
    xy = mytd.xy
    x0 = np.array( [np.hstack( (xy[n][0].flatten()) ) for n in range(0,nstates)] ) 
    #x0_td = np.array([np.hstack( (np.array([xy[n][0].flatten(),np.zeros(xy[n][0].flatten().shape)) ) for n in range(0,nstates)]) 
    x0_td = np.array( [np.hstack( (xy[n][0].flatten(),np.zeros(xy[n][0].flatten().shape)) ) for n in range(0,nstates)] ) 

    energies[n,1:(len(mytd.e)+1)] = mytd.e + energies[n,0]
    

print("E(r)-E_ref [Hartree]")
print(energies - E_ref)

plt.plot(r_vals*data.radii.BOHR,((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,-2.0*mu_sq*alpha_MM[0]/(r_vals**6),"--",label="S${}_0 (dip)$",color="black")
for n in range(0,1):
    plt.plot(r_vals*data.radii.BOHR,((energies[:,1+n]-E_ref-E_refs[n])),"-",label="S${}_1$")

plt.ylabel("Interaction Energy [Hartrees]")
plt.xlabel("Separation [Angstrom]")

plt.legend()
plt.show()




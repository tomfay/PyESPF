

import sys,os
sys.path.append('../pyespf/')

import numpy as np
import scipy as sp
from QMMMPol import QMMMPol, getPySCFAOInfo
from pyscf import gto, scf, ao2mo, data, dft, tddft
from pyscf.data import radii
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


"""
This example calculates the energy for a non-polarisable X- ion and Li+ as a function of Li-X separation
using RHF for the Li+ with the DRF method for the interaction between the QM and MM regions
"""

np.set_printoptions(linewidth=125)

#  separations in Bohr
N_r = 15

r_vals = np.linspace(5.,1.5,num=N_r) / radii.BOHR
nstates = 3
energies = np.zeros((r_vals.shape[0],nstates+1))

# basic info for the QM/MM system  (all coordinates in Bohr)
charge_QM = 0
spin_QM = 0
basis_set = "def2-TZVP"
# for a quick calculation
basis_set = "6-31G"

mol = gto.M(atom="./data/acrolein-pbe0-aligned.xyz",charge=charge_QM,spin=spin_QM,basis=basis_set)
Z_QM = mol.atom_charges()
QM_atoms = mol.elements
ind_O = list(QM_atoms).index("O")
x_QM = mol.atom_coords().T
x_O = x_QM[:,ind_O]
x_QM = x_QM - x_O.reshape((3,1))


# the MM water atom
q_O = -0.6690
q_H = -0.5*q_O
alpha_O = 9.7180 
alpha_H = 1e-10
alpha_O = 5.7494  
alpha_H = 2.7929 
#alpha_O = 1e-15
#alpha_H = 1e-15

# H2O geometry
theta_HOH = np.pi*104.50/(180.0)
r_OH = 0.957 / radii.BOHR
#x_MM_ref = r_OH*(np.array([[0.,0.,0.],[np.cos(theta_HOH/2.),np.sin(theta_HOH/2.),0.],[np.cos(theta_HOH/2.),-np.sin(theta_HOH/2.),0.]])).T
x_MM_ref = r_OH*(np.array([[1.,0.,0.],[0.,0.,0.],[1.0+np.cos(np.pi-theta_HOH),np.sin(np.pi-theta_HOH),0.]])).T
alpha_MM = np.array([alpha_O,alpha_H,alpha_H])
q_MM = np.array([q_O,q_H,q_H])
bonds = [[0,1],[0,2]]
print(x_MM_ref)


# set-up the pyscf mol and scf object for calculation without X-
atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))]
mol = gto.M(atom=atom_info, unit="Bohr", basis=basis_set, charge=charge_QM, spin=spin_QM)
mf = scf.RKS(mol)
xc_name = "HYB_GGA_XC_WB97X_D3"
mf.xc = xc_name
mf.tol_conv = 1e-12
mf.kernel()

dm = mf.make_rdm1()
print("Reference energy:")
E_ref = mf.energy_tot()

mytd = tddft.TDA(mf)
mytd.singlet = True
mytd.nstates = nstates
mytd.kernel()
mytd.analyze(verbose=4)
xy = mytd.xy
#x0 = np.array([np.hstack( (xy[n][0].flatten()) ) for n in range(0,nstates)]) 
E_refs = mytd.e 



R_dens_MM = {"O":1.71/radii.BOHR ,"H":1.54/radii.BOHR} # H 1.54 neutral ~1.0 cation O original 1.71
N_val_MM = {"O":(4.0-q_O) ,"H":(1.0-q_H)}
Z_MM = [8,1,1]

print("QMMM-ESPF-DRF calculations:")
for n,r in enumerate(r_vals):

    # set-up separation for MM Xe atom
    x_MM = (np.array([[r,0.,0.]])).T + x_MM_ref
    qmmmpol = QMMMPol()
    qmmmpol.charge_op_method = "espf"
    qmmmpol.esp_grid = (32,38)
    qmmmpol.l_max_Q = 1
    qmmmpol.setupQMMMPol(x_MM,q_MM,alpha_MM,x_QM)
    #print(q_MM)
    
    # thole parameter and bond topology needed for molecular MM environment
    qmmmpol.thole_a = 2.1304
    qmmmpol.setupBondsScalings(pairs_12=bonds)
    qmmmpol.damp_elec = True
    qmmmpol.use_exact_qmee = True
    qmmmpol.setDampingParameters(mode="cov",Z_QM=Z_QM,Z_MM=Z_MM)
    MM_atoms = ["O","H","H"]
    
    N_val = [N_val_MM[MM_atoms[B]] for B in range(0,len(MM_atoms))]
    
    #R_dens = radii .VDW[Z_MM]
    R_dens = [R_dens_MM[MM_atoms[B]] for B in range(0,len(MM_atoms))]
    qmmmpol.single_gto_exrep = False
    qmmmpol.add_exrep = True
    qmmmpol.exrep_scal = 10.0
    #qmmmpol.exrep_cut=5.0

    qmmmpol.setupExRepParams(N_val,R_dens)
    
    # set-up the pyscf mol and scf objects
    atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))]
    mol = gto.M(atom=atom_info, unit="Bohr", basis=basis_set, charge=charge_QM, spin=spin_QM)
    mf = dft.RKS(mol)
    mf.xc = xc_name
    # set-up the QM/MM calculation
    #mf,V = qmmmpol.setupQMEEPySCF(mf,use_exact_emb=False,return_V=True)
    mf = qmmmpol.setupDRFPySCF(mf)
    # get energy
    mf.kernel(dm=dm) 
    energies[n,0] = mf.energy_tot()
    mf.dip_moment(unit="AU")
    #dm = mf.make_rdm1()
    #print("Interaction energy = ",np.einsum('ij,ji',dm,V),"Hartree")

    mytd = tddft.TDA(mf)
    mytd.singlet = True
    mytd.nstates = nstates
    #mytd.max_cyc = 1000
    mytd = qmmmpol.setupDRFPySCFTD(mytd)
    
    #mytd.kernel(x0=x0)
    mytd.kernel()
    mytd.analyze(verbose=1)
    energies[n,1:] = mytd.e + energies[n,0]


print("E(r)-E_ref [Hartree]")
print(energies - E_ref)


plt.plot(r_vals*data.radii.BOHR,((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
plt.plot(r_vals*data.radii.BOHR,((energies[:,1]-E_ref-0*E_refs[0])),"-",label="S${}_1$",color="tab:red")
plt.plot(r_vals*data.radii.BOHR,((energies[:,2]-E_ref-0*E_refs[1])),"--",label="S${}_2$",color="tab:orange")

plt.ylabel("Energy [Hartrees]")
plt.xlabel("Separation [Angstrom]")

plt.legend()
plt.show()

#abs = lambda  x : x * 27.211399
abs = lambda  x : x * 1
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,1]-E_ref-E_refs[0])),"-",label="S${}_1$",color="tab:red")
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,2]-E_ref-E_refs[1])),"--",label="S${}_2$",color="tab:orange")

plt.ylabel("Energy [Hartree]")
plt.xlabel("Separation [Angstrom]")
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(-0.3,0.2)
plt.legend()
plt.show()



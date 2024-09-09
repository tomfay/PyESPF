

import sys,os
sys.path.append('../../qmmmpol/')

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
N_r = 30

r_vals = np.linspace(5.,1.5,num=N_r) / radii.BOHR
nstates = 5
energies = np.zeros((r_vals.shape[0],nstates+1))
energies_ex = np.zeros((r_vals.shape[0],nstates+1))

# basic info for the QM/MM system  (all coordinates in Bohr)
charge_QM = 0
spin_QM = 0
basis_set = "def2-TZVP"
mol = gto.M(atom="acrolein-pbe0-aligned.xyz",charge=charge_QM,spin=spin_QM,basis=basis_set)
Z_QM = mol.atom_charges()
QM_atoms = mol.elements
ind_O = list(QM_atoms).index("O")
x_QM = mol.atom_coords().T
x_O = x_QM[:,ind_O]
x_QM = x_QM - x_O.reshape((3,1))
basis_set = "def2-TZVP"
#basis_set = "aug-cc-pVDZ"
#basis_set = "cc-pVTZ"


# the MM water atom
q_O = -0.6690
q_H = -0.5*q_O
alpha_O = 9.7180 
alpha_H = 1e-10
alpha_O = 5.7494  
alpha_H = 2.7929 
#alpha_O = 1e-15
#alpha_H = 1e-15

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
mf = scf.RHF(mol)
mf = scf.RKS(mol)
xc_name = "HYB_GGA_XC_LRC_WPBEH"
mf.xc = xc_name
mf.tol_conv = 1e-12
mf.kernel()

dm = mf.make_rdm1()
print("Reference energy:")
E_ref = mf.energy_tot()


mytd = tddft.TDA(mf)
mytd = tddft.TDA(mf)
mytd.singlet = True
mytd.nstates = nstates
mytd.kernel()
mytd.analyze(verbose=4)
xy = mytd.xy
#x0 = np.array([np.hstack( (xy[n][0].flatten()) ) for n in range(0,nstates)]) 
E_refs = mytd.e 
#x = np.array([xy[n][0] for n in range(0,nstates)] )
#dm_elec = np.einsum('Aia,Aib->Aab',x,x)
#dm_hole = np.einsum('Aia,Aja->Aij',x,x)
#ddm = np.zeros((nstates,mol.nao,mol.nao))
#nocc = dm_hole.shape[1]
#nvir = dm_elec.shape[1]
#for n in range(0,nstates):
#    ddm[n,0:nocc,0:nocc] = -dm_hole[n,:,:]
#    ddm[n,nocc:,nocc:] = +dm_elec[n,:,:]
#    ddm[n,:,:] = np.sqrt(4.0)*mf.mo_coeff.dot(ddm[n,:,:].dot(mf.mo_coeff.T)) 
#mu_op = -mol.intor('cint1e_r_sph',comp=3) 
#
#dmu_exc = np.einsum('aij,Aji->Aa',mu_op,ddm)
#mu_MM = np.einsum('aA,A',x_MM_ref,q_MM)
#mu_gs = mf.dip_moment(unit="AU")
#mu_exc = dmu_exc + mu_gs.reshape((1,3))
#mu_states = np.vstack((mu_gs.reshape((1,3)),mu_exc))
#print(mu_MM)
#print(mu_states)

R_dens_MM = {"O":1.58925/radii.BOHR ,"H":1.52/radii.BOHR} # H 1.54 neutral ~1.0 cation O original 1.71
R_dens_MM = {"O":1.65/radii.BOHR ,"H":1.52/radii.BOHR} # H 1.54 neutral ~1.0 cation O original 1.71
R_dens_MM = {"O":1.6/radii.BOHR ,"H":1.54/radii.BOHR} # H 1.54 neutral ~1.0 cation O original 1.71
R_dens_MM = {"O":1.71/radii.BOHR ,"H":1.54/radii.BOHR} # H 1.54 neutral ~1.0 cation O original 1.71
#R_dens_MM = {"O":(1.71*(1-np.abs(q_O))+2.03*np.abs(q_O))/radii.BOHR ,"H":1.54*(1-np.abs(q_H))/radii.BOHR} # H 1.54 neutral ~1.0 cation O original 1.71

N_val_MM = {"O":(4.0-q_O) ,"H":(1.0-q_H)}
#N_val_MM = {"O":(4.0) ,"H":(1.0)}

print("QMMM-EE calculations:")
#for n,r in enumerate(r_vals):
#
#    # set-up separation for MM Xe atom
#    x_MM = (np.array([[r,0.,0.]])).T + x_MM_ref
#    qmmmpol = QMMMPol()
#    qmmmpol.charge_op_method = "esp"
#    #qmmmpol.charge_op_method = "mulliken"
#    #qmmmpol.grid_method = "cartesian"
#    #qmmmpol.box_pad = 10.0
#    #qmmmpol.N_cart = 10
#    qmmmpol.esp_grid = (32,38)
#    qmmmpol.l_max_Q = 1
#    qmmmpol.setupQMMMPol(x_MM,q_MM,alpha_MM,x_QM)
#    #qmmmpol.damp_elec = True
#    Z_QM = mol.atom_charges()
#    Z_MM = [8,1,1]
#    #print(q_MM)
#    qmmmpol.thole_a = 2.1304
#    qmmmpol.setupBondsScalings(pairs_12=bonds)
#    qmmmpol.damp_elec = True
#    qmmmpol.use_exact_qmee = False
#    qmmmpol.setDampingParameters(mode="cov",Z_QM=Z_QM,Z_MM=Z_MM)
#    MM_atoms = ["O","H","H"]
#    
#    N_val = [N_val_MM[MM_atoms[B]] for B in range(0,len(MM_atoms))]
#    
#    #R_dens = radii .VDW[Z_MM]
#    R_dens = [R_dens_MM[MM_atoms[B]] for B in range(0,len(MM_atoms))]
#    qmmmpol.single_gto_exrep = False
#    qmmmpol.add_exrep = True
#    qmmmpol.exrep_scal = 2.0
#    #qmmmpol.exrep_cut=5.0
#
#    qmmmpol.setupExRepParams(N_val,R_dens)
#    
#    # set-up the pyscf mol and scf objects
#    atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))]
#    mol = gto.M(atom=atom_info, unit="Bohr", basis=basis_set, charge=charge_QM, spin=spin_QM)
#    mf = dft.RKS(mol)
#    mf.xc = xc_name
#    # set-up the QM/MM calculation
#    #mf,V = qmmmpol.setupQMEEPySCF(mf,use_exact_emb=False,return_V=True)
#    mf = qmmmpol.setupDRFPySCF(mf)
#    # get energy
#    mf.kernel(dm=dm) 
#    energies[n,0] = mf.energy_tot()
#    mf.dip_moment(unit="AU")
#    #dm = mf.make_rdm1()
#    #print("Interaction energy = ",np.einsum('ij,ji',dm,V),"Hartree")
#
#    mytd = tddft.TDA(mf)
#    #mytd = tddft.TDA(qmmmpol.mf_copy)
#    mytd.singlet = True
#    mytd.nstates = nstates
#    #mytd.max_cyc = 1000
#    mytd = qmmmpol.setupDRFPySCFTD(mytd)
#    
#    #mytd.kernel(x0=x0)
#    mytd.kernel()
#    mytd.analyze(verbose=1)
#    energies[n,1:] = mytd.e + energies[n,0]
    
for n,r in enumerate(r_vals):

    # set-up separation for MM Xe atom
    x_MM = (np.array([[r,0.,0.]])).T + x_MM_ref
    qmmmpol = QMMMPol()
    qmmmpol.charge_op_method = "esp"
    #qmmmpol.charge_op_method = "mulliken"
    #qmmmpol.grid_method = "cartesian"
    #qmmmpol.box_pad = 10.0
    #qmmmpol.N_cart = 10
    qmmmpol.esp_grid = (32,38)
    qmmmpol.l_max_Q = 1
    qmmmpol.setupQMMMPol(x_MM,q_MM,alpha_MM,x_QM)
    #qmmmpol.damp_elec = True
    Z_QM = mol.atom_charges()
    Z_MM = [8,1,1]
    #print(q_MM)
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
    #mytd = tddft.TDA(qmmmpol.mf_copy)
    mytd.singlet = True
    mytd.nstates = nstates
    #mytd.max_cyc = 1000
    mytd = qmmmpol.setupDRFPySCFTD(mytd)
    
    #mytd.kernel(x0=x0)
    mytd.kernel()
    mytd.analyze(verbose=1)
    energies[n,1:] = mytd.e + energies[n,0]

    

atom_info = [[QM_atoms[A],x_QM[:,A]] for A in range(0,len(QM_atoms))] 
r = 3.0/radii.BOHR
x_MM = (np.array([[r,0.,0.]])).T + x_MM_ref
atom_info = atom_info +  [[MM_atoms[A],x_MM[:,A]] for A in range(0,len(MM_atoms))] 
mol_comb = gto.M(atom=atom_info, unit="Bohr")
mol_comb.tofile("acrolein_h2o.xyz",format="xyz")
sing = ""
if qmmmpol.use_mf_pol:
    mf = "mf"
else:
    mf = ""
mf = mf + ""
if qmmmpol.l_max_Q == 1:
    np.savetxt("data/drf/acroleinh2o/"+xc_name+basis_set+sing+mf+".csv",(energies-E_ref),delimiter=",",newline="\n")
    np.savetxt("data/drf/acroleinh2o/ref"+xc_name+basis_set+sing+mf+".csv",(E_refs-0*E_ref),delimiter=",",newline="\n")
    #np.savetxt("data/drf/hfhe/ref"+xc_name+basis_set+sing+mf+".csv",(E_refs-0*E_ref),delimiter=",",newline="\n")
elif qmmmpol.l_max_Q == 0:
    np.savetxt("data/drf/acroleinh2o/"+xc_name+basis_set+sing+mf+"-nodip.csv",(energies-E_ref),delimiter=",",newline="\n")
    np.savetxt("data/drf/acroleinh2o/ref"+xc_name+basis_set+sing+mf+"-nodip.csv",(E_refs-0*E_ref),delimiter=",",newline="\n")
    #np.savetxt("data/drf/hfhe/ref"+xc_name+basis_set+sing+mf+"-nodip.csv",(E_refs-0*E_ref),delimiter=",",newline="\n")

#mf = scf.RHF(mol)
#mf.tol_conv = 1e-10
#mf.kernel()
#print("Reference energy:")
#E_ref = mf.energy_tot()
#mf.dip_moment(unit="AU")
print("E(r)-E_ref [Hartree]")
print(energies - E_ref)
#U_rep = 0*U_rep * (1 - qmmmpol.add_rep1e - qmmmpol.add_exrep)
#Es_Q = np.array(Es_Q)
#Es_dip = np.array(Es_dip)
#print(Es_Q)
#print(Es_dip)

# plot calculated energies vs analytic energies
m = 4 
alpha_para = 6.42021 # alpha_para for H2 RHF/cc-pVDZ 
alpha_perp = 1.16480 # alpha_perp for H2 RHF/cc-pVDZ 
#U_rep = 0*(1 - qmmmpol.add_rep1e)*U_rep 
r3_inv = 1./((r_vals+0.5*r_OH*np.sin(theta_HOH/2.))**3)
#U = -2.0*np.einsum('na,a,m->nm',mu_states,mu_MM,r3_inv)
#print(U[0,:]/(energies[:,0]-E_ref))
#print(U[1,:]/(energies[:,1]-E_ref-E_refs[0]))

plt.plot(r_vals*data.radii.BOHR,((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
plt.plot(r_vals*data.radii.BOHR,((energies[:,1]-E_ref-0*E_refs[0])),"-",label="S${}_1$",color="tab:red")
plt.plot(r_vals*data.radii.BOHR,((energies[:,2]-E_ref-0*E_refs[1])),"--",label="S${}_2$",color="tab:orange")
plt.plot(r_vals*data.radii.BOHR,((energies[:,3]-E_ref-0*E_refs[2])),"-",label="S${}_3$",color="tab:green")

#plt.plot(r_vals*data.radii.BOHR,((energies_ex[:,0]-E_ref)),":",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,((energies_ex[:,1]-E_ref-0*E_refs[0])),":",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,((energies_ex[:,2]-E_ref-0*E_refs[1])),":",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,((energies_ex[:,3]-E_ref-0*E_refs[2])),":",label="S${}_3$",color="tab:green")

#plt.plot(r_vals*data.radii.BOHR,(U[0,:]),":",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,((U[1,:]+E_refs[0])),":",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,((U[2,:]+E_refs[1])),":",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,((U[3,:]+E_refs[2])),":",label="S${}_3$",color="tab:green")

#plt.plot(r_vals*data.radii.BOHR,(energies-E_ref),"--",label="Dipole (no rep.)",color="tab:blue")
#plt.plot(r_vals*data.radii.BOHR,(energies_l0-E_ref)+U_rep,"-",label="Charge",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,(energies_l0-E_ref),"--",label="Charge  (no rep.)",color="tab:red")
#C_6_fit = -(energies[-1]-E_ref)*(r_vals[-1]**m)
#C_6_fit_l0 = -(energies_l0[-1]-E_ref)*(r_vals[-1]**m)
#print("Fitted C_m (charge+dipole):",C_6_fit)
#print("Fitted C_m (charge):",C_6_fit_l0)
#V_r_analytical = -C_6_fit/(r_vals**m)
#plt.plot(r_vals,-V_r_analytical,label = "$ - C_6/r^6$")


#plt.plot(ccsd_data[:,0] , ccsd_data[:,1]-0*ccsd_data[0,1],'k.',label="CCSD")
#plt.plot(ccsd_data[:,0] , ccsdpt_data[:,1]-0*ccsdpt_data[0,1],'k+',label="CCSD(T)")

#plt.plot(r_vals,np.abs(energies-E_ref),"ko",label="RHF DRF")
#plt.plot(r_vals,np.abs(Es_Q),"-",label="charge")
#plt.plot(r_vals,np.abs(Es_dip),"--",label="dipole")
plt.ylabel("Energy [Hartrees]")
plt.xlabel("Separation [Angstrom]")
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.show()

abs = lambda  x : x * 27.211399
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,1]-E_ref-E_refs[0])),"-",label="S${}_1$",color="tab:red")
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,2]-E_ref-E_refs[1])),"--",label="S${}_2$",color="tab:orange")
plt.plot(r_vals*data.radii.BOHR,abs((energies[:,3]-E_ref-E_refs[2])),"-",label="S${}_3$",color="tab:green")

#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,0]-E_ref)),":",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,1]-E_ref-E_refs[0])),":",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,2]-E_ref-E_refs[1])),":",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,3]-E_ref-E_refs[2])),":",label="S${}_3$",color="tab:green")

#plt.plot(r_vals*data.radii.BOHR,(U[0,:]),":",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,((U[1,:]+0*E_refs[0])),":",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,((U[2,:]+0*E_refs[1])),":",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,((U[3,:]+0*E_refs[2])),":",label="S${}_3$",color="tab:green")
plt.ylabel("Energy [eV]")
plt.xlabel("Separation [Angstrom]")
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(-0.3,0.2)
plt.legend()
plt.show()

#abs = lambda  x : x * 27.211399
##plt.plot(r_vals*data.radii.BOHR,abs((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,abs((energies[:,1]-energies[:,0]-E_refs[0])),"-",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,abs((energies[:,2]-energies[:,0]-E_refs[1])),"--",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,abs((energies[:,3]-energies[:,0]-E_refs[2])),"-",label="S${}_3$",color="tab:green")
#
##plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,0]-E_ref)),":",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,1]-energies_ex[:,0]-E_refs[0])),":",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,2]-energies_ex[:,0]-E_refs[1])),":",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,abs((energies_ex[:,3]-energies_ex[:,0]-E_refs[2])),":",label="S${}_3$",color="tab:green")
#
##plt.plot(r_vals*data.radii.BOHR,(U[0,:]),":",label="S${}_0$",color="black")
##plt.plot(r_vals*data.radii.BOHR,((U[1,:]+0*E_refs[0])),":",label="S${}_1$",color="tab:red")
##plt.plot(r_vals*data.radii.BOHR,((U[2,:]+0*E_refs[1])),":",label="S${}_2$",color="tab:orange")
##plt.plot(r_vals*data.radii.BOHR,((U[3,:]+0*E_refs[2])),":",label="S${}_3$",color="tab:green")
#plt.ylabel("$\Delta$ Excitation Energy [eV]")
#plt.xlabel("Separation [Angstrom]")
##plt.xscale('log')
##plt.yscale('log')
#plt.legend()
#plt.show()
#
#plt.plot(r_vals*data.radii.BOHR,100.*((energies[:,0]-energies_ex[:,0]))/(energies_ex[:,0]-E_ref),"-",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,100.*((energies[:,1]-energies_ex[:,1]))/((energies_ex[:,1]-E_ref-E_refs[0])),"-",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,100.*((energies[:,2]-energies_ex[:,2]))/((energies_ex[:,2]-E_ref-E_refs[1])),"--",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,100.*((energies[:,3]-energies_ex[:,3]))/((energies_ex[:,3]-E_ref-E_refs[2])),"-",label="S${}_3$",color="tab:green")
#plt.ylabel("Percentage Energy Error [Hartrees]")
#plt.xlabel("Separation [Angstrom]")
#plt.legend()
#plt.show()
#
#plt.plot(r_vals*data.radii.BOHR,((energies[:,0]-energies_ex[:,0])),"-",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,((energies[:,1]-energies_ex[:,1])),"-",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,((energies[:,2]-energies_ex[:,2])),"--",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,((energies[:,3]-energies_ex[:,3])),"-",label="S${}_3$",color="tab:green")
#plt.ylabel("Energy Error [Hartrees]")
#plt.xlabel("Separation [Angstrom]")
#plt.legend()
#plt.show()

#plt.scatter(U[0,:],(energies[:,0]-E_ref))
#plt.scatter(U[1,:],(energies[:,1]-E_ref-E_refs[0]))
#plt.show()

#data = np.hstack( ((energies-E_ref),(energies_l0-E_ref),U_rep) )



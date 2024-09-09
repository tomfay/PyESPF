

import sys,os
sys.path.append('../../qmmmpol/')

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
This example calculates the energy for a non-polarisable X- ion and Li+ as a function of Li-X separation
using RHF for the Li+ with the DRF method for the interaction between the QM and MM regions
"""

np.set_printoptions(linewidth=125)

ccsd_data = np.loadtxt("./data/ref/hfhe/ccsd.dat")
ccsdpt_data = np.loadtxt("./data/ref/hfhe/ccsdpt.dat")

#ccsd_data = np.loadtxt("./data/ref/h2na/ccsd.dat",skiprows=0)
#ccsdpt_data = np.loadtxt("./data/ref/h2na/ccsdpt.dat",skiprows=0)
#plt.plot(ccsd_data[:,0] , ccsd_data[:,1]-ccsd_data[0,1],'k--',label="CCSD")
#plt.plot(ccsd_data[:,0] , ccsdpt_data[:,1]-ccsdpt_data[0,1],'k-',label="CCSD(T)")
#ccsd_data[:,1] = ccsd_data[:,1]-np.loadtxt("./data/ref/hfhe/ccsd.dat",skiprows=0)[:,1]  -np.loadtxt("./data/ref/h2na/ccsdb.dat",skiprows=0)[:,1]
#ccsdpt_data[:,1] = ccsdpt_data[:,1]-np.loadtxt("./data/ref/hfhe/ccsdpt.dat",skiprows=0)[:,1] -np.loadtxt("./data/ref/h2na/ccsdptb.dat",skiprows=0)[:,1]
plt.plot(ccsd_data[:,0] , ccsd_data[:,1]-ccsd_data[0,1],'k.',label="CCSD")
plt.plot(ccsd_data[:,0] , ccsdpt_data[:,1]-ccsdpt_data[0,1],'k+',label="CCSD(T)")
plt.legend()
plt.show()

#  separations in Bohr
N_r = 101

r_vals = np.linspace(10.,2.0,num=N_r) / radii.BOHR
nstates = 6
singlet = True
energies = np.zeros((r_vals.shape[0],nstates+1))
energies_l0 = np.zeros(r_vals.shape)
U_rep = np.zeros(r_vals.shape)
#Es_Q = [0. for n in range(0,N_r)]
#Es_dip = [0. for n in range(0,N_r)]
alpha_Xe = 1e-16*27.3
alpha_Xe = 0.17280
alpha_Xe = 1e-16

# basic info for the QM/MM system  (all coordinates in Bohr)
QM_atoms = ["H","F"]
r_bond = 0.910 / radii.BOHR
x_QM = r_bond*np.array([[-1.0,0.,0.],[0.0,0.,0.0]]).T
#QM_atoms = ["F"]
#x_QM = np.array([[-0.0,0.,0.]]).T
#x_QM = np.array([[0,-0.7,0.],[0,0.7,0.]]).T
#QM_atoms = ["He"]
#x_QM = np.array([[0.,0.,0.]]).T
#QM_atoms = ["Li","F"]
#x_QM = np.array([[-1.5,0.,0.],[1.5,0.,0.]]).T
#x_QM = np.array([[0.,-1.5,0.],[0.,1.5,0.]]).T

#QM_atoms = ["Ne","Ne"]
#x_QM = np.array([[0.,0.,0.],[0.,1e3,0.]]).T

x_MM = (np.array([[0.,0.,0.]])).T
alpha_MM = np.array([alpha_Xe]) 

#x_MM = (np.array([[0.,0.,0.],[0.,1e4,0.]])).T
#alpha_MM = np.array([alpha_Xe,alpha_Xe]) 

q_MM = np.array([0.0e0])
charge_QM = 0
spin_QM = 0
basis_set = "cc-pVQZ-F12"
#cc-pVQZ-F12
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
#basis_set = "aug-pc-2"
#basis_set = "6-31+G*"
#basis_set = {"H1":"6-31G","H2":"6-31G**"}
#basis_set = "aug-cc-pVQZ"
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
mf = scf.RHF(mol)
mf = scf.RKS(mol)
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


mytd = tddft.TDA(mf)
mytd.singlet = singlet
mytd.nstates = nstates
mytd.kernel()
mytd.analyze(verbose=4)

xy = mytd.xy
x0 = np.array([np.hstack( (xy[n][0].flatten()) ) for n in range(0,nstates)]) 

E_refs = mytd.e 



kJ_mol = 0.000380879803
nanometre = 1./5.29177210544e-2
#epsilon_QM = kJ_mol*np.array([0.276144,0.125520,0.125520,0.125520,0.125520]) # OPLS-AA
#sigma_QM = nanometre*np.array([0.35,0.25,0.25,0.25,0.25])
epsilon_QM = kJ_mol*np.array([0.4577296,0.0656888,0.0656888,0.0656888,0.0656888]) # DOI 10.1021/ja00043a027
sigma_QM = 2.0*nanometre*np.array([0.19082,0.14872,0.14872,0.14872,0.14872])

#sigma_MM = nanometre*np.array([0.2782])
epsilon_MM = kJ_mol*np.array([1.04512]) # doi 10.1063/1.479848
sigma_MM = nanometre*np.array([0.3345])






#Na
#q_MM = np.array([1.0e0])
#alpha_MM = np.array([1e-16]) 
#qmmmpol.damp_elec = True
#Z_QM = [9,1] 
#Z_MM = [11]
#qmmmpol.setupRepulsion(Z_QM,Z_MM)
#qmmmpol.setupLJRep(sigma_QM,epsilon_QM,sigma_MM,epsilon_MM,n_LJ=12,xi=12.0,Z_QM=Z_QM,Z_MM=Z_MM)
#qmmmpol.add_rep1e = False
#N_val = np.array([6.0])
#R_dens = np.array([1.62115])/radii.BOHR
##R_dens = radii.VDW[Z_MM]
#qmmmpol.single_gto_exrep = False
#qmmmpol.setupExRepParams(N_val,R_dens)
#print(qmmmpol.beta_MM)
#print(qmmmpol.c_MM)
#qmmmpol.add_exrep = True

# Ar/Ne
qmmmpol.damp_elec = True
Z_QM = [9,1] 

q_MM = np.array([0.0e0])

alpha_MM = np.array([1.12648]) # cc-pVTZ-F12
alpha_MM = np.array([1.20409]) # cc-pVQZ-F12
Z_MM = [2]
R_dens = np.array([1.34])/radii.BOHR
N_val = np.array([2.0])

#alpha_MM = np.array([10.0]) 
#Z_MM = [18]
#R_dens = np.array([1.97])/radii.BOHR
#N_val = np.array([2.0])
qmmmpol.setupRepulsion(Z_QM,Z_MM)
qmmmpol.setupLJRep(sigma_QM,epsilon_QM,sigma_MM,epsilon_MM,n_LJ=12,xi=12.0,Z_QM=Z_QM,Z_MM=Z_MM)
qmmmpol.add_rep1e = False

#R_dens = radii.VDW[Z_MM]
qmmmpol.single_gto_exrep = False
qmmmpol.setupExRepParams(N_val,R_dens)
print(qmmmpol.beta_MM)
print(qmmmpol.c_MM)
qmmmpol.add_exrep = True




print("DRF calculations:")
for n,r in enumerate(r_vals):

    # set-up separation for MM Xe atom
    #x_MM = (np.array([[r,0.,0.],[r,1e4,0]])).T
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
    qmmmpol.use_mf_pol = True
    qmmmpol.l_max_Q = 1
    qmmmpol.charge_op_method = "esp"
    qmmmpol.esp_grid = (32,38)
    qmmmpol.use_exact_qmee = False
    qmmmpol.damp_elec = True
    qmmmpol.n_damp = 6.0
    qmmmpol.setDampingParameters(mode="cov",Z_QM=Z_QM,Z_MM=Z_MM)
    qmmmpol.add_exrep = True
    qmmmpol.single_gto_exrep = False
    qmmmpol.exrep_scal = 10.0
    qmmmpol.setupExRepParams(N_val,R_dens)
    qmmmpol.scal_N_val = 0.5 
    mf = qmmmpol.setupDRFPySCF(mf)
    #print(mf.get_hcore())
    mf.tol_conv = 1e-12
    # get energy
    mf.kernel(dm=dm) 
    energies[n,0] = mf.energy_tot()
    mf.dip_moment(unit="AU")
    dm = mf.make_rdm1()

    
    #U_rep[n] = qmmmpol.calculateRepulsion()
    #U_rep[n] = qmmmpol.calculateBuckRepulsion()
    #U_rep[n] = qmmmpol.calculateLJRepulsion()
    #mytd = tddft.TDA(mf)
    #qmmmpol.mf_copy.kernel()
    mytd = tddft.TDA(mf)
    #mytd = tddft.TDA(qmmmpol.mf_copy)
    mytd.singlet = singlet
    mytd.nstates = nstates
    #mytd.max_cyc = 1000
    
    mytd = qmmmpol.setupDRFPySCFTD(mytd)
    
    mytd.kernel(x0=x0)
    #mytd.kernel()
    #mytd.analyze(verbose=1)
    xy = mytd.xy
    x0 = np.array( [np.hstack( (xy[n][0].flatten()) ) for n in range(0,nstates)] ) 
    #x0_td = np.array([np.hstack( (np.array([xy[n][0].flatten(),np.zeros(xy[n][0].flatten().shape)) ) for n in range(0,nstates)]) 
    x0_td = np.array( [np.hstack( (xy[n][0].flatten(),np.zeros(xy[n][0].flatten().shape)) ) for n in range(0,nstates)] ) 
    
    #mytd = tddft.TDDFT(mf)
    #mytd.singlet = singlet
    #mytd.nstates = nstates
    #mytd.max_cyc = 1000
    #mytd = qmmmpol.setupDRFPySCFTD(mytd)
    #mytd.kernel(x0=x0_td)
    
    
    energies[n,1:(len(mytd.e)+1)] = mytd.e + energies[n,0]
    


    
#mf = scf.RHF(mol)
#mf.tol_conv = 1e-10
#mf.kernel()
#print("Reference energy:")
#E_ref = mf.energy_tot()
#mf.dip_moment(unit="AU")
print("E(r)-E_ref [Hartree]")
print(energies - E_ref)
U_rep = 0*U_rep * (1 - qmmmpol.add_rep1e - qmmmpol.add_exrep)
#Es_Q = np.array(Es_Q)
#Es_dip = np.array(Es_dip)
#print(Es_Q)
#print(Es_dip)

# plot calculated energies vs analytic energies
m = 4 
alpha_para = 6.42021 # alpha_para for H2 RHF/cc-pVDZ 
alpha_perp = 1.16480 # alpha_perp for H2 RHF/cc-pVDZ 
#U_rep = 0*(1 - qmmmpol.add_rep1e)*U_rep 
#basis_set = "ccpvqzf12"
#basis_set = "augccpvtz"
if singlet:
    sing = "-sing"
else:
    sing = "-trip"
if qmmmpol.use_mf_pol:
    mf = "mf"
else:
    mf = ""
if type(basis_set) == type({}):
    basis_set = "cc-pVQZ-F12"
basis_set = basis_set+"-newrep"
if qmmmpol.l_max_Q == 1:
    np.savetxt("data/drf/hfhe/"+xc_name+basis_set+sing+mf+".csv",(energies-E_ref),delimiter=",",newline="\n")
    np.savetxt("data/drf/hfhe/ref"+xc_name+basis_set+sing+mf+".csv",(E_refs-0*E_ref),delimiter=",",newline="\n")
elif qmmmpol.l_max_Q == 0:
    np.savetxt("data/drf/hfhe/"+xc_name+basis_set+sing+mf+"-nodip.csv",(energies-0*E_ref),delimiter=",",newline="\n")
    np.savetxt("data/drf/hfhe/ref"+xc_name+basis_set+sing+mf+"-nodip.csv",(E_refs-0*E_ref),delimiter=",",newline="\n")

plt.plot(r_vals*data.radii.BOHR,((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
#plt.plot(r_vals*data.radii.BOHR,-2.0*mu_sq*alpha_MM[0]/(r_vals**6),"--",label="S${}_0 (dip)$",color="black")
for n in range(0,nstates):
    plt.plot(r_vals*data.radii.BOHR,((energies[:,1+n]-E_ref-0*E_refs[n])),"-",label="S${}_1$")
#plt.plot(r_vals*data.radii.BOHR,((energies[:,1]-E_ref-0*E_refs[0])),"-",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,((energies[:,2]-E_ref-0*E_refs[1])),"--",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,((energies[:,3]-E_ref-0*E_refs[2])),"-",label="S${}_3$",color="tab:green")
plt.plot(ccsd_data[:,0] , ccsdpt_data[:,1]-ccsdpt_data[0,1],'k+',label="CCSD(T)")
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

def scal(x):
    
    return x

plt.plot(r_vals*data.radii.BOHR,scal((energies[:,0]-E_ref)),"-",label="S${}_0$",color="black")
plt.plot(r_vals*data.radii.BOHR,scal(-0.5*alpha_MM[0]*(charge_QM**2)/(r_vals**4)-(2.0*mu_sq*alpha_MM)/((r_vals)**6)+np.sqrt(mu_sq)*q_MM[0]/((r_vals)**2)),"--",label="S${}_0$ (dip)",color="black")
#plt.plot(r_vals*data.radii.BOHR,scal((energies[:,1]-E_ref-E_refs[0])),"-",label="S${}_1$",color="tab:red")
#plt.plot(r_vals*data.radii.BOHR,scal((energies[:,2]-E_ref-E_refs[1])),"--",label="S${}_2$",color="tab:orange")
#plt.plot(r_vals*data.radii.BOHR,scal((energies[:,3]-E_ref-E_refs[2])),"-",label="S${}_3$",color="tab:green")
for n in range(0,nstates):
    if n%2==0:
        style = "-" 
    else: 
        style = "--" 
    plt.plot(r_vals*data.radii.BOHR,((energies[:,1+n]-E_ref-E_refs[n])),style,label="S${}_1$")

plt.ylabel("Energy [Hartrees]")
plt.xlabel("Separation [Angstrom]")
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.show()

#data = np.hstack( ((energies-E_ref),(energies_l0-E_ref),U_rep) )



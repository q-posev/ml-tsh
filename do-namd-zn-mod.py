import os

from ase.units import Hartree,Bohr,fs,kB

import schnetpack as spk
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model

from torch import device as torch_device

from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.io import read
from pandas import read_csv

import sys
import numpy as np

from ase.calculators.demonnano import DemonNano

# Get the parent directory of SchNetPack
spk_path = os.path.abspath(os.path.join(os.path.dirname(spk.__file__), '../..'))
# number and size of MD steps from the arguments passed through command line
n_md = int(sys.argv[1])
step_md = float(sys.argv[2])  #in fs

# Load Machine Learning model (trained) and make it calculator
model_dir = '/tmpdir/evgeny/phenant-neural-net/s2-bio-T500-nose/make-model/model_force_10k'
model2_dir = '/tmpdir/evgeny/phenant-neural-net/s3-bio-T500-nose/make-model/model_force_10k'

path_s1 = os.path.join(model_dir, 'best_model')
path_s2 = os.path.join(model2_dir, 'best_model')

# flag that indicates on which state the trajectory is running
init_es = 3
do_hop = True
flag_es = init_es

hop_every = 1

# Check if a GPU is available and use a CPU otherwise
#if torch.cuda.is_available():
#    md_device = "cuda"
#else:
#    md_device = "cpu"

#md_model = torch.load(path_s1, map_location=md_device).to(md_device)
md_device = "cpu"

model = load_model(path_s1,map_location=torch_device('cpu'))
model2 = load_model(path_s2,map_location=torch_device('cpu'))

calc = SpkCalculator(model, device="cpu", energy="energy", forces="forces")
calc2 = SpkCalculator(model2, device="cpu", energy="energy", forces="forces")

# demon-nano calculator
input_arguments = {'DFTB': 'SCC LRESP EXST=3',
                   'CHARGE': '0.0',
                   'PARAM': 'PTYPE=BIO'}
demon = DemonNano(label='rundir/',input_arguments=input_arguments)

# read geometry from file and set calculator
atoms=read('geom-phen.xyz')

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, 300 *kB)
#Stationary(atoms)  # zero linear momentum
#ZeroRotation(atoms)  # zero angular momentum

velo = atoms.get_velocities()
data = read_csv('velo', delimiter='\s+', header=None, index_col=False)
data.columns = ["atom", "x", "y", "z"]
velo[:,0]=data.x
velo[:,1]=data.y
velo[:,2]=data.z

atoms.set_velocities(velo/fs)

# We want to run MD with constant energy using the VelocityVerlet algorithm. 
#print("MD step: ", step_md)
dyn = VelocityVerlet(atoms, step_md * fs)	
#, trajectory='md.traj')#, logfile='md.log')

# dynamical variables
gap_12 = np.zeros(n_md+1,dtype=np.float64)

t_step = np.zeros(n_md+1,dtype=np.float32)
t_step = np.arange(0, n_md+1)
t_step = t_step*step_md

j_md=0

tau_0 = 0.02418881
dt = step_md/tau_0

df = open("gaps_12.txt","w+")

if flag_es==3:
    atoms.set_calculator(calc2)
else:
    atoms.set_calculator(calc)

# seed random number generator
# for reproducible results
#np.random.seed(1)

do_hop = True
hop = False
count = 0
count_interpol = 0

force_up_t1 = atoms.get_forces()*Bohr/Hartree
force_up_t3 = atoms.get_forces()*Bohr/Hartree
force_up_t2 = atoms.get_forces()*Bohr/Hartree
coordinates_t3 = atoms.get_positions()
coordinates_t1 = atoms.get_positions()
coordinates = atoms.get_positions()
atoms.set_calculator(calc)
etot = atoms.get_total_energy()/Hartree
ex = atoms.get_potential_energy()/Hartree

velocities = atoms.get_velocities()
force_down_t1 = atoms.get_forces()*Bohr/Hartree
force_down_t2 = atoms.get_forces()*Bohr/Hartree
force_down_t3 = atoms.get_forces()*Bohr/Hartree
atoms.set_calculator(calc2)

masses = atoms.get_masses()
# print energies and compute S1/S2 gap
def printenergy(a=atoms):  # store a reference to atoms in the definition.

    global j_md, count, count_interpol
    global flag_es, dt
    global do_hop, hop
    global force_up_t1,coordinates_t1,force_down_t2,force_down_t1
    global force_up_t3,coordinates_t3,force_up_t2,force_down_t3
    global coordinates,velocities,etot,ex,masses   

    betta=0.0
 
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy()/Hartree #/ len(a)
    ekin = a.get_kinetic_energy()/Hartree #/ len(a)

    if j_md == count_interpol:
        force_up_t1 = a.get_forces()*Bohr/Hartree
        coordinates_t1 = a.get_positions()
        a.set_calculator(calc)
        force_down_t1 = a.get_forces()*Bohr/Hartree    
        a.set_calculator(calc2)

    if (j_md == (count_interpol+1)):
        etot = a.get_total_energy()/Hartree
        ex = a.get_potential_energy()/Hartree
        coordinates = a.get_positions()
        velocities = a.get_velocities()
        force_up_t2 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc)
        force_down_t2 = a.get_forces()*Bohr/Hartree    
        a.set_calculator(calc2)

    #print('Total energy diff: {}'.format(etot - (epot+ekin)))

    if j_md == (count_interpol+2):
        force_up_t3 = a.get_forces()*Bohr/Hartree
        coordinates_t3 = a.get_positions()
        a.set_calculator(calc)
        force_down_t3 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc2)
        #print(count_interpol)
        count_interpol = count_interpol + 1

    if flag_es==3:
        a.set_calculator(calc)
    else:
        a.set_calculator(calc2)

    emod = a.get_potential_energy()/Hartree    

    # resetting of calculators at the end of this function

    gap_12[j_md] = np.abs(epot - emod)

    p_zn = 0.0
    p_lz = 0.0
    if (gap_12[j_md-1] <= gap_12[j_md]) and (gap_12[j_md-2] >= gap_12[j_md-1]) and (j_md>1):
        #print('Possible crossing at {}'.format(j_md))
        
        #force1_x = (force_down_t3*(coordinates-coordinates_t1) - force_up_t1*(coordinates-coordinates_t3))/(coordinates_t3-coordinates_t1)
        #force2_x = (force_up_t3*(coordinates-coordinates_t1) - force_down_t1*(coordinates-coordinates_t3))/(coordinates_t3-coordinates_t1)
        sum_G = force_up_t2+force_down_t2
        dGc = (force_up_t2-force_down_t2) - (force_up_t1-force_down_t1)
        dGc /= dt
        print(dGc) 
        
        factor=gap_12[j_md-1]/(4.0*np.tensordot(dGc,velocities))        
 
        force1_x = 0.5*sum_G - factor*dGc
        force2_x = 0.5*sum_G + factor*dGc
        
        force_diff_acc = 0.0
        force_prod_acc = 0.0
        for i in range(0,len(a)):
            for j_coord in range(0,3):   
                temp0  = force2_x[i,j_coord]-force1_x[i,j_coord]     
                temp = temp0*temp0
                temp2 = force2_x[i,j_coord]*force1_x[i,j_coord]    

                force_diff_acc += temp/masses[i]
                force_prod_acc += temp2/masses[i]

        force_diff = np.sqrt(force_diff_acc)
        if force_prod_acc > 0.0:
            do_plus = True
        else:
            do_plus = False
        force_prod = np.sqrt(np.abs(force_prod_acc))
    
        a2 = 0.5*force_diff*force_prod/(np.power(2.0*gap_12[j_md-1],3))
        b2 = (etot-ex)*force_diff/(force_prod*2.0*gap_12[j_md-1])
        if (a2<0.0) or (b2<0.0) :
            print('Alert!')

        root = 0.0
        if do_plus:
            root = b2 + np.sqrt(b2*b2 + 1.0)
        else:
            root = b2 + np.sqrt(np.abs(b2*b2 - 1.0))

        p_zn = np.exp(-np.pi*0.25*np.sqrt(2.0/(a2*root)))
        
        dgap = (gap_12[j_md] - 2.0*gap_12[j_md-1] + gap_12[j_md-2])/(dt*dt)
        #dgap2 = (gap_12[j_md+1] - 2.0*gap_12[j_md] + gap_12[j_md-1])/(dt*dt)
        c_ij = np.power(gap_12[j_md-1],3)/dgap
        p_lz = 0.0
        if(dgap<0.0):
            print('alert, small d^2/dt^2',dgap)
            #p_lz = np.exp(-0.5*np.pi*np.sqrt(-c_ij)) 
        else:
            p_lz = np.exp(-0.5*np.pi*np.sqrt(c_ij)) 

        if do_hop and (not hop):
            xi = np.random.rand(1)
            if xi <= p_lz and xi > p_zn:
                print('Hop according to Landau-Zener at {}'.format(j_md-1))
                #print('Landau-Zener prob: {}'.format(float(p_lz)))
                #hop = True
            elif xi <= p_zn and xi > p_lz:
                print('Hop according to Zhu-Nakamura at {}'.format(j_md-1))
                print('Zhu-Nakamura prob: {}'.format(float(p_zn)))
                hop = True
            elif xi <= p_zn and xi <= p_lz:
                print('Hop according to LZ and ZN at {} !'.format(j_md-1))
                print('LZ and ZN probs: {0} {1}'.format(float(p_lz),float(p_zn)))
                hop = True

            betta = gap_12[j_md-1]/ekin
            # check for frustrated hop
            if betta >= 1.0:
                print("Frustrated hop",j_md, betta)
                hop = False
            
            if hop :
                if flag_es==3:
                    flag_es=2
                else:
                    flag_es=3
                
                a.set_positions(coordinates)
                #a.set_velocities(velocities)
		# velocity rescaling to conserve total energy
                a.set_velocities(np.sqrt(1.0+betta)*velocities)
                # set j_md to j_md-1 because we kind of make a step back in time 
                j_md -= 1
                count_interpol -= 1
    
    # set SPK model back to running state 
    # this already takes into account the switch if performed
    if flag_es==3:
        a.set_calculator(calc2)
    else:
    	a.set_calculator(calc)

    if j_md != 0:
        df.write('{0:0.2f} {1:0.4f} {2} {3} {4}\n'.format(t_step[j_md-1],gap_12[j_md-1]*Hartree,float(p_zn),float(p_lz),flag_es))
    
    j_md += 1

# Now run the dynamics

dyn.attach(printenergy, interval=1)
dyn.run(n_md)

k = 11
idx = np.argpartition(gap_12, k)
print('Indices of {0} smallest gaps: {1}'.format(k,idx[:k]))
#This returns the k-smallest values. Note that these may not be in sorted order.
with np.printoptions(precision=4, suppress=True):
    print('Smallest values:')
    print(gap_12[idx[:k]]*Hartree)

#print(np.argmin(gap_12),np.min(gap_12))
df.close()


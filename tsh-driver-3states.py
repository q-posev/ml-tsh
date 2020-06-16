from os import path, fsync
from sys import argv

from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model

from torch import device as torch_device

from ase.units import Hartree,Bohr,fs,kB
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

from pandas import read_csv
import numpy as np

from ase.calculators.demonnano import DemonNano

assert len(argv)>4, 'Not all arguments provided'
# number and size of MD steps from the arguments passed through command line
n_md = int(argv[1])
step_md = float(argv[2])  #in fs
# type of calculator (demonnano or schnet)
calc_type = str(argv[3])  
# type of hopping strategy (lz or zn)
hop_type = str(argv[4])  

# Load Machine Learning model (trained) and make it calculator
model_dir = '/tmpdir/evgeny/phenant-neural-net/s2-bio-T500-nose/make-model/model_force_10k'
model2_dir = '/tmpdir/evgeny/phenant-neural-net/s3-bio-T500-nose/make-model/model_force_10k'
model3_dir = '/tmpdir/evgeny/phenant-neural-net/s4-bio-T500-nose/make-model/model_force_10k'
path1 = path.join(model_dir, 'best_model')
path2 = path.join(model2_dir, 'best_model')
path3 = path.join(model3_dir, 'best_model')

do_lz = False
do_zn = False
# determine the hopping strategy
if hop_type=='lz':
    do_lz = True  
if hop_type=='zn':
    do_zn = True

assert do_lz or do_zn, 'Please enter a hopping type: lz or zn'
do_both = do_lz and do_zn

# flag that indicates on which state the trajectory is running
flag_es = 3

do_hop = True
hop_every = 1

# SchNet calculator
model = load_model(path1,map_location=torch_device('cpu'))
model2 = load_model(path2,map_location=torch_device('cpu'))
model3 = load_model(path3,map_location=torch_device('cpu'))

# demon-nano calculator
input_arguments = {'DFTB': 'SCC LRESP EXST=2',
                   'CHARGE': '0.0',
                   'PARAM': 'PTYPE=BIO'}
input_arguments2 = {'DFTB': 'SCC LRESP EXST=3',
                   'CHARGE': '0.0',
                   'PARAM': 'PTYPE=BIO'}
input_arguments3 = {'DFTB': 'SCC LRESP EXST=4',
                   'CHARGE': '0.0',
                   'PARAM': 'PTYPE=BIO'}

if (calc_type=="demonnano"):
    calc = DemonNano(label='rundir/',input_arguments=input_arguments)
    calc2 = DemonNano(label='rundir/',input_arguments=input_arguments2)
    calc3 = DemonNano(label='rundir/',input_arguments=input_arguments3)
if (calc_type=="schnet"):
    calc = SpkCalculator(model, device="cpu", energy="energy", forces="forces")
    calc2 = SpkCalculator(model2, device="cpu", energy="energy", forces="forces")
    calc3 = SpkCalculator(model3, device="cpu", energy="energy", forces="forces")


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
gap_mid_down = np.zeros(n_md,dtype=np.float64)
gap_mid_up = np.zeros(n_md,dtype=np.float64)

t_step = np.zeros(n_md,dtype=np.float32)
t_step = np.arange(0, n_md)
t_step = t_step*step_md

j_md=0

tau_0 = 0.02418881
dt = step_md/tau_0

df = open("gaps.txt","w+")

if flag_es==3:
    atoms.set_calculator(calc2)
else:
    print("Are you sure?")

# seed random number generator for reproducible results
#np.random.seed(1)

do_hop = True
hop = False

count_interpol = 0

force_up_t1 = atoms.get_forces()*Bohr/Hartree
force_up_t3 = atoms.get_forces()*Bohr/Hartree
force_up_t2 = atoms.get_forces()*Bohr/Hartree
force_mid_t1 = atoms.get_forces()*Bohr/Hartree
force_mid_t3 = atoms.get_forces()*Bohr/Hartree
force_mid_t2 = atoms.get_forces()*Bohr/Hartree
force_down_t1 = atoms.get_forces()*Bohr/Hartree
force_down_t2 = atoms.get_forces()*Bohr/Hartree
force_down_t3 = atoms.get_forces()*Bohr/Hartree

coordinates_t3 = atoms.get_positions()
coordinates_t1 = atoms.get_positions()
coordinates = atoms.get_positions()
velocities = atoms.get_velocities()

etot = atoms.get_total_energy()/Hartree
ex = atoms.get_potential_energy()/Hartree
masses = atoms.get_masses()

def tsh(a=atoms):  # store a reference to atoms in the definition.
    global j_md, count_interpol
    global flag_es, dt
    global do_hop, hop
    global force_up_t1,coordinates_t1,force_down_t2,force_down_t1
    global force_up_t3,coordinates_t3,force_up_t2,force_down_t3
    global coordinates,velocities,etot,ex,masses   
    global force_mid_t1,force_mid_t3,force_mid_t2
    global gaps_mid_down, gaps_mid_up
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy()/Hartree #/ len(a)
    ekin = a.get_kinetic_energy()/Hartree #/ len(a)

    if j_md == count_interpol:
        a.set_calculator(calc3)
        force_up_t1 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc)
        force_down_t1 = a.get_forces()*Bohr/Hartree    
        a.set_calculator(calc2)
        force_mid_t1 = a.get_forces()*Bohr/Hartree    
        coordinates_t1 = a.get_positions()
    if j_md == (count_interpol+1):
        etot = a.get_total_energy()/Hartree
        ex = a.get_potential_energy()/Hartree
        coordinates = a.get_positions()
        velocities = a.get_velocities()
        a.set_calculator(calc3)
        force_up_t2 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc)
        force_down_t2 = a.get_forces()*Bohr/Hartree    
        a.set_calculator(calc2)
        force_mid_t2 = a.get_forces()*Bohr/Hartree    
    if j_md == (count_interpol+2):
        a.set_calculator(calc3)
        force_up_t3 = a.get_forces()*Bohr/Hartree
        coordinates_t3 = a.get_positions()
        a.set_calculator(calc)
        force_down_t3 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc2)
        force_mid_t3 = a.get_forces()*Bohr/Hartree    
        count_interpol = count_interpol + 1

    def check_hop(gap,forces_upper_t2,forces_upper_t1,forces_lower_t2,forces_lower_t1,target_state):
        global dt, j_md
        global coordinates, velocities
        global hop, do_hop
        global etot, ex, ekin

        p_zn = 0.0
        p_lz = 0.0
        # check for the local minimum of the energy gap between 2 states
        if (gap[j_md-1] <= gap[j_md]) and (gap[j_md-2] >= gap[j_md-1]):
            #print('Possible crossing at {}'.format(j_md))
            
            # Landau-Zener part
            dgap = (gap[j_md] - 2.0*gap[j_md-1] + gap[j_md-2])/(dt*dt)
            #dgap2 = (gap_12[j_md+1] - 2.0*gap_12[j_md] + gap_12[j_md-1])/(dt*dt)
            c_ij = np.power(gap[j_md-1],3)/dgap
            # compute Landau-Zener probability
            if(dgap<0.0):
                print('alert, small d^2/dt^2',dgap)
            else:
                p_lz = np.exp(-0.5*np.pi*np.sqrt(c_ij)) 
 
            # Zhu-Nakamura part
 
            # compute diabatic gradients according to Yu et al. 2014 (PCCP)
            #force1_x = (force_down_t3*(coordinates-coordinates_t1) - force_up_t1*(coordinates-coordinates_t3))/(coordinates_t3-coordinates_t1)
            #force2_x = (force_up_t3*(coordinates-coordinates_t1) - force_down_t1*(coordinates-coordinates_t3))/(coordinates_t3-coordinates_t1)
            # compute diabatic gradients according to Hanasaki et al. 2018 (JCP)
            sum_G = force_upper_t2+force_lower_t2
            dGc = (force_upper_t2-force_lower_t2) - (force_upper_t1-force_lower_t1)
            dGc /= dt
           
            dGxVelo = np.tensordot(dGc,velocities)
            if (dGxVelo < 0.0):
                print('negative product, use BL')
                factor=0.5*np.sqrt(gap_12[j_md-1]/dgap)
            else:
                factor=0.5*np.sqrt(gap_12[j_md-1]/dGxVelo)
  
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
     
            a2 = 0.5*force_diff*force_prod/(np.power(2.0*gap[j_md-1],3))
            b2 = (etot-ex)*force_diff/(force_prod*2.0*gap[j_md-1])
            if (a2<0.0) or (b2<0.0) :
                print('Alert!')
            root = 0.0
            if do_plus:
                root = b2 + np.sqrt(b2*b2 + 1.0)
            else:
                root = b2 + np.sqrt(np.abs(b2*b2 - 1.0)) 
            # compute Zhu-Nakamura probability
            p_zn = np.exp(-np.pi*0.25*np.sqrt(2.0/(a2*root)))
            
            # comparison with a random number
            if do_hop and (not hop):
                xi = np.random.rand(1)
                if xi <= p_lz and do_lz:
                    print('Attempted hop according to Landau-Zener at {}'.format(j_md-1),flush=True)
                    print('Landau-Zener prob: {}'.format(float(p_lz)),flush=True)
                    hop = True
                elif xi <= p_zn and do_zn:
                    print('Attempted hop according to Zhu-Nakamura at {}'.format(j_md-1),flush=True)
                    print('Zhu-Nakamura prob: {}'.format(float(p_zn)),flush=True)
                    hop = True
                elif xi <= p_zn and xi <= p_lz and do_both:
                    print('Attempted hop according to LZ and ZN at {} !'.format(j_md-1),flush=True)
                    print('LZ and ZN probs: {0} {1}'.format(float(p_lz),float(p_zn)),flush=True)
                    hop = True

                betta = gap[j_md-1]/ekin
                # check for frustrated hop condition should be imposed only for upward hops because for hops down betta is always positive 
                if (hop and betta <= 1.0 and target_state>flag_es):
                    print("Rejected (frustrated) hop",j_md-1, betta,flush=True)
                    hop = False
                if hop :
                    print('Switch from {0} to {1}'.format(flag_es,target_state),flush=True)
                    flag_es = target_state                   
                    # make one MD step back since local minimum was at t and we are at t+dt 
                    a.set_positions(coordinates)
            	    # velocity rescaling to conserve total energy
                    a.set_velocities(np.sqrt(1.0+betta)*velocities)
                    # set j_md to j_md-1 because we kind of make a step back in time 
                    j_md -= 1
                    count_interpol -= 1
                    
            if do_zn:
                return p_zn
            else:
                return p_lz

    # reset calculators to compute other energies
    a.set_calculator(calc)
    epot_down = a.get_potential_energy()/Hartree    
    a.set_calculator(calc2)
    epot_mid = a.get_potential_energy()/Hartree    
    a.set_calculator(calc3)
    epot_up = a.get_potential_energy()/Hartree    

    gap_mid_up[j_md] = np.abs(epot_up - epot_mid)
    gap_mid_down[j_md] = np.abs(epot_mid - epot_down)
    if j_md > 0:
        p_up = 0.0
        p_down = 0.0
        if flag_es==3:
            p_up = check_hop(gap_mid_up,force_up_t2,force_up_t1,force_mid_t2,force_mid_t1,flag_es+1)
            p_down = check_hop(gap_mid_down,force_mid_t2,force_mid_t1,force_down_t2,force_down_t1,flag_es-1)
        elif flag_es==2:
            p_up = check_hop(gap_mid_down,force_mid_t2,force_mid_t1,force_down_t2,force_down_t1,flag_es+1)
        elif flag_es==4:
            p_down = check_hop(gap_mid_up,force_up_t2,force_up_t1,force_mid_t2,force_mid_t1,flag_es-1)
 
    # set SPK model back to running state this already takes into account the switch if performed
    if flag_es==3:
        a.set_calculator(calc2)
    if flag_es==2:
    	a.set_calculator(calc)
    if flag_es==4:
    	a.set_calculator(calc3)
  
    # data output (time,energy gap in eV, up and down hopping probabilities, active state)
    if j_md > 0:
        df.write('{0:0.2f} {1:0.5f} {2:0.5f} {3} {4} {5}\n'.format(t_step[j_md-1],\
                gap_mid_down[j_md-1]*Hartree,gap_mid_up[j_md-1]*Hartree,float(p_down),float(p_up),flag_es))
        df.flush()
        fsync(df)
        # comment the line below to enable only one hop along the trajectory if not - several hops (also upward) are allowed
        hop = False
    
    j_md += 1

# run the molecular dynamics with TSH module
dyn.attach(tsh, interval=1)
dyn.run(n_md)

df.close()

#k = 10
#idx = np.argpartition(gap_12, k)
#print('Indices of {0} smallest gaps: {1}'.format(k,idx[:k]),flush=True)
	#This returns the k-smallest values. Note that these may not be in sorted order.
#with np.printoptions(precision=4, suppress=True):
#    print('Smallest values:',flush=True)
#    print(gap_12[idx[:k]]*Hartree,flush=True)
#print(np.argmin(gap_12),np.min(gap_12))


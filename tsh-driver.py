from os import path, fsync
from sys import argv
import subprocess

from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from torch import device as torch_device

from ase.units import Hartree,Bohr,fs,kB
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.calculators.demonnano import DemonNano

from pandas import read_csv
import numpy as np

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
print('Last git commit: {}'.format(get_git_revision_hash()))
print('Shorter hash: {}'.format(get_git_revision_short_hash()))

assert len(argv)>4, 'Not all arguments provided'
# number and size of MD steps from the arguments passed through command line
n_md = int(argv[1])
step_md = float(argv[2])  #in fs
# type of calculator (demonnano or schnet)
calc_type = str(argv[3])  
# type of hopping strategy (lz or zn)
hop_type = str(argv[4])  
# type of model (2state or 3state)
model_type = str(argv[5])  

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
elif hop_type=='zn':
    do_zn = True
assert do_lz or do_zn, 'Please enter a hopping type: lz or zn'
do_both = do_lz and do_zn

# flag to switch between 2-state and 3-state model 
do_3state = True
if model_type=='2state':
    do_3state = False
elif model_type=='3state':
    do_3state = True
else:
    print('Provide a model type (2state or 3state) as an argument',flush=True)
# flag that indicates on which state the trajectory is running
flag_es = 3
# flags for hopping
do_hop = True
hop = False
skip_next = False

# SchNet calculator
model = load_model(path1,map_location=torch_device('cpu'))
model2 = load_model(path2,map_location=torch_device('cpu'))
if do_3state:
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
# define calculators for 2- or 3-state model
if (calc_type=="demonnano"):
    calc = DemonNano(label='rundir/',input_arguments=input_arguments)
    calc2 = DemonNano(label='rundir/',input_arguments=input_arguments2)
    if do_3state:
        calc3 = DemonNano(label='rundir/',input_arguments=input_arguments3)
if (calc_type=="schnet"):
    calc = SpkCalculator(model, device="cpu", energy="energy", forces="forces")
    calc2 = SpkCalculator(model2, device="cpu", energy="energy", forces="forces")
    if do_3state:
        calc3 = SpkCalculator(model3, device="cpu", energy="energy", forces="forces")
# read geometry from file and set calculator
atoms=read('geom-phen.xyz')
# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, 300 *kB)
#Stationary(atoms)  # zero linear momentum
#ZeroRotation(atoms)  # zero angular momentum
# read velocities from velo file
velo = atoms.get_velocities()
data = read_csv('velo', delimiter='\s+', header=None, index_col=False)
data.columns = ["atom", "x", "y", "z"]
velo[:,0]=data.x
velo[:,1]=data.y
velo[:,2]=data.z
atoms.set_velocities(velo/fs)
#print(atoms.get_kinetic_energy()/ (1.5 * kB* len(atoms)))  
# we want to run MD with constant energy using the VelocityVerlet algorithm. 
dyn = VelocityVerlet(atoms, step_md * fs)	#, trajectory='md.traj', logfile='md.log')
# dynamical variables
gap_mid_down = np.zeros(n_md+1,dtype=np.float64)
gap_mid_up = np.zeros(n_md+1,dtype=np.float64)

t_step = np.zeros(n_md,dtype=np.float32)
t_step = np.arange(0, n_md)
t_step = t_step*step_md

j_md=0
tau_0 = 0.02418884326
dt = step_md/tau_0
# output file
df = open("gaps.txt","w+")
#df2 = open("energies.txt","w+")
# set initially excited state
if flag_es==3:
    atoms.set_calculator(calc2)
else:
    print("Are you sure?",flush=True)
# seed random number generator for reproducible results
#np.random.seed(1)
force_up_t1 = atoms.get_forces()*Bohr/Hartree
force_up_t2 = atoms.get_forces()*Bohr/Hartree
force_mid_t1 = atoms.get_forces()*Bohr/Hartree
force_mid_t2 = atoms.get_forces()*Bohr/Hartree
force_down_t1 = atoms.get_forces()*Bohr/Hartree
force_down_t2 = atoms.get_forces()*Bohr/Hartree

coordinates = atoms.get_positions()
velocities = atoms.get_velocities()
masses = atoms.get_masses()

ekin = 666.0
epot = 666.0
    
skip_count = 0

def tsh(a=atoms,dt=dt):  # store a reference to atoms in the definition.
    global j_md, flag_es, skip_count
    global do_hop, hop, skip_next
    global force_up_t1,force_down_t2,force_down_t1,force_up_t2,force_mid_t1,force_mid_t2
    global coordinates_t1,coordinates,velocities,velocities_t1,masses   
    global ekin,epot,gaps_mid_down, gaps_mid_up
    """TSH driver for 2- or 3-state model"""
    if j_md == 0:
        coordinates_t1 = a.get_positions()
        velocities_t1 = a.get_velocities()
        if do_3state:
            a.set_calculator(calc3)
            force_up_t1 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc)
        force_down_t1 = a.get_forces()*Bohr/Hartree    
        a.set_calculator(calc2)
        force_mid_t1 = a.get_forces()*Bohr/Hartree    
    if j_md > 0:
        coordinates = a.get_positions()
        velocities = a.get_velocities()
        if do_3state:
            a.set_calculator(calc3)
            force_up_t2 = a.get_forces()*Bohr/Hartree
        a.set_calculator(calc)
        force_down_t2 = a.get_forces()*Bohr/Hartree    
        a.set_calculator(calc2)
        force_mid_t2 = a.get_forces()*Bohr/Hartree    
    # function to attempt a hopping event
    def check_hop(atoms,energy_kin,energy_pot,gap,force_upper_t2,force_upper_t1,force_lower_t2,force_lower_t1,target_state):
        global flag_es, j_md
        global coordinates_t1, velocities_t1, dt
        global hop, do_hop

        etot = energy_pot + energy_kin
        #print(etot)
        p_zn = 0.0
        p_lz = 0.0
        small_dgap = False
        v_diab = 0.0
        # check for the local minimum of the energy gap between 2 states
        if (gap[j_md-1] < gap[j_md]) and (gap[j_md-2] > gap[j_md-1]):
            #print('Possible crossing at {}'.format(j_md))
            # Landau-Zener part
            dgap = (gap[j_md] - 2.0*gap[j_md-1] + gap[j_md-2])/(dt*dt)
            #dgap2 = (gap[j_md+1] - 2.0*gap[j_md] + gap[j_md-1])/(dt*dt)
            # compute Landau-Zener probability
            if(dgap<1E-12):
                #print('alert, small or negative d^2/dt^2',dgap,flush=True)
                small_dgap = True
            else:
                c_ij = np.power(gap[j_md-1],3)/dgap
                p_lz = np.exp(-0.5*np.pi*np.sqrt(c_ij)) 

            tau = 0.02418884326
            print('{0} {1} {2}'.format(dt*(j_md-1)*tau,dgap,np.power(gap[j_md-1],3)))

            # Zhu-Nakamura part
            # compute diabatic gradients according to Hanasaki et al. 2018 (JCP)
            sum_G = force_upper_t2+force_lower_t2
            dGc = (force_upper_t2-force_lower_t2) - (force_upper_t1-force_lower_t1)
            dGc /= dt
            # dGc*velo has to be in a.u. 
            conversion_velo = fs*tau/Bohr	#fs/(tau*Bohr)
            dGxVelo = np.tensordot(dGc,velocities_t1*conversion_velo)
            if (dGxVelo < 0.0):
                #print('negative product, use BL at ',j_md-1,flush=True)
                if not small_dgap:
                    factor=0.5*np.sqrt(gap[j_md-1]/dgap)
                else:
                    factor=0.0
            else:
                factor=0.5*np.sqrt(gap[j_md-1]/dGxVelo)
  
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
            
            v_diab = gap[j_md-1]/2.0
            a2 = 0.5*force_diff*force_prod/(np.power(2.0*v_diab,3))
            b2 = energy_kin*force_diff/(force_prod*2.0*v_diab)
            if (a2<0.0) or (b2<0.0) :
                print('Alert!',flush=True)
            root = 0.0
            if do_plus:
                root = b2 + np.sqrt(b2*b2 + 1.0)
            else:
                root = b2 + np.sqrt(np.abs(b2*b2 - 1.0)) 
            # compute Zhu-Nakamura probability
            if not small_dgap:
                p_zn = np.exp(-np.pi*0.25*np.sqrt(2.0/(a2*root)))
            else:
                #print('Issue with second derivative of gap in BL',flush=True)
                p_zn = 0.0
            # comparison with a random number
            if do_hop and (not hop):
                xi = np.random.rand(1)
                if xi <= p_lz and do_lz:
                    #print('Attempted hop according to Landau-Zener at {}'.format(j_md-1),flush=True)
                    #print('Landau-Zener prob: {}'.format(float(p_lz)),flush=True)
                    hop = True
                elif xi <= p_zn and do_zn:
                    #print('Attempted hop according to Zhu-Nakamura at {}'.format(j_md-1),flush=True)
                    #print('Zhu-Nakamura prob: {}'.format(float(p_zn)),flush=True)
                    hop = True
                elif xi <= p_zn and xi <= p_lz and do_both:
                    #print('Attempted hop according to LZ and ZN at {} !'.format(j_md-1),flush=True)
                    #print('LZ and ZN probs: {0} {1}'.format(float(p_lz),float(p_zn)),flush=True)
                    hop = True

                betta = gap[j_md-1]/energy_kin
                # check for frustrated hop condition should be imposed only for upward hops because for hops down betta is always positive 
                if (hop and betta > 1.0 and target_state>flag_es):
                    #print("Rejected (frustrated) hop at",j_md-1, betta,flush=True)
                    hop = False
                if hop :
                    #print('Switch from {0} to {1}'.format(flag_es,target_state),flush=True)
                    #print('{0} {1}'.format(target_state, gap[j_md-1]))
                    # make one MD step back since local minimum was at t and we are at t+dt 
                    a.set_positions(coordinates_t1)
            	    # velocity rescaling to conserve total energy
                    if target_state<flag_es:
                        a.set_velocities(np.sqrt(1.0+betta)*velocities_t1)
                    else:
                        a.set_velocities(np.sqrt(1.0-betta)*velocities_t1)
                    # change the running state
                    flag_es = target_state                   

        if do_zn:
            return p_zn
        else:
            return p_lz

    epot_up = 0.0
    # reset calculators to compute other energies   
    if skip_count!=1:	#not skip_next:
        a.set_calculator(calc)
        epot_down = a.get_potential_energy()/Hartree    
        a.set_calculator(calc2)
        epot_mid = a.get_potential_energy()/Hartree    
        if do_3state:
            a.set_calculator(calc3)
            epot_up = a.get_potential_energy()/Hartree    
            gap_mid_up[j_md] = np.abs(epot_up - epot_mid)
 
        gap_mid_down[j_md] = np.abs(epot_mid - epot_down)

    if flag_es==3:
        a.set_calculator(calc2)
    elif flag_es==2:
    	a.set_calculator(calc)
    elif flag_es==4 and do_3state:
    	a.set_calculator(calc3)

    p_up = 0.0
    p_down = 0.0
    if j_md > 1 and not skip_next:
        if flag_es==3:
            p_down = check_hop(a,ekin,epot,gap_mid_down,force_mid_t2,force_mid_t1,force_down_t2,force_down_t1,flag_es-1)
            if do_3state:
                p_up = check_hop(a,ekin,epot,gap_mid_up,force_up_t2,force_up_t1,force_mid_t2,force_mid_t1,flag_es+1)
        elif flag_es==2:
            p_up   = check_hop(a,ekin,epot,gap_mid_down,force_mid_t2,force_mid_t1,force_down_t2,force_down_t1,flag_es+1)
        elif flag_es==4 and do_3state:
            p_down = check_hop(a,ekin,epot,gap_mid_up,force_up_t2,force_up_t1,force_mid_t2,force_mid_t1,flag_es-1)
        #print(p_up,p_down)
    # set SPK model back to running state this already takes into account the switch if performed
    if flag_es==3:
        a.set_calculator(calc2)
    elif flag_es==2:
    	a.set_calculator(calc)
    elif flag_es==4 and do_3state:
    	a.set_calculator(calc3)
    # data output (time,energy gap in eV, up and down hopping probabilities, active state)
    if j_md > 0 and skip_count!=1 :	# and not skip_next:
        df.write('{0:0.2f} {1:0.5f} {2:0.5f} {3:0.5f} {4:0.5f} {5:0.5f} {6}\n'.format(t_step[j_md-1],\
                 float(epot_down*Hartree),float(epot_mid*Hartree),float(epot_up*Hartree),float(epot*Hartree),\
                 float(p_down),float(p_up),flag_es))
        #df.write('{0:0.2f} {1:0.5f} {2:0.5f} {3:0.5f} {4:0.5f} {5}\n'.format(t_step[j_md-1],\
                 #gap_mid_down[j_md-1]*Hartree,gap_mid_up[j_md-1]*Hartree,float(p_down),float(p_up),flag_es))
        #df.flush()
        #fsync(df)

    ekin = a.get_kinetic_energy()/Hartree
    epot = a.get_potential_energy()/Hartree 
    velocities_t1 = velocities
    coordinates_t1 = coordinates
    force_up_t1 = force_up_t2
    force_down_t1 = force_down_t2
    force_mid_t1 = force_mid_t2
    if skip_next:
        if skip_count==3:
            skip_count = 0
            skip_next = False
        else:
            skip_count += 1

    if hop:
        # decrement j_md because we kind of make a step back in time and skip next step
        j_md -= 1
        skip_next = True

        # comment the line below to enable only one hop along the trajectory if not - several hops (also upward) are allowed
        hop = False
    
    j_md += 1

# run the molecular dynamics with TSH module
dyn.attach(tsh, interval=1)
dyn.run(n_md)

df.close()
#df2.close()
#print('finished',flush=True)


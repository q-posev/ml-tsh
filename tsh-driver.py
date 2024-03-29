from os import path, fsync
from sys import argv, exit

import subprocess

from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from torch import device as torch_device

from ase.units import Hartree,Bohr,fs,kB
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.demonnano import DemonNano

from pandas import read_csv
import numpy as np

# output hash of the last git commit
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
print('Last git commit: {}'.format(get_git_revision_hash()))

assert len(argv)>4, 'Not all arguments provided'
# number of MD steps
n_md = int(argv[1])
# md time step in fs (e.g. 0.25)
step_md = float(argv[2])  
# type of calculator (demonnano or schnet)
calc_type = argv[3]  
# type of hopping strategy (lz or zn)
hop_type = argv[4]
# type of model (2state or 3state)
model_type = argv[5]
# load SchNet models (pre-trained)
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

# flag to switch between 2-state and 3-state model 
do_3state = True
if model_type=='2state':
    do_3state = False
elif model_type=='3state':
    do_3state = True
else:
    print('Provide a model type (2state or 3state) as an argument')
# flag that indicates on which state the trajectory is running
flag_es = 3
# flags for hopping
do_hop = True
hop = False
skip_next = False

# SchNet calculator
model_s = load_model(path1,map_location=torch_device('cpu'))
model2_s = load_model(path2,map_location=torch_device('cpu'))
if do_3state:
    model3_s = load_model(path3,map_location=torch_device('cpu'))

model = model_s.double()
model2 = model2_s.double()
if do_3state:
    model3 = model3_s.double()

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
# define calculators for ASE electronic structure part
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

# set initial conditions for the molecule
atoms=read('geom-phen.xyz')
MaxwellBoltzmannDistribution(atoms, 300 *kB)
velo = atoms.get_velocities()
data = read_csv('velo', delimiter='\s+', header=None, index_col=False)
data.columns = ["atom", "x", "y", "z"]
velo[:,0]=data.x
velo[:,1]=data.y
velo[:,2]=data.z
atoms.set_velocities(velo/fs)
# set VelocityVerlet driver for MD
dyn = VelocityVerlet(atoms, step_md * fs)	
	#, trajectory='md.traj', logfile='md.log') 
        # uncomment parameters above for more detailed output
# energy gaps between neighboring states
gap_mid_down = np.zeros(n_md+1,dtype=np.float64)
gap_mid_up = np.zeros(n_md+1,dtype=np.float64)
# time properties
t_step = np.zeros(n_md,dtype=np.float32)
t_step = np.arange(0, n_md)
t_step = t_step*step_md
# constant to go from fs to a.u.
tau_0 = 0.02418884326
dt = step_md/tau_0
# output files
df = open("gaps.txt","w+")
#df2 = open("deriv_gaps.txt","w+")
# set calculator according to initally excited state
if flag_es==3:
    atoms.set_calculator(calc2)
else:
    print("undefined behaviour")
    exit()
# seed random number generator for reproducible results
#np.random.seed(1)
# properties required for ZN
force_up_t1 = atoms.get_forces()*Bohr/Hartree
force_up_t2 = atoms.get_forces()*Bohr/Hartree
force_mid_t1 = atoms.get_forces()*Bohr/Hartree
force_mid_t2 = atoms.get_forces()*Bohr/Hartree
force_down_t1 = atoms.get_forces()*Bohr/Hartree
force_down_t2 = atoms.get_forces()*Bohr/Hartree
# common properties
coordinates = atoms.get_positions()
velocities = atoms.get_velocities()

m0 = 1836.229
masses = atoms.get_masses()*m0

ekin = 666.0
epot = 666.0
# MD step counter
j_md = 0
skip_count = 0

def tsh(a=atoms,dt=dt):  # store a reference to atoms in the definition.
    global j_md, flag_es, skip_count
    global do_hop, hop, skip_next
    global force_up_t1,force_down_t2,force_down_t1,force_up_t2,force_mid_t1,force_mid_t2
    global coordinates_t1,coordinates,velocities,velocities_t1,masses   
    global ekin,epot,gaps_mid_down, gaps_mid_up, tau_0
    """Main TSH driver for 2- or 3-state model"""
    # get the properties needed for hopping probability
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

    def check_hop(atoms,energy_kin,energy_pot,gap,force_upper_t2,force_upper_t1,force_lower_t2,force_lower_t1,target_state):
        global flag_es, j_md
        global coordinates_t1, velocities_t1, dt, tau_0
        global hop, do_hop
        """Check for local minimum and hopping attempt"""
        etot = energy_pot + energy_kin
        p_zn = 0.0
        p_lz = 0.0
        small_dgap = False
        v_diab = 0.0
        # check for the local minimum of the energy gap
        if (gap[j_md-1] < gap[j_md]) and (gap[j_md-2] > gap[j_md-1]):
            # finite difference calculation of second order derivative
            dgap = (gap[j_md] - 2.0*gap[j_md-1] + gap[j_md-2])/(dt*dt)
            # compute Belyaev-Lebedev probability
            if(dgap<1E-12):
                #print('small or negative d^2/dt^2',dgap)
                small_dgap = True
            else:
                c_ij = np.power(gap[j_md-1],3)/dgap
                p_lz = np.exp(-0.5*np.pi*np.sqrt(c_ij)) 

            # output second time derivative between S2 and S3
            #if target_state==2:
            #    df2.write('{0:0.2f} {1} {2} \n'.format(dt*(j_md-1)*tau_0,\
            #                     dgap,np.power(gap[j_md-1],3))) 

            # compute diabatic gradients for ZN
            sum_G = force_upper_t2+force_lower_t2
            dGc = (force_upper_t2-force_lower_t2) - (force_upper_t1-force_lower_t1)
            dGc /= dt
            # dGc*velo has to be in a.u. 
            conversion_velo = fs*tau_0/Bohr
            dGxVelo = np.tensordot(dGc,velocities_t1*conversion_velo)
            if (dGxVelo < 0.0):
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
                print('Alert!')
            root = 0.0
            if do_plus:
                root = b2 + np.sqrt(b2*b2 + 1.0)
            else:
                root = b2 + np.sqrt(np.abs(b2*b2 - 1.0)) 
            # compute Zhu-Nakamura probability
            if not small_dgap:
                p_zn = np.exp(-np.pi*0.25*np.sqrt(2.0/(a2*root)))
            else:
                p_zn = 0.0
            # comparison with a random number
            if do_hop and (not hop):
                xi = np.random.rand(1)
                if xi <= p_lz and do_lz:
                    #print('Attempted hop according to BL at {}'.format(j_md-1))
                    hop = True
                elif xi <= p_zn and do_zn:
                    #print('Attempted hop according to ZN at {}'.format(j_md-1))
                    hop = True

                betta = gap[j_md-1]/energy_kin
                # check for frustrated hop condition 
                if (hop and betta > 1.0 and target_state>flag_es):
                    hop = False
                if hop :
                    # output energy gap at the hopping point
                    print('{0} {1}'.format(target_state, gap[j_md-1]))

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
    # reset calculators to energy gaps along a trajectory
    if skip_count != 1:	
        a.set_calculator(calc)
        epot_down = a.get_potential_energy()/Hartree    
        a.set_calculator(calc2)
        epot_mid = a.get_potential_energy()/Hartree    
        if do_3state:
            a.set_calculator(calc3)
            epot_up = a.get_potential_energy()/Hartree    
            gap_mid_up[j_md] = np.abs(epot_up - epot_mid)
 
        gap_mid_down[j_md] = np.abs(epot_mid - epot_down)
    # set the calculator back to the running state
    if flag_es==3:
        a.set_calculator(calc2)
    elif flag_es==2:
    	a.set_calculator(calc)
    elif flag_es==4 and do_3state:
    	a.set_calculator(calc3)

    p_up = 0.0
    p_down = 0.0
    # main part for 2- and 3-state models
    if j_md > 1 and not skip_next:
        if flag_es==3:
            p_down = check_hop(a,ekin,epot,gap_mid_down,force_mid_t2,force_mid_t1,force_down_t2,force_down_t1,flag_es-1)
            if do_3state and not hop:
                p_up = check_hop(a,ekin,epot,gap_mid_up,force_up_t2,force_up_t1,force_mid_t2,force_mid_t1,flag_es+1)
        elif flag_es==2:
            p_up   = check_hop(a,ekin,epot,gap_mid_down,force_mid_t2,force_mid_t1,force_down_t2,force_down_t1,flag_es+1)
        elif flag_es==4 and do_3state:
            p_down = check_hop(a,ekin,epot,gap_mid_up,force_up_t2,force_up_t1,force_mid_t2,force_mid_t1,flag_es-1)
    # set the calculatior back to the running state, this already takes into account the switch if performed
    if flag_es==3:
        a.set_calculator(calc2)
    elif flag_es==2:
    	a.set_calculator(calc)
    elif flag_es==4 and do_3state:
    	a.set_calculator(calc3)
    # data output (time, energies in eV, hopping probabilities, active state)
    if j_md>0 and skip_count!=1 :
        df.write('{0:0.2f} {1:0.5f} {2:0.5f} {3:0.5f} {4:0.5f} {5:0.5f} {6:0.5f} {7}\n'.format(t_step[j_md-1],\
                 float(epot_down*Hartree),float(epot_mid*Hartree),float(epot_up*Hartree),float(epot*Hartree),\
                 float(p_down),float(p_up),flag_es))
    # save values from the previous step
    ekin = a.get_kinetic_energy()/Hartree
    epot = a.get_potential_energy()/Hartree 
    velocities_t1 = velocities
    coordinates_t1 = coordinates
    force_up_t1 = force_up_t2
    force_down_t1 = force_down_t2
    force_mid_t1 = force_mid_t2
    # condition to avoid hops immediately after each other
    if skip_next:
        if skip_count==3:
            skip_count = 0
            skip_next = False
        else:
            skip_count += 1
    # if the hop was accepted
    if hop:
        # decrement j_md because we made one MD step back
        j_md -= 1
        skip_next = True
        hop = False
    # increment global MD counter 
    j_md += 1

"""Launcher for molecular dynamics"""
# run MD for n_md steps with TSH module
dyn.attach(tsh, interval=1)
dyn.run(n_md)

# close data files
df.close()
#df2.close()


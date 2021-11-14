


import math as m
import numpy as np
import shutil

import sys
from mpi4py import MPI
from scipy.special import erf
import time
import glob, os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools
from dedalus.tools import post

import logging

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

# Some helper functions 

def format_params(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    params = []
    for line in lines:
        line = line.split('#')[0]
        line = line.rstrip()
        if "'" not in line:
            try:
                params.append(int(line))
            except:
                params.append(float(line))
        else:
            try:
                params.append(line.replace("'", ''))
            except:
                if line != '':
                    params.append(line)
                else:
                    pass
    
    return params


def list_files(path):
    # List of files in complete directory
    # Sourced from https://www.askpython.com/python/examples/list-files-in-a-directory-using-python
    file_list = []

    """
        Loop to extract files inside a directory

        path --> Name of each directory
        folders --> List of subdirectories inside current 'path'
        files --> List of files inside current 'path'

    """
    for path, folders, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(path, file))

    return list(file_list)



# Calculate boat function K on supplied grid points
def mask(x,z,phi,delta,lx,lz, inflow):
    cos, sin = np.cos(phi), np.sin(phi)
    x1,x2 = (x * cos + z * sin), (z * cos - x * sin)
    return 0.5 * (1 - erf(C(x1, x2, lx, lz, inflow) / delta))

def C(x, z, lx, lz, inflow, grad=False):
    Z0 = -1
    m, n = 2, 2
    Z = np.abs(2 * x / lx) ** m + np.abs(z / lz) ** n + Z0 
    return Z
    
    
def plot_q(q, Γ, domain, iteration):
    res = 3/2
    x2,y2 = domain.grids(res)
    xx2,yy2 = x2+0*y2, 0*x2+y2
    fig, ax = plt.subplots()
    q.set_scales(res)
    ax.pcolormesh(xx2,yy2,q['g'],cmap='PuOr_r',vmin=-4,vmax=4)
    ax.contourf(xx2,yy2,Γ['g'],[.05,1.1],colors='black')
    clean_ax(ax)
    plt.tight_layout()
    plt.savefig(f'frames/flow-vorticity-{iteration:0>5d}.jpg')
    plt.close('all')
    
def clean_ax(ax):
    ax.set(aspect=1,xticks=[],yticks=[])
    for n in ax.spines: ax.spines[n].set_visible(False)
    
    
def plot_interface_surface(p, f_int, Γ, domain, iteration):
    res = 3/2
    x2,y2 = domain.grids(res)
    xx2,yy2 = x2+0*y2, 0*x2+y2
    fig, ax = plt.subplots()
    q.set_scales(res)
    ax.plot(x2,p['g'][:,-1])#,cmap='PuOr_r',vmin=-20,vmax=20)
    ax.plot(x2, f_int['g'][:,0])
    
    ax.contourf(xx2,yy2,Γ['g'],[.05,1.1],colors='black')
    clean_ax(ax)
    plt.tight_layout()
    plt.savefig(f'frames/flow-heights-{iteration:0>5d}.jpg')
    plt.close('all')



def dead_water(path, sim_name, iteration_num, fluid_params, boat_params, sim_params,
               boundary_params=[False, 1,.1], linear=False,speed=[False,0.7,1.5]):
    
    # Parameter definitions
    nu, kappa, rho1, rho2, Lx, H1, H2, dz, T0, inflow = tuple(fluid_params)
    gamma, delta, lxb, lzb, xb, zb, Ub, Wb, phi_b, omega_b, f = tuple(boat_params)
    
    resx, resz, wall_time, dt, steps, sim_name, calc_freq, print_freq, save_freq, save_num = tuple(sim_params)
    boundary, offset, delta_boundary = tuple(boundary_params)
    
    drho = rho2 - rho1
    dB = 2 * drho * g / (rho2 + rho1)
    
    # Create bases and domain
    xbasis = de.Fourier('x', int(resx * Lx), interval=(0, Lx), dealias=3/2)
    zbasis = de.Chebyshev('z', int(resz * (H1 + H2)), interval=(-H2, H1), dealias=3/2)

    domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)    
    x, z = domain.grids(scales=domain.dealias)

    
    X, Z, Rx, Rz, U, W, K = domain.new_field(), domain.new_field(), domain.new_field(), domain.new_field(), domain.new_field(), domain.new_field(), domain.new_field()    
    for field in [X, Z, Rx, Rz, U, W, K]:
        field.set_scales(domain.dealias, keep_data=False)
    
    X['g'] = xb
    Z['g'] = zb    
    Rx['g'] = np.mod(x - xb + Lx / 2, Lx) - Lx / 2 # Boat repeats horizontally. 
    Rz['g'] = z - zb # zb = boat depth (probably?). Boat no deeper
    K['g'] = mask(Rx['g'], Rz['g'], phi_b, delta, lxb, lzb, inflow)
    U['g'] = Ub - omega_b * Rz['g']
    W['g'] = Wb + omega_b * Rx['g']
    vol_b = K.integrate('x','z')['g'][0][0]
    
    # Constant speed simulations
    if speed[0] == True:
        umax, tau = speed[1], speed[2]
        X['g'] = xb
        U['g'] = umax*0.5*(np.tanh(-3) + 1)
        
        
    # Optional immersed boundary field about x = 0
    if boundary:
        K_x = domain.new_field()
        K_x.set_scales(domain.dealias,keep_data=False)
        K_x['g'] = 0.5*(erf((x - Lx + offset)/delta_boundary) - erf((x - offset)/delta_boundary) + 2)

    # Background buoyancy fields
    B = domain.new_field()
    Bz = domain.new_field()
    B.set_scales(domain.dealias, keep_data=False)
    Bz.set_scales(domain.dealias, keep_data=False)
    B['g'] = dB*(erf(z/dz))/2

    if linear:
        B['g'] = dB*( (z-H1)/(H1+H2) + 0.5)
    Bz = B.differentiate(z=1)
    

    # Problem definition and state variables. p is the gradient scalar, and q is the vorticity
    problem = de.IVP(domain, variables=['p','u','w','q','b','bz'])
    
    
    # Problem Parameters
    params = [nu, kappa, gamma, delta, K, X, Z, U, W, B, Bz, resx*Lx, resz*(H1+H2), vol_b, rho1, rho2, H1, H2, Lx, lxb, lzb, f, g, T0, inflow]
    param_names = ['nu','kappa','gamma','delta','K','X','Z','U','W','B','Bz','N_x','N_z','vol_b', 'rho_1','rho_2','Lz_1','Lz_2','Lx','lxb','lzb','f','g','T0', 'inflow']
    for param, param_name in zip(params, param_names):
        problem.parameters[param_name] = param
    
    if boundary:
        problem.parameters['K_x'] = K_x

    if speed[0]:
        problem.parameters['umax'] = umax
        problem.parameters['tau'] = tau
        
        
    # Variable information
    problem.meta[:]['z']['dirichlet'] = True
    Bz.meta['x']['constant'] = True
    X.meta['x','z']['constant'] = True
    Z.meta['x','z']['constant'] = True
    U.meta['x','z']['constant'] = True
    W.meta['x','z']['constant'] = True
    
    if boundary:
        K_x.meta['z']['constant'] = True
    
    # Substitutions
    problem.substitutions['drag'] = "gamma*integ(K*(u-U))/vol_b"
    problem.substitutions['p_real'] = "p - 0.5*(u*u + w*w)"
    problem.substitutions['U_in'] = "inflow * (tanh((t - T0) / T0) + 1) / 2"
    problem.substitutions['enstrophy'] = "integ(q**2)"




    # Equations
    u_eqn = "dt(u) - nu*dz(q) + dx(p) = -q*w - gamma*(K*(u-U))"
    w_eqn = "dt(w) + nu*dx(q) + dz(p) - b = q*u -gamma*(K*(w-W))"


    # Constant velocity at inflow and outflow. 
    
    if boundary:
        u_eqn = u_eqn[:-1] + " + K_x*(u - U_in)" + u_eqn[-1] 
        w_eqn = w_eqn[:-1] + " + K_x*w" + w_eqn[-1]
        
        
        
        
        
        
    problem.add_equation("dx(u) + dz(w) = 0")
    problem.add_equation(u_eqn)
    problem.add_equation(w_eqn)
    problem.add_equation("q - dz(u) + dx(w) = 0")
    problem.add_equation("dt(b) - kappa*d(b,x=2) - kappa*dz(bz) = -u*dx(b) - w*bz - w*Bz ")
    problem.add_equation("bz - dz(b) = 0 ")
    
    
    # Boundary Conditions
    # What are the correct boundary conditions?
    problem.add_bc("left(u) = U_in") 
    problem.add_bc("left(w) = 0") 
    problem.add_bc("left(bz) = 0")
    problem.add_bc("right(q) = 0")    
    problem.add_bc("right(dt(p) - g * w) = 0")    
    problem.add_bc("right(bz) = 0")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')


    try:	
        if iteration_num == 0:
            raise Exception()
        load_path = f'{path}/data_{sim_name}'.replace(f'iter-{iteration_num}',f'iter-{iteration_num-1}')
        files = sorted(list_files(load_path))
        j = 1
        while os.path.isfile(load_path + f'/data_{sim_name}_s{j}.h5'):
            j += 1
        loadfile = load_path + f'/data_{sim_name}_s{j-1}.h5'
    except:
        loadfile = None

    
    # ONLY IF RESTART CHECK NUM FILES IN DIRECTORY ONLY ACCESS THE LAST ONES
    if loadfile != None:
        write, dt = solver.load_state(loadfile, -1)
        # Load the variables into their states
        # u, w, p, q, b, bz = solver.state['u'], solver.state['w'], solver.state['p'], solver.state['q'], solver.state['b'], solver.state['bz']
    else:            
        # Initial conditions
        u, w, p, q, b, bz = solver.state['u'], solver.state['w'], solver.state['p'], solver.state['q'], solver.state['b'], solver.state['bz']
        
        for field in [u,w,p,q,b,bz]:
            field.set_scales(domain.dealias, keep_data=False)
            field['g'] = 0
    
    # Integration parameters
    solver.stop_sim_time = np.inf
    solver.stop_wall_time = wall_time
    solver.stop_iteration = steps
    # Saving state variable data
    snapshots = solver.evaluator.add_file_handler('{}/data_{}'.format(path, sim_name), iter = save_freq[0], max_writes = save_num[0])
    for task in ['p','u','w','q', 'b', 'bz', 'K', 'enstrophy']:
        snapshots.add_task(task)
        
    
    # Saving simulation parameters
    parameters = solver.evaluator.add_file_handler('{}/parameters_{}'.format(path, sim_name), iter=np.inf, max_writes=save_num)
    param_names = ['nu','kappa','gamma','delta','N_x','N_z','vol_b', 'rho_1','rho_2','Lz_1','Lz_2','Lx','lxb','lzb','f','g','T0', 'inflow']
    
    for param_name in param_names:
        parameters.add_task(param_name)
    
    if boundary:
        parameters.add_task('K_x')
    
    if speed[0]:
        parameters.add_task('tau')
        parameters.add_task('umax')
        
    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=calc_freq)
    flow.add_property("abs(q)", name='q')
    flow.add_property("abs(u)", name='u')
    flow.add_property("U_in", name='U_in')
    flow.add_property('2 * lxb * (((u - U_in) ** 2 + w ** 2) ** 0.5) / nu', name='Re')
    
    # Naive force
    force = flow_tools.GlobalFlowProperty(solver, cadence=1)
    force.add_property("K*(u-U)", name='fx', precompute_integral=True)
    force.add_property("K*(w-W)", name='fz', precompute_integral=True)
    
    


    # Main loop
    try:

        logger.info('Starting loop')
        start_time = time.time()
        while solver.ok:
            t = solver.sim_time
            solver.step(dt)
            # Calculate boat acceleration and update position
            Ax = gamma*(force.properties['_fx_integral']['g'][0,0])/vol_b
            Az = gamma*(force.properties['_fz_integral']['g'][0,0])/vol_b
            if not speed[0]:
                X['g'] = X['g'] + U['g']*dt + 0.5 * np.sin(max([t-T0, 0]))
                U['g'] = U['g'] + (Ax+f)*dt
            if speed[0]:
                X['g'] = xb #+ 0.05 * np.sin(2 * max([t-20., 0.])) * (t >= 20) + umax*0.5*(tau*(np.log(np.cosh(t/tau -3))-np.log(np.cosh(-3))) + t)
                U['g'] = 0#(0.05 * 2) * np.cos(2 * max([t-20., 0.])) * (t >= 20) + umax*0.5*(np.tanh(t/tau - 3) + 1)
                
            Rx['g'] = np.mod(x-X['g']+Lx/2,Lx)-Lx/2            
            K['g'] = mask(Rx['g'], Rz['g'], phi_b, delta, lxb, lzb, inflow)
            
            if X['g'][0,0] > Lx - offset - lxb:
                logger.info('Reached end')
                break
            
            if (solver.iteration-1) % print_freq == 0:
                logger.info('Iteration: %i, Time: %e, dt: %e'%(solver.iteration, solver.sim_time, dt))
                logger.info('Max|q| = {}'.format(flow.max('q')))
                logger.info('Boat vel = {}'.format(U['g'][0,0]))
                logger.info('Volume Averaged Re: {:.2e}'.format(flow.volume_average('Re')))
                logger.info('Enstropy: {:.3e}'.format(flow.max('enstropy')))
                
                #plot_inf(domain, b, q, p, K, solver.iteration, t)
                

	            
            if (solver.iteration-1) % calc_freq == 0 and np.isnan(flow.max('q')):
                logger.info('Simulation broken')
                break
            

    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr'%((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    for directory in sorted(glob.glob('{}/*_{}'.format(path, sim_name))):
        post.merge_analysis(directory, cleanup=True)
        
        
args = [k for k in sys.argv]
f = args[1]
params = format_params(f)
root_path = args[2]

# Fluid Parameters
(nu, kappa, g, rho1, rho2, delta_z, H1, H2, Lx, tau), T0 = params[0:10], params[13]
c = lambda g, rho1, rho2, H1, H2: np.sqrt((rho2-rho1)*g/(rho1/H1 + rho2/H2))
cmax = c(g, rho1, rho2, H1, H2)
const_speed = [bool(params[10]), params[11] * cmax, params[12]]

# Boat Parameters
f, gamma, delta, lxb, lzb, xb, zb, Ub, Wb, phi_b, omega_b = params[14:25]


(exp, t, dt, steps, resx, resz, calc_freq, print_freq) = params[25:33]
save_freq = params[33:35]
save_num = params[35:37]
sim_name = '{}_const_template'.format(exp)
wall_time = params[37]
boundary = bool(params[38])
offset = params[39]
delta_boundary = params[40]
inflow = params[41]

path = '{}exp{}'.format(root_path, exp)

i = 0
while os.path.isdir(path + f'-iter-{i}'):
    i += 1

path = path + f'-iter-{i}'

import shutil

if not os.path.isdir(path):
    os.mkdir(path)
else:
    shutil.rmtree(path)
    os.mkdir(path)

iteration_num = i

#Buoyancy field
fluid_params = [nu, kappa, rho1, rho2, Lx, H1, H2, delta_z, T0, inflow]
boat_params = [gamma, delta, lxb, lzb, xb, zb, Ub, Wb, phi_b, omega_b, f]
sim_params = [resx, resz, wall_time, dt, steps, sim_name, calc_freq, print_freq, save_freq, save_num]
boundary_params = [boundary, offset, delta_boundary]
# Run simulation
dead_water(path, sim_name, iteration_num, fluid_params, boat_params, sim_params, boundary_params=boundary_params, speed=const_speed)

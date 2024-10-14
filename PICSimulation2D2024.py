import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm

from scipy.sparse import kron, eye, diags
from scipy.sparse.linalg import spsolve

def calculateNumberDensity(particle_positions):
	global Nc
	global box_size

	cell_size = box_size/Nc

	# Start by fining the coordinates of the cell in the cell coordinates (i,j) -> (i+1,j+1) that contain each particle in order bl, br, tl, tr
	ij_BL = np.floor(particle_positions / cell_size[:, np.newaxis]).astype(int)
	ij_BR = np.array([ij_BL[0]+1,ij_BL[1]])
	ij_TL = np.array([ij_BL[0],ij_BL[1]+1])
	ij_TR = np.array([ij_BL[0]+1,ij_BL[1]+1])

	# The displacement of each particle from the ij to its bottom left in ij coordinates
	dx, dy = particle_positions / cell_size[:, np.newaxis] - ij_BL

	# Finding the weight each particle has on each of the corners of the cell that contains it
	ij_BL_weight	= (1-dx) * (1-dy)
	ij_BR_weight	= dx * (1-dy)
	ij_TL_weight	= (1-dx) * dy
	ij_TR_weight 	= dx * dy

	global plot_cell_markers
	# Define the colormap
	cmap = cm.get_cmap('Grays')
	if plot_cell_markers == True:
		plt.scatter(ij_BL[0]*cell_size[0],ij_BL[1]*cell_size[1],marker='+',c=cmap(ij_BL_weight))
		plt.scatter(ij_BR[0]*cell_size[0],ij_BR[1]*cell_size[1],marker='+',c=cmap(ij_BR_weight))
		plt.scatter(ij_TL[0]*cell_size[0],ij_TL[1]*cell_size[1],marker='+',c=cmap(ij_TL_weight))
		plt.scatter(ij_TR[0]*cell_size[0],ij_TR[1]*cell_size[1],marker='+',c=cmap(ij_TR_weight))

	# Create a density array and add the weights from every particle
	n = np.zeros((Nc[0]+1,Nc[1]+1))
	# Function to add weights to grid
	def addToGrid(ij, wij):
		for (i, j), w in zip(zip(ij[0], ij[1]), wij):
				n[i, j] += w
    
    # Add weights from each direction
	addToGrid(ij_BL, ij_BL_weight)
	addToGrid(ij_BR, ij_BR_weight)
	addToGrid(ij_TL, ij_TL_weight)
	addToGrid(ij_TR, ij_TR_weight)

	global plot_number_density
	if plot_number_density == True:
		#scaled_grid = np.kron(n, np.ones((5, 5)))

		# Plot the shifted grid
		plt.imshow((n.T)/5%1, origin='lower',extent=[-cell_size[0]/2,box_size[0]+cell_size[0]/2,-cell_size[1]/2,box_size[1]+cell_size[1]/2], cmap='jet', interpolation='bilinear',vmax=0.5)

	return n, ij_BL, ij_BR, ij_TL, ij_TR, ij_BL_weight, ij_BR_weight, ij_TL_weight, ij_TR_weight

def calculatePotential(n):
	"""
    Calculate the potential matrix phi given the number density matrix n and cell sizes.
    
    Parameters:
    n (numpy.ndarray): 2D array representing the number density matrix.
    cell_size (list or tuple): Array containing the cell sizes [Delta x, Delta y].
    
    Returns:
    numpy.ndarray: 2D array representing the potential matrix phi.
    """
	global Nc
	global box_size
	cell_size = box_size/Nc

	global N
	n0 = N/(box_size[0]*box_size[1])

	n = n - n0

    # Extract grid sizes and cell sizes
	Ny, Nx = n.shape
	delta_x, delta_y = cell_size

    # Create 1D finite difference matrix for x-direction
	main_diag_x = -2 * np.ones(Nx)
	off_diag_x = np.ones(Nx - 1)
	A1D_x = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
	A1D_x = A1D_x + diags([1], [Nx-1], shape=(Nx, Nx)) + diags([1], [-Nx+1], shape=(Nx, Nx))

    # Create 1D finite difference matrix for y-direction
	main_diag_y = -2 * np.ones(Ny)
	off_diag_y = np.ones(Ny - 1)
	A1D_y = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))
	A1D_y = A1D_y + diags([1], [Ny-1], shape=(Ny, Ny)) + diags([1], [-Ny+1], shape=(Ny, Ny))


    # Identity matrices
	Ix = eye(Nx)
	Iy = eye(Ny)

    # Create 2D Laplacian matrix using Kronecker products
	A = (kron(Iy, A1D_x) / delta_x**2) + (kron(A1D_y, Ix) / delta_y**2)

    # Flatten the number density matrix to a vector
	b = n.flatten()

    # Solve the linear system A * phi = b
	phi = spsolve(A, b)

    # Reshape the solution vector back to a 2D grid
	phi = phi.reshape((Ny, Nx))

	global plot_potential
	if plot_potential == True:
		if plot_potential == True:
			# Plot the shifted grid
			plt.imshow((phi.T), origin='lower',extent=[-cell_size[0]/2,box_size[0]+cell_size[0]/2,-cell_size[1]/2,box_size[1]+cell_size[1]/2], cmap='jet', interpolation='bilinear')

	return phi

def calculateElectricField(phi):
	global Nc
	global box_size
	cell_size = box_size/Nc

    # Extract grid sizes and cell sizes
	Ny, Nx = phi.shape
	delta_x, delta_y = cell_size

    # Create 1D finite difference matrix for x-direction
	off_diag_x = np.ones(Nx - 1)
	A1D_x = diags([-off_diag_x, 0, off_diag_x], [-1, 0, 1], shape=(Nx, Nx)).toarray()
	A1D_x[0, -1] = -1
	A1D_x[-1, 0] = 1

    # Create 1D finite difference matrix for y-direction
	off_diag_y = np.ones(Ny - 1)
	A1D_y = diags([-off_diag_y, 0, off_diag_y], [-1, 0, 1], shape=(Ny, Ny)).toarray()
	A1D_y[0, -1] = -1
	A1D_y[-1, 0] = 1

	#print(phi.shape)
	E_fieldy = -phi@A1D_x/(2*delta_x)
	E_fieldx = A1D_y@phi/(2*delta_y)

	global plot_electric_field
	if plot_electric_field == True:
		x,y = np.meshgrid(np.linspace(0,box_size[0],Ny),np.linspace(0,box_size[1],Nx))
		plt.quiver(x,y,E_fieldx.T,E_fieldy.T)

	E_field = np.array([E_fieldx,E_fieldy])
	return E_field

def calculateAcceleration(E_field, ij_BL, ij_BR, ij_TL, ij_TR, ij_BL_weight, ij_BR_weight, ij_TL_weight, ij_TR_weight):
	E_x_on_particles = []
	E_y_on_particles = []
	N = ij_BL.shape[1]
	
	def applyField(ij, wij):
		ExFromij = []
		EyFromij = []
		for n in range(N):
			i = ij[0, n]
			j = ij[1, n]
			weight = wij[n]
			Ex = E_field[0, i, j] * weight
			Ey = E_field[1, i, j] * weight
			ExFromij.append(Ex)
			EyFromij.append(Ey)
		return ExFromij, EyFromij
	
	E_x_BL, E_y_BL = applyField(ij_BL, ij_BL_weight)
	E_x_BR, E_y_BR = applyField(ij_BR, ij_BR_weight)
	E_x_TL, E_y_TL = applyField(ij_TL, ij_TL_weight)
	E_x_TR, E_y_TR = applyField(ij_TR, ij_TR_weight)
	
	for n in range(N):
		Ex = E_x_BL[n] + E_x_BR[n] + E_x_TL[n] + E_x_TR[n]
		Ey = E_y_BL[n] + E_y_BR[n] + E_y_TL[n] + E_y_TR[n]
		E_x_on_particles.append(Ex)
		E_y_on_particles.append(Ey)
		
	EOnParticles = np.array([E_x_on_particles, E_y_on_particles])
    
	global plot_electric_field_on_particles
	if plot_electric_field_on_particles == True:
		global particle_positions
		x,y = particle_positions
		plt.quiver(x,y,E_x_on_particles,E_y_on_particles)
	
	return EOnParticles

# Simulation parameters
N			= 5000						# Number of particles
Nc			= np.array([150,100])		# Mesh grid subdivisions
t			= 0							# Start time of simulation (s)
tEnd		= 50						# End time of simulation (s)
Nt			= 100						# Number of timesteps
dt			= (tEnd-t)/Nt				# Time step size (s)
box_size		= np.array([150,100])		# Size of domain (From the origin)
n0			= N/(box_size[0]*box_size[1])	# Average density

# Output parameters
plot_particles = True
plot_cell_markers = False
plot_number_density = False
plot_potential = True
plot_electric_field = False
plot_electric_field_on_particles = False

# Initial particle conditions
np.random.seed(42)
# Create initial plasma field
particle_positions = np.random.rand(2, N) * box_size[:, np.newaxis]

particle_velocities = np.random.rand(2, N) * 0

particle_accelerations = np.random.rand(2, N) * 0

# Initialize figure and axis
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_aspect('equal', adjustable='box')

# Main Loop Stuff
def update():
		global particle_positions
		global particle_velocities
		global particle_accelerations
		global box_size

		# 1/2 kick
		particle_velocities += particle_accelerations * dt / 2

		# Drift and applying periodic boundry conditions
		particle_positions += particle_velocities * dt
		particle_positions = np.mod(particle_positions, box_size[:, np.newaxis])

		# Find new number densities
		n, ij_BL, ij_BR, ij_TL, ij_TR, ij_BL_weight, ij_BR_weight, ij_TL_weight, ij_TR_weight = calculateNumberDensity(particle_positions)

		phi = calculatePotential(n)

		E_field = calculateElectricField(phi)

		# Update accelerations
		particle_accelerations = calculateAcceleration(E_field, ij_BL, ij_BR, ij_TL, ij_TR, ij_BL_weight, ij_BR_weight, ij_TL_weight, ij_TR_weight)

		# 1/2 kick
		particle_velocities += particle_accelerations * dt / 2
		return

def plot():
	global plot_particles
	if plot_particles == True:
		plt.scatter(particle_positions[0],particle_positions[1],c='black',s=0.5)

def animate(frame):
	ax.clear()
	plt.xlim(0-10, box_size[0]+10)	# Adjust these limits according to your data
	plt.ylim(0-10, box_size[1]+10)	# Adjust these limits according to your data
	update()
	plot()

ani = FuncAnimation(fig, animate, frames=1000, interval=10)
plt.show()
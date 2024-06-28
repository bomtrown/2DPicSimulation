import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm

from scipy.sparse import kron, eye, diags
from scipy.sparse.linalg import spsolve

# Test Change

def calculateNumberDensity(particlePositions):
	global Nc
	global boxSize
	cellSize = boxSize/Nc

	# Start by fining the coordinates of the cell in the cell coordinates (i,j) -> (i+1,j+1) that contain each particle in order bl, br, tl, tr
	ijBL = np.floor(particlePositions / cellSize[:, np.newaxis]).astype(int)
	ijBR = np.array([ijBL[0]+1,ijBL[1]])
	ijTL = np.array([ijBL[0],ijBL[1]+1])
	ijTR = np.array([ijBL[0]+1,ijBL[1]+1])

	# The displacement of each particle from the ij to its bottom left in ij coordinates
	dx, dy = particlePositions / cellSize[:, np.newaxis] - ijBL

	# Finding the weight each particle has on each of the corners of the cell that contains it
	wijBL	= (1-dx) * (1-dy)
	wijBR	= dx * (1-dy)
	wijTL	= (1-dx) * dy
	wijTR 	= dx * dy

	global plotCellMarkers
	# Define the colormap
	cmap = cm.get_cmap('Grays')
	if plotCellMarkers == True:
		plt.scatter(ijBL[0]*cellSize[0],ijBL[1]*cellSize[1],marker='+',c=cmap(wijBL))
		plt.scatter(ijBR[0]*cellSize[0],ijBR[1]*cellSize[1],marker='+',c=cmap(wijBR))
		plt.scatter(ijTL[0]*cellSize[0],ijTL[1]*cellSize[1],marker='+',c=cmap(wijTL))
		plt.scatter(ijTR[0]*cellSize[0],ijTR[1]*cellSize[1],marker='+',c=cmap(wijTR))

	# Create a density array and add the weights from every particle
	n = np.zeros((Nc[0]+1,Nc[1]+1))
	# Function to add weights to grid
	def addToGrid(ij, wij):
		for (i, j), w in zip(zip(ij[0], ij[1]), wij):
				n[i, j] += w
    
    # Add weights from each direction
	addToGrid(ijBL, wijBL)
	addToGrid(ijBR, wijBR)
	addToGrid(ijTL, wijTL)
	addToGrid(ijTR, wijTR)

	global plotNumberDensity
	if plotNumberDensity == True:
		#scaled_grid = np.kron(n, np.ones((5, 5)))

		# Plot the shifted grid
		plt.imshow((n.T)/5%1, origin='lower',extent=[-cellSize[0]/2,boxSize[0]+cellSize[0]/2,-cellSize[1]/2,boxSize[1]+cellSize[1]/2], cmap='jet', interpolation='bilinear',vmax=0.5)

	return n, ijBL, ijBR, ijTL, ijTR, wijBL, wijBR, wijTL, wijTR

def calculatePotential(n):
	"""
    Calculate the potential matrix phi given the number density matrix n and cell sizes.
    
    Parameters:
    n (numpy.ndarray): 2D array representing the number density matrix.
    cellSize (list or tuple): Array containing the cell sizes [Delta x, Delta y].
    
    Returns:
    numpy.ndarray: 2D array representing the potential matrix phi.
    """
	global Nc
	global boxSize
	cellSize = boxSize/Nc

	global N
	n0 = N/(boxSize[0]*boxSize[1])

	n = n - n0

    # Extract grid sizes and cell sizes
	Ny, Nx = n.shape
	delta_x, delta_y = cellSize

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

	# Potential object in the middle
	#for i in range(-10,10):
	#	for j in range(-20,20):
	#		phi[int(Ny/2)+i,int(Nx/2)+j] = phi[int(Ny/2)+i,int(Nx/2)+j]-1000*(1-(i/10)**2)*(i/10+1)**3*(1-(j/20)**2)

	# Periodic potential
	#for i in range(Nc[0]):
	#	for j in range(Nc[1]):
	#		phi[i,j] = phi[i,j]-10*(np.sin(i*np.pi/4)*np.sin(j*np.pi/4))**10

	#print(np.mean(phi))
	#print(np.min(phi))
	#print(np.max(phi))

	global plotPotential
	if plotPotential == True:
		if plotPotential == True:
			# Plot the shifted grid
			plt.imshow((phi.T), origin='lower',extent=[-cellSize[0]/2,boxSize[0]+cellSize[0]/2,-cellSize[1]/2,boxSize[1]+cellSize[1]/2], cmap='jet', interpolation='bilinear')

	return phi

def calculateElectricField(phi):
	global Nc
	global boxSize
	cellSize = boxSize/Nc

    # Extract grid sizes and cell sizes
	Ny, Nx = phi.shape
	delta_x, delta_y = cellSize

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
	EFieldy = -phi@A1D_x/(2*delta_x)
	EFieldx = A1D_y@phi/(2*delta_y)

	global plotElectricField
	if plotElectricField == True:
		x,y = np.meshgrid(np.linspace(0,boxSize[0],Ny),np.linspace(0,boxSize[1],Nx))
		plt.quiver(x,y,EFieldx.T,EFieldy.T)

	# Adding external field
	EFieldx = EFieldx - 0.1

	EField = np.array([EFieldx,EFieldy])
	return EField

def calculateAcceleration(EField, ijBL, ijBR, ijTL, ijTR, wijBL, wijBR, wijTL, wijTR):
	ExOnParticles = []
	EyOnParticles = []
	N = ijBL.shape[1]
	
	def applyField(ij, wij):
		ExFromij = []
		EyFromij = []
		for n in range(N):
			i = ij[0, n]
			j = ij[1, n]
			weight = wij[n]
			Ex = EField[0, i, j] * weight
			Ey = EField[1, i, j] * weight
			ExFromij.append(Ex)
			EyFromij.append(Ey)
		return ExFromij, EyFromij
	
	ExBL, EyBL = applyField(ijBL, wijBL)
	ExBR, EyBR = applyField(ijBR, wijBR)
	ExTL, EyTL = applyField(ijTL, wijTL)
	ExTR, EyTR = applyField(ijTR, wijTR)
	
	for n in range(N):
		Ex = ExBL[n] + ExBR[n] + ExTL[n] + ExTR[n]
		Ey = EyBL[n] + EyBR[n] + EyTL[n] + EyTR[n]
		ExOnParticles.append(Ex)
		EyOnParticles.append(Ey)
		
	EOnParticles = np.array([ExOnParticles, EyOnParticles])
    
	global plotElectricFieldOnParticles
	if plotElectricFieldOnParticles == True:
		global particlePositions
		x,y = particlePositions
		plt.quiver(x,y,ExOnParticles,EyOnParticles)
	
	return EOnParticles

# Simulation parameters
N			= 5000						# Number of particles
Nc			= np.array([150,100])		# Mesh grid subdivisions
t			= 0							# Start time of simulation (s)
tEnd		= 50						# End time of simulation (s)
Nt			= 1000					# Number of timesteps
dt			= (tEnd-t)/Nt				# Time step size (s)
boxSize		= np.array([150,100])		# Size of domain (From the origin)
n0			= N/(boxSize[0]*boxSize[1])	# Average density

# Output parameters
plotParticles = True
plotCellMarkers = False
plotNumberDensity = True
plotPotential = False
plotElectricField = False
plotElectricFieldOnParticles = False

# Initial particle conditions
np.random.seed(42)
# Create initial plasma field
particlePositions = np.random.rand(2, N) * boxSize[:, np.newaxis]
particlePositions = [particlePositions[0]*0.01+0.5*boxSize[0],particlePositions[1]*0.0+0.5*boxSize[1]]

#particleVelocities = np.random.rand(2, N) * 0
particleVelocities = np.squeeze(np.array([0*np.ones(N),np.zeros(N)]))
particleAccelerations = np.random.rand(2, N) * 0
# Initialise matricies


# Initialize figure and axis
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_aspect('equal', adjustable='box')

# Main Loop Stuff
def update():
		global particlePositions
		global particleVelocities
		global particleAccelerations
		global boxSize

		# 1/2 kick
		particleVelocities += particleAccelerations * dt / 2

		# Drift and applying periodic boundry conditions
		particlePositions += particleVelocities * dt
		particlePositions = np.mod(particlePositions, boxSize[:, np.newaxis])

		# Find new number densities
		n, ijBL, ijBR, ijTL, ijTR, wijBL, wijBR, wijTL, wijTR = calculateNumberDensity(particlePositions)

		phi = calculatePotential(n)

		EField = calculateElectricField(phi)

		# Update accelerations
		particleAccelerations = calculateAcceleration(EField, ijBL, ijBR, ijTL, ijTR, wijBL, wijBR, wijTL, wijTR)

		# 1/2 kick
		particleVelocities += particleAccelerations * dt / 2
		return

def plot():
	global plotParticles
	if plotParticles == True:
		plt.scatter(particlePositions[0],particlePositions[1],c='black',s=0.5)

def animate(frame):
	ax.clear()
	plt.xlim(0-10, boxSize[0]+10)	# Adjust these limits according to your data
	plt.ylim(0-10, boxSize[1]+10)	# Adjust these limits according to your data
	update()
	plot()

ani = FuncAnimation(fig, animate, frames=1000, interval=10)
plt.show()
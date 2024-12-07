
#####################################################################
#####                   IMPORTS & FUNCTIONS                     #####
#####################################################################
import numpy as np
from scipy.fftpack import fftfreq
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



###-----Lambda-Omega System
def lambdaFun(A2):
    return 1 - A2

def omegaFun(A2):
    return -beta*A2



###-----REACTION DIFFUSION SYSTEM
def reaction_diffusion_fft(t, UV):
    #Define Wave Numbers for Fourier Transforms
    kx = 2 * np.pi * fftfreq(n, x[1] - x[0])
    ky = 2 * np.pi * fftfreq(n, y[1] - y[0])
    KX, KY = np.meshgrid(kx, ky)
    K2 = -(KX**2 + KY**2)                       #Laplacian operator in Fourier Domain
    
    U, V = np.split(UV, 2)
    U = U.reshape((n, n))
    V = V.reshape((n, n))

    A2 = U**2 + V**2                #Compute A^2
    A = np.sqrt(A2)                 #Compute A
    lambda_A = lambdaFun(A)         #Lambda Reaction Term
    omega_A = omegaFun(A)           #Omega Reaction Term

    U_hat = fft2(U)                 #transforming U from real space to Fourier space
    V_hat = fft2(V)                 #transforming V from real space to Fourier space

    U_diff = ifft2(D1 * K2 * U_hat).real    #Diffusion terms in Fourier Domain
    V_diff = ifft2(D2 * K2 * V_hat).real    #Diffusion terms in Fourier Domain

    dUdt = lambda_A*U - omega_A*V + U_diff
    dVdt = omega_A*U + lambda_A*V + V_diff
    dUdt_dVdt = np.hstack([dUdt.ravel(), dVdt.ravel()])
    
    return dUdt_dVdt



###-----CHEBYCHEV REACTION DIFFUSION SYSTEM
def reaction_diffusion_cheb(t,UV):
    U, V = np.split(UV,2)
 
    A2 = U**2 + V**2                #Compute A^2
    #A2 = np.dot(U,U) + np.dot(V,V)
    #A2 = U.reshape((n,n)) + V.reshape((n,n))
    #A = np.sqrt(A2)                 #Compute A
    lambda_A = lambdaFun(A2)         #Lambda Reaction Term
    omega_A = omegaFun(A2)           #Omega Reaction Term

    U_diff = D1 * DN2 @ U
    V_diff = D2 * DN2 @ V

    dUdt = lambda_A * U - omega_A * V + U_diff
    dVdt = omega_A * U + lambda_A * V + V_diff
    dUdt_dVdt = np.hstack([dUdt.ravel(), dVdt.ravel()])
    
    return dUdt_dVdt



###-----CHEBYCHEV DERIVATIVE MATRIX
def cheb(N):
    if N==0:
        D = 0
        x = 1
        print("Chebychev can't operate with zero (N=0) discretized points.")
    else:
        n = np.arange(0, N+1)
        x = np.cos(np.pi*n/N).reshape(N+1,1)
        c = (np.hstack(([2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
        X = np.tile(x,(1,N+1))
        dX = X - X.T
        D = np.dot(c, 1./c.T) / (dX+np.eye(N+1))
        D -= np.diag(np.sum(D.T, axis=0))
        x = x.reshape(N+1)
        
        return D, x





#####################################################################
#####           FFT  -  PERIODIC BOUNDARY CONDITIONS            #####
#####################################################################
#COMPUTATIONAL DOMAIN
xStart = -10
xStop = 10
yStart = xStart
yStop = xStop
N = 64



x = np.linspace(xStart, xStop, N)
y = np.linspace(yStart, yStop, N)
X_fft, Y_fft = np.meshgrid(x,y)
n = len(x)



#REACTION PARAMETERSs
beta = 1.0
D1 = 0.1
D2 = 0.1



tStart = 0
tEnd = 4.0
dt = 0.5
tStop = tEnd + dt
tSpan = (tStart, tEnd)
tEval = np.arange(tStart, tStop, dt)



#INITIAL CONDITIONS
numSpirals = 1
A = np.sqrt(X_fft**2 + Y_fft**2)
theta = np.angle(X_fft + 1j*Y_fft)
U0 = np.tanh(A) * np.cos(numSpirals * theta - A)
V0 = np.tanh(A) * np.sin(numSpirals * theta - A)
U0_fft = U0
V0_fft = V0
    


###-----SOLVING THE PROBLEM
UV0 = np.hstack([U0.ravel(), V0.ravel()])
print("UV0_fft :", UV0)
sol = solve_ivp(reaction_diffusion_fft, tSpan, UV0, t_eval=tEval, method='RK45')

U_sol_fft = sol.y[:n*n, :].reshape(n,n,-1)
V_sol_fft = sol.y[n*n:, :].reshape(n,n,-1)

U_sol_V_sol_fft = sol.y
A1 = U_sol_V_sol_fft
np.save('A1.npy', A1)
print("A1 shape", A1.shape)





#####################################################################
#####               CHEBYCHEV  -  NO FLUX BOUNDARIES            #####
#####################################################################

###-----COMPUTATIONAL DOMAIN
xStart = -10
xStop = 10
yStart = xStart
yStop = xStop
N = 30



###-----ASSEMBLING THE CHEBY LAPLACIAN
DN, cheb_x = cheb(N)
cheb_y = cheb_x
X, Y = np.meshgrid(cheb_x,cheb_y)
n = len(cheb_x)
DN2 = np.dot(DN,DN)
#print(cheb_x)



###-----BOUNDARY CONDITIONS
DN2[0,:] = np.zeros((1,N+1))
DN2[-1,:] = np.zeros((1,N+1))
#print(DN2[0,:])

I = np.eye(n)
DN2 = np.kron(I,DN2) + np.kron(DN2,I)



###-----REACTION PARAMETERS
beta = 1.0
D1 = 0.1
D2 = 0.1


tStart = 0
tEnd = 4.0
dt = 0.5
tStop = tEnd + dt
tSpan = (tStart, tEnd)
tEval = np.arange(tStart, tStop, dt)



###-----INITIAL CONDITIONS
numSpirals = 1
A = np.sqrt(X**2 + Y**2)
theta = np.angle(X + 1j*Y)
U0 = np.tanh(A) * np.cos(numSpirals * theta - A)
V0 = np.tanh(A) * np.sin(numSpirals * theta - A)
U0_cheb = U0
V0_cheb = V0

    

###-----SOLVING THE PROBLEM
UV0 = np.hstack([U0.ravel(), V0.ravel()])
tol=1e-6
print("UV0 shape and size: ", UV0.shape, UV0.size)
sol = solve_ivp(reaction_diffusion_cheb, tSpan, UV0, t_eval=tEval, method='RK45',atol=tol,rtol=tol)

U_sol_cheb = sol.y[:n*n, :].reshape(n,n,-1)
V_sol_cheb = sol.y[n*n:, :].reshape(n,n,-1)

U_sol_V_sol_cheb = sol.y
A2 = U_sol_V_sol_cheb
np.save('A2.npy', A2)
print("A2 shape", A2.shape)



###-----MAPPING BACK TO REAL SPACE FOR VISUALIZATION
x = 0.5 * (xStop - xStart) * (cheb_x + 1) + xStart  # Maps [-1, 1] to [xStart, xStop]
y = 0.5 * (yStop - yStart) * (cheb_y + 1) + yStart

x = np.linspace(xStart, xStop, len(cheb_x))
y = np.linspace(yStart, yStop, len(cheb_y))

X_cheb, Y_cheb = np.meshgrid(x,y)




#####################################################################
#####                       VISUALIZATION                       #####
#####################################################################
numTimePoints = len(tSpan)
numCols = 4                 #number of columns in the subplot grid
numRows = 2 * int(np.ceil(numTimePoints / numCols))

plt.figure(figsize=(12, 8))

plt.subplot(numRows, numCols, 1)
plt.contourf(X_fft, Y_fft, U0_fft, levels=50, cmap='viridis')
plt.title(f"U at t={tSpan[0]:.1f}")
plt.ylabel("FFT")
plt.colorbar()

plt.subplot(numRows, numCols, 2)
plt.contourf(X_fft, Y_fft, U_sol_fft[:, :, -1], levels=50, cmap='viridis')
plt.title(f"U at t={tSpan[-1]:.1f}")
plt.colorbar()

plt.subplot(numRows, numCols, 3)
plt.contourf(X_fft, Y_fft, V0_fft, levels=50, cmap='plasma')
plt.title(f"U at t={tSpan[0]:.1f}")
plt.colorbar()

plt.subplot(numRows, numCols, 4)
plt.contourf(X_fft, Y_fft, V_sol_fft[:, :, -1], levels=50, cmap='plasma')
plt.title(f"V at t={tSpan[-1]:.1f}")
plt.colorbar()

plt.subplot(numRows, numCols, 5)
plt.contourf(X_cheb, Y_cheb, U0_cheb, levels=50, cmap='viridis')
plt.title(f"U at t={tSpan[0]:.1f}")
plt.colorbar()
plt.ylabel("ChebyChev")

plt.subplot(numRows, numCols, 6)
plt.contourf(X_cheb, Y_cheb, U_sol_cheb[:, :, -1], levels=50, cmap='viridis')
plt.title(f"V at t={tSpan[-1]:.1f}")
plt.colorbar()

plt.subplot(numRows, numCols, 7)
plt.contourf(X_cheb, Y_cheb, V0_cheb, levels=50, cmap='plasma')
plt.title(f"U at t={tSpan[0]:.1f}")
plt.colorbar()

plt.subplot(numRows, numCols, 8)
plt.contourf(X_cheb, Y_cheb, V_sol_cheb[:, :, -1], levels=50, cmap='plasma')
plt.title(f"V at t={tSpan[-1]:.1f}")
plt.colorbar()

plt.tight_layout()
plt.savefig("swirl_fft_cheb_comparison.png")
plt.show()






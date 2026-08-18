import numpy as np
from scipy.signal import butter,tf2ss,StateSpace,dbode
import matplotlib.pyplot as plt

from dsm.initDesign import optIIR,computeMSE

# =============================================================================
# Initial DSM Design from MATLAB
# =============================================================================
order = 4
gamma = 1.5
cutoff = np.pi / 32
ts = 1

# Define Output filter
b, a = butter(order, cutoff / np.pi, btype='low',analog=False)
Ah, Bh, Ch, Dh = tf2ss(b, a)

# Run optimization
Ropt = optIIR(order,gamma,Ah,Bh,Ch,Dh,ts)

assert Ropt!=None, "Optimization returned None"
print(Ropt)

# Frequency response
Href = StateSpace(Ah,Bh,Ch,Dh,dt=ts)
frq = np.logspace(-4,np.log10(np.pi),10000)
_,Hmag,_ = dbode(Href,w=frq)
_,Rmag,_ = dbode(Ropt,w=frq)

# MSE computation
MSE = computeMSE(Href,Ropt)
print(f"Initial MSE = {MSE:.4f} dB")

# =============================================================================
# Apply FWBT
# =============================================================================
from fwbt.problem import FWBTProblem
reduc_order = 2

# Model reduction through FWBT
prob = FWBTProblem(Ropt,order_target=reduc_order,W_o=(Ah,Bh,Ch,Dh),discrete=True)
result = prob.solve()
R_reduc = result.to_statespace()

# Frequency response + MSE
_,R_reduc_mag,_ = dbode(R_reduc,w=frq)
MSE_reduc = computeMSE(Href,R_reduc)
print(f"Reduced MSE = {MSE_reduc:.4f} dB")

# =============================================================================
# Plit graphs
# =============================================================================

plt.figure
plt.semilogx(frq,Hmag,label="Hz order 4")
plt.semilogx(frq,Rmag,label="IIR design order 4")
plt.semilogx(frq,R_reduc_mag,label="Reduced order 2")
plt.title("Magnitude Response")
plt.legend()
plt.xscale('log')
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xlabel('Normalized Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(frq[0],frq[-1])
plt.ylim(-120,10)

plt.show()


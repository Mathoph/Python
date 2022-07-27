import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt

# field of each stator magnet (flat rectangular shape) on the (external) stator

def F(X,Y,L):
    return (Y+L/2)/( X**2 + (Y+L/2)**2 ) - (Y-L/2)/( X**2 + (Y-L/2)**2 ) 

def G(X,Y,L):
    return X/( X**2 + (Y-L/2)**2 ) - X/( X**2 + (Y+L/2)**2 ) 

def dFdx(X,Y,L):
    return (2*(Y-L/2)*X)/(X**2+(Y-L/2)**2)**2 - (2*(Y+L/2)*X)/(X**2+(Y+L/2)**2)**2

def dFdy(X,Y,L):
    return 1/((Y+L/2)**2+X**2) - (2*(Y+L/2)**2)/((Y+L/2)**2+X**2)**2 - 1/((Y-L/2)**2+X**2) - (2*(L/2-Y)*(Y-L/2))/((Y-L/2)**2+X**2)**2

def dGdx(X,Y,L):
    return -1/(X**2+(Y+L/2)**2) + (2*X**2)/(X**2+(Y+L/2)**2)**2 + 1/(X**2+(Y-L/2)**2) - (2*X**2)/(X**2+(Y-L/2)**2)**2

def dGdy(X,Y,L):
    return (2*X*(Y+L/2))/((Y+L/2)**2+X**2)**2 - (2*X*(Y-L/2))/((Y-L/2)**2+X**2)**2

# rotary angle (angular position of the 1st rotor magnet)
phi = np.linspace(0,2*pi,300)

# magnet parameters ############################################################

L = 10 # stator magnet length (mm)
g = 25/360*2*pi # tilt angle
Rs = 100 # distance (mm) of stator magnet center from origin
R = 60 # distance (mm) of rotor magnet center from origin
ns = 7 # number of stator magnets
n = 12 # number of rotor magnets

# rotor magnets angular positions
i = np.arange(1,n+1)
b = 2*pi*(i-1)/n # linear sequence

# stator magnets angular positions
j = np.arange(1,ns+1)
#ap = 2*pi*(j-1)/ns # linear sequence
ap = 2*pi/(ns*(ns-1))*(j-1)**2 # quadratic sequence, if you want

# plot magnets angle positions
plt.scatter(i,b*360/2/pi,label = 'rotor magnet angle sequence') 
plt.scatter(j,ap*360/2/pi,color='red',label = 'stator magnet angle sequence')
plt.xlabel('magnet index')
plt.ylabel('angle value (°)')
plt.legend()

# plot magnets magnetizations
Xs = Rs*cos(ap); Ys = Rs*sin(ap); Msx = cos(ap); Msy = sin(ap) # coordinates and orientation of stator magnets
Xr = R*cos(b); Yr = R*sin(b); Mrx = cos(b+g); Mry = sin(b+g) # coordinates and orientation of rotor magnets
plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.quiver(Xs,Ys,Msx,Msy,color='red',label = 'stator magnets')
plt.quiver(Xr,Yr,Mrx,Mry,color='blue',label = 'rotor magnets')
plt.xlim(-1.3*Rs,1.3*Rs)
plt.ylim(-1.3*Rs,1.3*Rs)
plt.ylabel('y (mm)')
plt.xlabel('x (mm)') 
plt.legend()
plt.axhline(y = 0,xmin = 0,xmax = 1,color = 'grey',linestyle = '--')
plt.axvline(x = 0,ymin = 0,ymax = 1,color = 'grey',linestyle = '--')

# plot stator magnets field

x = np.linspace(-1.3*Rs,1.3*Rs,30)
y = np.linspace(-1.3*Rs,1.3*Rs,30)
X, Y = np.meshgrid(x, y)
Hx = np.zeros(X.shape); Hy = np.zeros(X.shape)

for p in range(1,ns+1):
        Xt = Rs*cos(ap[p-1]); Yt = Rs*sin(ap[p-1]); # stator magnet center
        Xr = cos(ap[p-1])*(X-Xt) + sin(ap[p-1])*(Y-Yt) # anti-clockwise rotation of the magnet
        Yr = -sin(ap[p-1])*(X-Xt) + cos(ap[p-1])*(Y-Yt) # anti-clockwise rotation of the magnet      
        Hx = Hx + cos(ap[p-1])*F(Xr,Yr,L) - sin(ap[p-1])*G(Xr,Yr,L) 
        Hy = Hy + sin(ap[p-1])*F(Xr,Yr,L) + cos(ap[p-1])*G(Xr,Yr,L)

H = sqrt(Hx**2 + Hy**2)
Hx = Hx/np.max(H)
Hy = Hy/np.max(H)
plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
CS = plt.contourf(X,Y,H,levels=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar()
plt.quiver(X,Y,Hx,Hy)
plt.xlabel('x/L')
plt.ylabel('y/L')
plt.axhline(y = 0,xmin = 0,xmax = 1,color = 'grey',linestyle = '--')
plt.axvline(x = 0,ymin = 0,ymax = 1,color = 'grey',linestyle = '--')

# Torque computation ###########################################################

C = np.zeros(phi.shape) 

for k in range(1,n+1): # rotor
    xk = R*cos(b[k-1]+phi); yk = R*sin(b[k-1]+phi) # rotor magnets positions
    ck = cos(b[k-1]+phi+g); sk = sin(b[k-1]+phi+g) # rotor magnets magnetizations
    for p in range(1,ns+1): # stator
        xp = Rs*cos(ap[p-1]); yp = Rs*sin(ap[p-1]) # stator magnets positions     
        cp = cos(ap[p-1]); sp = sin(ap[p-1]) 
        xpk =  cp*(xk - xp) + sp*(yk - yp)
        ypk = -sp*(xk - xp) + cp*(yk - yp)
        # field computation
        hx = cp*F(xpk,ypk,L) - sp*G(xpk,ypk,L)
        hy = sp*F(xpk,ypk,L) + cp*G(xpk,ypk,L)
        dxhx = cp*( cp*dFdx(xpk,ypk,L) - sp*dFdy(xpk,ypk,L) ) - sp*( cp*dGdx(xpk,ypk,L) - sp*dGdy(xpk,ypk,L) )
        dyhx = cp*( sp*dFdx(xpk,ypk,L) + cp*dFdy(xpk,ypk,L) ) - sp*( sp*dGdx(xpk,ypk,L) + cp*dGdy(xpk,ypk,L) )
        dxhy = sp*( cp*dFdx(xpk,ypk,L) - sp*dFdy(xpk,ypk,L) ) + cp*( cp*dGdx(xpk,ypk,L) - sp*dGdy(xpk,ypk,L) )
        dyhy = sp*( sp*dFdx(xpk,ypk,L) + cp*dFdy(xpk,ypk,L) ) + cp*( sp*dGdx(xpk,ypk,L) + cp*dGdy(xpk,ypk,L) )
        # torque computation           
        C = C + ck*( hy + yk*dyhy + xk*dyhx ) - sk*( hx + xk*dxhx + yk*dxhy )

# plot torque
plt.figure()
plt.plot(phi/2/pi*360,C)
plt.ylabel('Torque (N.m)')
plt.xlabel('angle $\phi$ (°)')
plt.ylim(-1.1*np.max(abs(C)),1.1*np.max(abs(C)))
plt.show()

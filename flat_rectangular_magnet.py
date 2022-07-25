from numpy import pi, cos, sin, sqrt, exp, linspace, meshgrid
import matplotlib.pyplot as plt

def F(X,Y,L):
    return (Y+L/2)/( X**2 + (Y+L/2)**2 ) - (Y-L/2)/( X**2 + (Y-L/2)**2 ) 

def G(X,Y,L):
    return X/( X**2 + (Y-L/2)**2 ) - X/( X**2 + (Y+L/2)**2 ) 

# 
L = 10 # hauteur aimant selon l'axe y
x = linspace(-3*L,3*L,20)
y = linspace(-3*L,3*L,20)
X, Y = meshgrid(x, y)

# déf champ magnétique
Hx = F(X,Y,L)
Hy = G(X,Y,L)
H = sqrt(Hx**2 + Hy**2)

# traçage courbe
plt.gca().set_aspect('equal', adjustable='box')
CS = plt.contourf(X,Y,H,levels=[0.0,0.2,0.4,0.6,0.8,1])
plt.colorbar()
plt.quiver(X,Y,Hx,Hy)
plt.xlabel('x/L')
plt.ylabel('y/L')
plt.axhline(y = 0,xmin = 0,xmax = 1,color = 'grey',linestyle = '--')
plt.axvline(x = 0,ymin = 0,ymax = 1,color = 'grey',linestyle = '--')
plt.show()

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.signal import convolve
from PIL import Image
from scipy.linalg import inv
from numpy.linalg import eig
from matplotlib import cm

plt.close('all')

#%%

def GaussMultVar(X, Y, M1, M2):
    
    S = np.asarray([X, Y])
    S = np.moveaxis(S, 0, 2) - M1
    
    A = inv(M2)
    
    B = np.einsum('ij, lmj -> ilm', A, S)
    C = np.einsum('ijk, kij -> ij', S, B)

    g = np.exp( - 0.5 * C )
    
    return g


def gauss2d(X, Y, mu, sigma):
    
    R = np.sqrt(X**2 + Y**2)
    g = np.exp( -(R - mu)**2/(2*sigma**2) )
    
    return g

#%%

im = Image.open('STED_beam.tif')
im = np.asarray(im)

sz = im.shape

plt.figure()
plt.imshow(im)

#%%

pxsize = 3.45 #um

y = pxsize * ( np.arange(sz[0]) - sz[0]//2 )
x = pxsize * ( np.arange(sz[1]) - sz[1]//2 )

X, Y = np.meshgrid(x, y) 
mu, sigma = 0, 90

K = gauss2d(X, Y, mu, sigma)
K /= np.sum(K)

plt.figure()
plt.imshow(K)

#%%

im2 = convolve(im, K, mode = 'same')

plt.figure()
plt.imshow(im2)

#%%

fit_model = lambda xdata, a, b, c, d, e, f: f * GaussMultVar(xdata[0].reshape(sz), xdata[1].reshape(sz), np.asarray([a, b]), np.asarray([[c, d], [d, e]])).ravel()

p0 = [ 0, 0, 800, 0, 800, 100 ]

xdata = np.vstack( ( X.ravel(), Y.ravel() ) ) 

popt, pcov = opt.curve_fit(fit_model, xdata, im2.ravel(), p0)

plt.figure()
plt.imshow( fit_model(xdata, *popt).reshape(sz) )

Sigma = np.asarray( [[popt[2], popt[3]], [popt[3], popt[4]]] )

Sdiag = np.diag( eig(Sigma)[0] )

D4sigma = 4*np.sqrt( Sdiag )/1e3

print( D4sigma  )

# fig = plt.figure()
# ax = fig.gca(projection ='3d')
# ax.plot_surface( X, Y, fit_model(xdata, *popt).reshape(sz), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.scatter(X,Y,im2)
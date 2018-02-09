from pylab import *
from scipy import *
from scipy.integrate import quad
from scipy import interpolate
from scipy.special import jn
from scipy.misc import derivative
from scipy.integrate import nquad
from scipy.special import jn_zeros
from emcee.utils import MPIPool

data_mean = loadtxt('fulle_bins2D_cross_jk_final.dat')[:,5].reshape(25,-1).T
rp_bins = loadtxt('fulle_bins2D_cross_jk_final.dat')[:,0].reshape(25,-1).T
pi_bins = loadtxt('fulle_bins2D_cross_jk_final.dat')[:,1].reshape(25,-1).T


zmin,zmax = 0.16, 0.36
zcenter,dndz = load('dndz_lowz.npy')
pz = interpolate.interp1d(zcenter,dndz,bounds_error=0,fill_value=0.)
zarr=linspace(0.1,0.4,1000)
pz_test = pz(zarr)


## cosmology WMAP9
h = 0.7
H0 = h*100
ob = 0.046
oc = 0.236
om = ob+oc
ol = 1-om#0.718
ns = 0.9646
s8 = 0.817


### constants and small functions
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s^2

H = lambda z: H0*sqrt(om*(1+z)**3+ol)
H_inv = lambda z:  1/(H0*sqrt(om*(1+z)**3+ol))
Hcgs = lambda z: H(z)*3.24e-20
DC = lambda z: c*quad(H_inv, 0, z)[0]
W_fcn = lambda z: (pz(z) / DC(z))**2 / c * H(z) # dchi/dz = c/H
rho_cz = lambda z: 0.375*Hcgs(z)**2/pi/Gnewton

zarr1=linspace(0,1,1001)
Wnorm = quad(W_fcn, zmin, zmax) [0]
W_arr = array([W_fcn(iz)/Wnorm for iz in zarr])
W = interpolate.interp1d(zarr,W_arr,bounds_error=0,fill_value=0.)


## interpolate Pk
Ptable = genfromtxt('P_delta_Hinshaw')
aa = array([1/1.05**i for i in arange(33)])
zz = 1.0/aa-1 # redshifts
kk = Ptable.T[0] ## Mpc/h
iZ, iK = meshgrid(zz,kk)
Z, K = iZ.flatten(), iK.flatten()
Pk = Ptable[:,1:34].flatten()

### interpolate on actual k and P, without the h
Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K*h, Z]).T, 2.0*pi**2*Pk/(K*h)**3)
Pmatter = lambda k, z: Pmatter_interp (k, z)

## growth D(a)
zarr1=linspace(0,1,1001)
az = lambda z: 1.0/(1.0+z)
za = lambda a: 1.0/a-1
D_fcn = lambda a: H(za(a))/H0 * quad(lambda a: (om/a+ol*a**2)**(-1.5), 0, a)[0]
D1 = D_fcn(1)
D_arr = array([D_fcn (az(iz)) for iz in zarr1])/D1

###### logrithmic growth f=dln(D)/dln(a)
###### first do dD/da then x a/D
dnda_arr = array([derivative(D_fcn, az(iz), dx=1e-5) for iz in zarr1])
f_arr = az(zarr1)/D_arr * dnda_arr


Dinterp = interpolate.interp1d(zarr1,D_arr,bounds_error=0,fill_value=0.)
finterp = interpolate.interp1d(zarr1,f_arr,bounds_error=0,fill_value=0.)


C1rhoc = 0.0134 #C1*rho_crit
bD = 1.77 #LOWZ measurement

kabs = lambda kz, kp: sqrt(kz**2+kp**2)
Ai=4.4
const = Ai*bD*C1rhoc*om/2.0/pi**2

############ use fixed z ###########
iz=0.3
ifinterp = finterp(iz)
iWD = W(iz)/Dinterp(iz)
###################################

def xi_gp(kz,kp,z,rp,PI): 
    #print kz,kp,z,rp,PI
    #kz,kp=10**logkz,10**logkp
    k=kabs(kz,kp)
    mu2=(kz/k)**2
    out=(1-mu2)*kp*Pmatter(k,z)*jn(2,kp*rp)*(1.0+ifinterp/bD*mu2)
    out*=cos(kz*PI)
    #out*=W(z)/Dinterp(z)
    out*=iWD
    return out*const

#irp=10
#iPI=10

def genxi(rpPi, z=iz):
    irp, iPI = rpPi
    print irp, iPI
#    J2zeros = jn_zeros(2,100)/irp
#    opts1={'points':J2zeros}
    xi_test=nquad(xi_gp, [[1e-3, 10], [1e-3, 10]], args=(z, irp, iPI))#,opts=[{}, opts1])
    return xi_test[0]

rp_arr = linspace(0.5, 60.5, 21)
Pi_arr = linspace(-60, 60, 25)
rppi_arr = [[irp, ipi] for irp in rp_arr for ipi in Pi_arr]
            
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
out = pool.map(genxi, rppi_arr)
    
save('out',out)
print 'done done done'
pool.close()

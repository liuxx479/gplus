from pylab import *
from scipy import *
from scipy.integrate import quad
from scipy import interpolate
from scipy.special import jn
from scipy.misc import derivative
from scipy.integrate import nquad
from scipy.special import jn_zeros
from emcee.utils import MPIPool
import os
import camb
from camb import model, initialpower

data_mean = loadtxt('fulle_bins2D_cross_jk_final.dat')[:,5].reshape(25,-1).T
rp_bins = loadtxt('fulle_bins2D_cross_jk_final.dat')[:,0].reshape(25,-1).T
pi_bins = loadtxt('fulle_bins2D_cross_jk_final.dat')[:,1].reshape(25,-1).T

zmin,zmax = 0.15, 0.37
zcenter,dndz = load('dndz_lowz.npy')
pz = interpolate.interp1d(zcenter,dndz,bounds_error=0,fill_value=0.)
zarr=linspace(0.1,0.4,1000)
#pz_test = pz(zarr)

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
#Ptable = genfromtxt('P_delta_Hinshaw')
#aa = array([1/1.05**i for i in arange(33)])
#zz = 1.0/aa-1 # redshifts
#kk = Ptable.T[0] ## Mpc/h
#iZ, iK = meshgrid(zz,kk)
#Z, K = iZ.flatten(), iK.flatten()
#Pk = Ptable[:,1:34].flatten()

### interpolate on actual k and P, without the h
#Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K*h, Z]).T, 2.0*pi**2*Pk/(K*h)**3)
#Pmatter = lambda k, z: Pmatter_interp (k, z)

z_Pk = linspace(zmin,zmax,51)
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ob*h**2, omch2=oc*h**2)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(redshifts=z_Pk, kmax=10.0)
results = camb.get_results(pars)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100.0, npoints = 401)

iK, iZ = meshgrid(kh_nonlin,z_Pk)
Z, K = iZ.flatten(), iK.flatten()
Pk = array(pk_nonlin).flatten()
Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([log10(K), Z]).T, Pk, fill_value=0.)
Pmatter = lambda k, z: Pmatter_interp (log10(k), z)

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
bD = 1.77 #LOWZ measurement, SM15 paper

kabs = lambda kz, kp: sqrt(kz**2+kp**2)
Ai=4.4
const = Ai*bD*C1rhoc*om/2.0/pi**2

############ use fixed z ###########
iz=0.3
ifinterp = finterp(iz)
iWD = W(iz)/Dinterp(iz)
###################################

#def xi_gp(kz,kp,z,rp,PI): 
    ##print kz,kp,z,rp,PI
    ##kz,kp=10**logkz,10**logkp
    #k=kabs(kz,kp)
    #mu2=(kz/k)**2
    #out=(1-mu2)*kp*Pmatter(k,z)*jn(2,kp*rp)*(1.0+ifinterp/bD*mu2)
    #out*=cos(kz*PI)
    ##out*=W(z)/Dinterp(z)
    #out*=iWD
    #return out*const

xi_gp = lambda kz,kp,rp,PI,z:cos(kz*PI)*kp**3/(kp**2+kz**2)*Pmatter(kabs(kz,kp),z)*jn(2,kp*rp)*(1.0+finterp(z)/bD*kz**2/(kp**2+kz**2))

nbin=501
ik = logspace(-3, 1,nbin)
dk = ik[1:]-ik[:-1]
ikc = 0.5*(ik[1:]+ik[:-1])
kz, kp = array(meshgrid(ikc,ikc)).reshape(2,-1)
dkz, dkp = array(meshgrid(dk,dk)).reshape(2,-1)

#out = xi_int (10.,10.,z)

#irp=10
#iPI=10

#def genxi(rpPi, z=iz):
    #irp, iPI = rpPi
    #ifn='xi_arr/rp%.2f_Pi%.2f.out'%(irp,iPI)
    #if os.path.isfile(ifn):
        #print 'skip', irp, iPI
        #return float(genfromtxt(ifn))
    #print 'computing', irp, iPI
##    J2zeros = jn_zeros(2,100)/irp
##    opts1={'points':J2zeros}
    #xi_test=nquad(xi_gp, [[1e-3, 10], [1e-3, 10]], args=(z, irp, iPI))#,opts=[{}, opts1])
    #savetxt(ifn, [xi_test[0]])
    #return xi_test[0]

rp_arr = linspace(0.5, 60.5, 21)
Pi_arr = linspace(0.5, 60, 15)
rppi_arr = [[irp, ipi] for irp in rp_arr for ipi in Pi_arr]

#for z in z_Pk[:1]:#zz[3:7]:
    #print z
    #def xi_int (rpPI,z=z):
        #print rpPI
        #rp,PI=rpPI
        #xi_arr = xi_gp(kz,kp,rp,PI,z)
        #out = sum(xi_arr*dkz*dkp)
        #return out
    #out = map(xi_int, rppi_arr)
    #save('xi_camb/xi_z%.3f.npy'%(z),array(out).reshape(21,15).T)

def genxi(z):
    print z
    def xi_int (rpPI,z=z):
        print rpPI
        rp,PI=rpPI
        xi_arr = xi_gp(kz,kp,rp,PI,z)
        out = sum(xi_arr*dkz*dkp)
        return out
    out = map(xi_int, rppi_arr)
    save('xi_camb/xi_z%.4f.npy'%(z),array(out).reshape(21,15).T)
    return out
    
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
out = pool.map(genxi, z_Pk)
#out = map(genxi, rppi_arr)

save('out',out)
print 'done done done'
#pool.close()

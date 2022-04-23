import sympy
sympy.init_printing()
import matplotlib.pyplot as plt
from sympy.plotting import plot as Plot
from matplotlib import style
from pandas import DataFrame
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy import signal
import numpy as np
from sympy import fourier_series, pi, fourier_transform, exp, sqrt, inverse_fourier_transform, cos, DiracDelta, I
from sympy.abc import x as X
from sympy.abc import k

print('laplace Transform in Python, using SymPy')
t, s= sympy.symbols('t, s')
a= sympy.symbols('a', real= True, positive= True)

# start with a simple function
f= exp(-a*t)
F= sympy.integrate(f*sympy.exp(-s*t), (t, 0, sympy.oo)) #calculate laplace transform with integral
print(F) # ⎪1s(as+1)∫0∞e−ate−stdtfor|arg(s)|≤π2otherwise

print('Calculate Laplace transform using the function')
F1= sympy.laplace_transform(f, t, s, noconds= True)
print(F1) #1/(a+s)

# make the process faster by creating functions
def L(f):
    return sympy.laplace_transform(f, t, s, noconds= True)

#inverse Laplace
def invL(F):
    return sympy.inverse_laplace_transform(F, s, t)
print(invL(F1))

Plot(sympy.Heaviside(t)) #plot of the function of Heaviside u(t)
print(L(f).subs({a: 2}))
print(invL(F1).subs({a: 2})) # just make a=2

p= Plot(f.subs({a: 2}), invL(F1).subs({a: 2}), title= 'Function exp(-2t) and the invert Laplace of her', 
     xlim=(-1, 4), ylim=(0, 3), legend= True, show= False)
p[0].line_color= 'red'
p[1].line_color= 'blue'
style.use('ggplot')
p.show()

print('Create a table with some basic functions and their Laplace Transforms')
omega= sympy.symbols('omega', real= True)
exp= sympy.exp
sin= sympy.sin
cos= sympy.cos
functions= [1, exp(-a*t), t*exp(-a*t), t**2*exp(-a*t), sin(omega*t), cos(omega*t), 1 - exp(-a*t), 
            exp(-a*t)*sin(omega*t), exp(-a*t)*cos(omega*t),]
Fs= [L(f) for f in functions] # calculate Laplace transform of the table
print(Fs)
iL= [invL(i) for i in Fs] #calculate inverse Laplace
print(iL)

print('Make the result more beautiful!')

table= DataFrame(list(zip(functions, Fs)))
print(table)
table.to_excel('Laplace.xlsx')

table2= DataFrame(list(zip(Fs, iL)))
print(table2)
table2.to_excel('invLaplace.xlsx')

print('Calculate more difficult inverse transforms!')
F2 = ((s+ 1)*(s+ 2)*(s+ 3))/((s+ 4)*(s+ 5)*(s+ 6))
F2= F2.apart(s) #conversion to simple fractions
print(F2)
f2= invL(F2)
print(f2)
f2= f2.simplify() #simplify result
print(f2)

print('Discrete Fourier transform with scipy:')
x= np.array([1.0, 2.0, 1.0, -1.0, 1.5])
y= fft(x) #calculate laplace of an array
print(y)
yinv= ifft(y) #inverse Laplace of the array
print(yinv)

print('The next example calculates the Fourier Transform of the sum of 2 sines')
N= 600 #number of sample points
T= 1.0/ 800.0 #sample spacing
x= np.linspace(0.0, N*T, N, endpoint= False)
y= np.sin(50.0* 2.0*np.pi*x)+ 0.5*np.sin(80.0*2.0*np.pi*x)
plt.plot(x, y, label= 'sin(100*π*x)+0.5*sin(160*π*x)')
plt.ylabel('Volts')
plt.xlabel('Time')
plt.legend()
plt.show()
yf= fft(y)
print(yf)
xf= fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N* np.abs(yf[0:N//2]), label= 'F{sin(100*π*x)+0.5*sin(160*π*x)}')
plt.legend()
plt.show()

print('The next example calculates the Fourier Transform of 2 complex exponentials')
N= 400 #number of signal points
T= 1.0/ 800.0 #sample spacing
x= np.linspace(0.0, N*T, N, endpoint= False)
y = np.exp(50.0 * 1.j * 2.0*np.pi*x) + 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)
plt.plot(x, y, label= '100jπx+0.5exp(-160πjx)')
plt.ylabel('Volts')
plt.xlabel('Time')
plt.legend()
plt.show()
yf= fft(y)
print(yf)
xf= fftfreq(N,T)
xf= fftshift(xf)
yplot= fftshift(yf)
plt.plot(xf, 1.0/N*np.abs(yplot), label= 'F{100jπx+0.5exp(-160πjx)}')
plt.legend()
plt.show()

print('Calculating Fourier series using SymPy:')
s= fourier_series(X**2, (X, -pi, pi)) #Fourier series of x^2 for x=[-π,π]
print(s)
print(s.scale(2).truncate()) # Scale the function by a term independent of x
print(s.scalex(2).truncate()) # Scale x by a term independent of x
print(s.shift(1).truncate()) # Shift the function by a term independent of x
print(s.shiftx(1).truncate()) # Shift x by a term independent of x
print(s.sigma_approximation(4)) # sigma_approximation(n=4): Return σ-approximation of Fourier series with respect to order n.
print(s.truncate(n= 4)) # truncate(n=4): Return the first 4 nonzero terms of the series.

print('Calculate a Fourier transform using SymPy:')
print(fourier_transform(exp(-X**2), X, k)) # Fourier of exp(-x^2)

def F(f): # make a function that calculates the Fourier transform
    return fourier_transform(f, X, k, noconds= False)

f1= cos(2*X)
F1= F(f1) # Fourier of cos(ωt)
print(F1)
f2= exp(-2*X)
F2= F(f2) # Fourier of exp(-2t)
print(F2)
f3= DiracDelta(X)
F3= F(f3) # Fourier of δ(t)
print(F3)
f4= DiracDelta(X-2)
F4= F(f4) # Fourier of δ(t-2)
print(F4)

print('Calculate an inverse Fourier transform using SymPy:')

def invF(F): # make a function that calculates the inverse Fourier transform
    return inverse_fourier_transform(F, X, k, noconds= False)

F5= exp(-I*k*2)
f5= invF(F5)
print(f5)
F6= DiracDelta(k)
f6= invF(F6)
print(f6)

print('The square signal in Scipy')
t= np.linspace(0, 1, 500, endpoint= False)
plt.plot(t, signal.square(2*np.pi*5*t))
plt.ylim(-2, 2)
plt.show()

print('Create the rectangular pulse using NumPy')
t= np.arange(-3, 3, 0.01)
y= np.zeros(len(x))
y[200:400]= 1
plt.plot(y)
plt.title('The rectangular pulse')
plt.xlabel('Time')
plt.show()

print('Calculate the Fourier transform of the Rectangular Pulse')
ffty= np.fft.fft(y)
ffty= np.fft.fftshift(ffty)
plt.plot(np.real(ffty))
plt.title('sinc(f)')
plt.xlabel('frequency')
plt.xlim(190, 210)
plt.show()

print('The triangular signal in SciPy')
window= signal.triang(51)
plt.plot(window)
plt.title('Triangular signal')
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.show()

print('Calculate Fourier transform of Triangular Pulse')
A= fft(window, 4000)/ (len(window)/ 2.0)
freq= np.linspace(-0.5, 0.5, len(A))
plt.plot(A, freq)
plt.title('Frequency of triangular pulse')
plt.show()

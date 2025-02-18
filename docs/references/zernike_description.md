# Zernike polynomials

Analytical expressions of the surface sag, its derivatives, and surface normal vector.\
[drpaprika](https://github.com/drpaprika), 2025. 


## The Fringe Zernike Surface 
A point on a freeform surface represented by a set of Fringe Zernike polynomials up to degree $N$ is modeled as:  

$$
z(x,y) = z_{\mathrm{conic}}(x,y)+\sum_{i=0}^{N-1}c_iZ_i\bigl(\rho,\theta\bigr)
$$  

The polar coordinates are extracted from the cartesian coordinates:

$$
\rho =\sqrt{\bigl(\tfrac{x}{\text{norm}_x}\bigr)^2 + \bigl(\tfrac{y}{\text{norm}_y}\bigr)^2}
\quad \quad \text{and} \quad \quad
\theta = \arctan2\bigl(\tfrac{y}{\text{norm}_y},\tfrac{x}{\text{norm}_x}\bigr)
$$

The base conic is in the form:

$$
z_{\text{base}}(x,y) = \frac{r^2}{R\Bigl(1 + \sqrt{1 - (1+cc)r^2 / R^2}\Bigr)} \qquad \text{with} \qquad
\begin{cases}
  r^2 = x^2 + y^2\\
  cc \text{ the conic constant}
\end{cases}
$$

<br />

A given $Z_n^m(\rho,\theta)$ can be written as the combination of a radial term $R_n^m(\rho)$ and an azimutal term:

$$
Z_n^m(\rho,\theta) =
\sum_{k=0}^{\lfloor (n - |m|)/2 \rfloor}
(-1)^k\frac{(n-k)!}{k!\bigl(\tfrac{n+|m|}{2}-k\bigr)!\bigl(\tfrac{n-|m|}{2}-k\bigr)!}
\rho^{n-2k}.
  \begin{cases}
    \cos(m\theta) & \text{if } m \ge 0\\
    \sin(|m|\theta) & \text{if } m < 0
  \end{cases}
$$  

The conversion from Standard to Fringe ordering follows :

$$i = int\biggr[\biggr(\frac{n+|m|}{2}+1\biggr)^2 -2|m|+\frac{1-sgn(m)}{2} \biggr]$$

Alternatively the radial and azimutal indexes $(n,m)$ from the fringe index $i$:

$$
n = \text{ceil}\biggr[\frac{-3+\sqrt{9+8i}}{2}\biggr] 
\quad \quad \text{and} \quad \quad 
m = 2j - n(n+2)
$$

<br />

## Base conic derivatives 
The base conic part $z_{\text{base}}$ often has a known partial derivative formula. 
Example for a conic (sphere, paraboloid ...):

$$
z_{\text{conic}}(x,y) = \frac{r^2}{R\Bigl(1 + \sqrt{1 - (1+cc)r^2 / R^2}\Bigr)} \qquad \text{with} \qquad r^2 = x^2 + y^2
$$

A typical derivative for the conic portion is (and similarly for $\partial z / \partial y$):

$$
\frac{\partial z_{\text{conic}}}{\partial x} = \frac{x}{R \sqrt{1 - (1+cc)r^2 / R^2}}
$$

<br />

## Zernike derivatives 

In order to get $\frac{\partial Z}{\partial x}$ and $\frac{\partial Z}{\partial y}$ we note that:

$$
\frac{\partial Z}{\partial x} = \frac{\partial Z}{\partial \rho}\frac{\partial \rho}{\partial x} +
\frac{\partial Z}{\partial \theta}\frac{\partial \theta}{\partial x}.
$$

Since

$$
\rho = \sqrt{\Bigl(\tfrac{x}{\text{norm}_x}\Bigr)^2 + \Bigl(\tfrac{y}{\text{norm}_y}\Bigr)^2}
\quad \quad \text{and} \quad \quad \theta = \arctan2\Bigl(\tfrac{y}{\text{norm}_y}, \tfrac{x}{\text{norm}_x}\Bigr)
$$

one derives:

$$
\begin{cases}
  \frac{\partial \rho}{\partial x} = \frac{x}{\text{norm}_x^2 \cdot \rho} \\
  \frac{\partial \rho}{\partial y} = \frac{y}{\text{norm}_y^2 \cdot \rho} \\
\end{cases}
\quad \quad \text{and} \quad \quad
\begin{cases}
  \frac{\partial \theta}{\partial x} = -\frac{y}{\rho^2 \cdot \text{norm}_y \cdot \text{norm}_x} \\
  \frac{\partial \theta}{\partial y} = +\frac{x}{\rho^2 \cdot \text{norm}_y \cdot \text{norm}_x} \\
\end{cases}
$$

The  signs and denominators come from the derivative of $\arctan⁡2(y/x)$.

<br />

We also compute the surface derivatives in polar coordinates:

- Derivative wrt $\rho$:

$$
\frac{\partial Z}{\partial \rho} =
\sum_{k=0}^{\lfloor (n - |m|)/2 \rfloor}
(-1)^k\frac{(n-k)!}{k!\bigl(\tfrac{n+|m|}{2}-k\bigr)!\bigl(\tfrac{n-|m|}{2}-k\bigr)!}\bigl(n-2k\bigr)\rho^{n-2k-1} \cdot
\begin{cases}
    \cos(m\theta) & m \ge 0\\
    \sin(|m|\theta) & m<0\\
\end{cases}
$$

- Derivative wrt $\theta$:

$$
\frac{\partial Z}{\partial \theta} = \frac{\partial}{\partial \theta}
\begin{cases}
  R_n^m(\rho)\cos(m\theta) = -mR_n^m(\rho)\sin(m\theta)\quad & m>0 \\
  R_n^m(\rho)\sin(|m|\theta) = |m|R_n^m(\rho)\cos(|m|\theta)\quad & m<0 \\
  R_n^0(\rho)\quad\text{(no } \theta \text{-dependence)}\quad & m=0
\end{cases}
$$

<br />

## Surface Normal

Once we have the two partial derivatives:

$$
dzdx = \frac{\partial z}{\partial x}
\quad \quad \text{and} \quad \quad
dzdy = \frac{\partial z}{\partial y}
$$

we form the (unnormalized) normal vector:

$$
\Bigl(-dzdx,-dzdy,1\Bigr)
$$

Its magnitude is:

$$
\sqrt{(dzdx)^2 + (dzdy)^2 + 1}
$$

so the **unit normal** is:

$$
\vec{N}(x,y) = \frac{\bigl(-dzdx,-dzdy,1\bigr)}{\sqrt{ (dzdx)^2 + (dzdy)^2+1}}
$$

<br />

## Final Note

The method described is general to any explicit surface $z(x,y)$. The Zernike part just happens to be a sum of polynomials in $(\rho,\theta)$, which requires an extra step to do the chain rule from $(\rho,\theta)$ back to $(x,y)$.

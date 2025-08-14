# Zernike polynomials

Analytical expressions of the surface sag, its derivatives, and surface normal vector.\
[drpaprika](https://github.com/drpaprika), 2025. 


## The Zernike Polynomial Surface 
A point on a freeform surface represented by a set of Fringe Zernike polynomials up to degree $N$ is modeled as:  

$$
z(x,y) = z_{\mathrm{conic}}(x,y) + \sum_{i=0}^{N - 1}c_iZ_i\bigl(\rho,\theta\bigr)
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

<br />

There are multiple ways to convert the radial and azimuthal orders $(n, m)$ to a single index $i$.
Optiland implements the OSA/ANSI, Noll and Fringe single-index schemes.

### OSA/ANSI

$${\displaystyle i={\frac {n(n+2)+m}{2}}}$$

### Noll

$$
{\displaystyle i={\frac {n(n+1)}{2}}+|m|+
\left\{
  {\begin{array}{ll}
  0,&m>0\land n\equiv \{0,1\}{\pmod {4}};\\
  0,&m<0\land n\equiv \{2,3\}{\pmod {4}};\\
  1,&m\geq 0\land n\equiv \{2,3\}{\pmod {4}};\\
  1,&m\leq 0\land n\equiv \{0,1\}{\pmod {4}}.\end{array}}
\right.}
$$

### Fringe (University of Arizona)

$$
{\displaystyle i=\left(1+{\frac {n+|m|}{2}}\right)^{2}-2|m|+\left\lfloor {\frac {1-\operatorname {sgn} m}{2}}\right\rfloor }
$$

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


## References

- Niu, K., & Tian, C. (2022). Zernike polynomials and their applications. Journal of Optics, 24(12), 123001. https://doi.org/10.1088/2040-8986/ac9e08
- Zernike polynomials. (2025). In Wikipedia. https://en.wikipedia.org/w/index.php?title=Zernike_polynomials&oldid=1304849323
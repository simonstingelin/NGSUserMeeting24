---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

`````{margin}
````{note}
Executable md-file
```
---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```
````
`````

# Finite element method getting started examples

The following page is intended as an example of both the
- interaction between `numpy` and `ngsolve` in teaching, as well as
- for `ngsolve` in a Markdown file.

## One-dimensional case

As an introduction to the finite element method, let us consider the scalar boundary value problem
`````{margin}
````{note}
Label
```
$$\begin{split}
-u''(x) & = f(x)\quad\forall\ x\in (0,1)\\
u(0) & = u(1) = 0,
\end{split}$$ (eq:strongEquation)
```
````
`````
$$\begin{split}
-u''(x) & = f(x)\quad\forall\ x\in (0,1)\\
u(0) & = u(1) = 0,
\end{split}$$ (eq:strongEquation)
with $f(x) = 1$.


We can easily obtain the analytical solution in this case. By integrating the right-hand side twice, we obtain a 2nd order polynomial

$$u(x) = \frac{1}{2} x^2 + C_1 x + C_2.$$

By inserting the boundary conditions, the analytical solution follows

$$u(x) = -\frac{1}{2} x (x-1).$$

`````{margin}
````{note}
Executable code block
```
```{code-cell} ipython3
def uanalytic(x):
    return -0.5*x*(x-1)
```
````
`````

```{code-cell} ipython3
def uanalytic(x):
    return -0.5*x*(x-1)
```

For the numerical solution, we multiply the differential equation with an arbitrary test function $v(x)\in C_0^\infty(0,1)$ and integrate over the interval $(0,1)$

$$-\int_0^1 u''(x) v(x)\,dx = \int_0^1 f(x) v(x)\,dx \quad\forall\ v\in C_0^\infty(0,1).$$

Using partial integration, we obtain the weak equation

$$\int_0^1 u'(x) v'(x)\,dx - \underbrace{\big[u'(x) v(x)\big]_0^1}_{=0\, \text{since}\, v(0)=v(1)=0} = \int_0^1 f(x) v(x)\,dx \quad\forall\ v\in C_0^\infty(0,1).$$

The weak problem is given by: Find $u(x)$ such that $u(0)=u(1)=0$ and

$$\int_0^1 u'(x) v'(x)\,dx  = \int_0^1 f(x) v(x)\,dx \quad\forall\ v\in C_0^\infty(0,1).$$ (eq:weakEquation)

`````{margin}
````{note}
Reference
```
problem {eq}`eq:weakEquation`
```
````
`````
To fulfill the equation, the solution $u$ does not necessarily have to be a twice continuously differentiable function. Instead, $u$ and $v$ are each differentiated once. It is sufficient to use the functions from the Sobolev space $H_0^1(0,1)$. The right-hand side also no longer has to be continuous. It is sufficient if the function $f$ is square integrable, hence $f\in L_2(0,1)$. From the problem {eq}`eq:weakEquation` we obtain solutions which do not always necessarily have to be solutions of the strong equation {eq}`eq:strongEquation`.

Let us discretize the interval $(0,1)$ into subintervals $(x_{i-1},x_i)$ for $i=1,\ldots, n$. On this decomposition we define the basis functions

$$\varphi_i(x) = \begin{cases}
\frac{x-x_{i-1}}{x_i-x_{i-1}}\quad \text{for}\ x \in (x_{i-1},x_i)\\
\frac{x_{i+1}-x}{x_{i+1}-x_{i}}\quad \text{for}\ x \in (x_{i},x_{i+1})\\
0\quad \text{else,}\end{cases}$$

with which we approximate the Sobolev space $H_0^1(0,1)$.

### Numpy solution

`````{margin}
````{note}
Hide code, remove output
```
```{code-cell} ipython3
:tags: [hide-cell, remove-output]
```
````
`````

```{code-cell} ipython3
:tags: [hide-cell, remove-output]

import numpy as np
import matplotlib.pyplot as plt
from myst_nb import glue

n=5
a=0
b=1
xi = np.linspace(a,b,n+1)

def phi(i,x,xi=xi):
    """
    1d first order FEM basis functions
    
    Parameters
    ----------
    i : int, nummer of the basis function
    x : nparray, coordinates to compute
    xi : nparray, knots

    Return
    ------
    y : nparray, function values for x
    """

    y = np.zeros_like(x)
    if i > 0:
        ind = (xi[i-1]<=x)*(x<=xi[i])
        y[ind] = (x[ind]-xi[i-1])/(xi[i]-xi[i-1])
    if i < n:
        ind = (xi[i]<=x)*(x<xi[i+1]) 
        y[ind] = (xi[i+1]-x[ind])/(xi[i+1]-xi[i])
    return y

xp = np.linspace(a,b,400)
fig, ax = plt.subplots(figsize=(6, 2))
for i in range(n+1):
    ax.plot(xp,phi(i,xp),label=r'$\varphi_'+str(i)+'$')
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
glue("FEM_1d_p1_fig", fig, display=False)
glue("FEM_1d_p1_n",n,display=False)
```

`````{margin}
````{note}
reference to figures
```
    {numref}`fig-FEM1dBasisFct`
```
````
`````

In the figure {numref}`fig-FEM1dBasisFct`, the interval is divided into {glue:}`FEM_1d_p1_n` subintervals. Accordingly, we have {glue:}`FEM_1d_p1_n` +1 basis functions. Apart from the first and last basis function, the support extends over two subintervals in each case. The basis functions are exactly one in one node $x_i$, in all others the function value is 0.

`````{margin}
````{note}
glue values in text
```
{glue:}`FEM_1d_p1_n`
```
````
`````

```{glue:figure} FEM_1d_p1_fig
---
figwidth: 400px
name: fig-FEM1dBasisFct
---

FEM 1d first order basis functions
```

`````{margin}
````{note}
glue figures
```
```{glue:figure} FEM_1d_p1_fig
---
figwidth: 400px
name: fig-FEM1dBasisFct
---

FEM 1d first order basis functions
```
````
`````

For the weak equation {eq}`eq:weakEquation` we obtain the finite dimensional problem: Find

$$u_h(x) = \sum_{i=0}^n u_i \, \varphi_i(x)$$

such, that $u(0) = u(1) = 0$ and

$$\int_0^1 u_h'(x) \varphi_j'(x)\,dx  = \int_0^1 f(x) \varphi_j(x)\,dx \quad\forall\ j = 1, \ldots, n-1.$$ (eq:discretweekEquation)

Applying $u_h$ into {eq}`eq:discretweekEquation` leads to the system of equation

$$\sum_{i=0}^n u_i\ \int_0^1 \varphi_i'(x) \varphi_j'(x)\,dx  = \int_0^1 f(x) \varphi_j(x)\,dx \quad\forall\ j = 1, \ldots, n-1.$$ (eq:discweekequation)

We define the matrix $A$ and the vector $b$

$$\begin{split}
A_{j,i} & = \int_0^1 \varphi_i'(x) \varphi_j'(x)\,dx\\
b_j & = \int_0^1 f(x) \varphi_j(x)\,dx
\end{split}$$ (eq:linBilinForm1dproblem)

and obtain the (reduced) linear system of equations for the coefficients $u_1, \ldots, u_{n-1}$

$$\sum_{i=1}^{n-1} A_{j,i} u_i = b_j \quad \forall\ j=1, \ldots, n-1.$$(eq:FEM1dLinOrderSys)

or in compact form

$$A\cdot \vec{u} = \vec{b}$$

with $\vec{u} = (u_1, \ldots, u_{n-1})^T$ and $\vec{b} = (b_1, \ldots, b_{n-1})^T$. Due to the Dirichlet boundary conditions we have $u_0 = u_n = 0$.

Assuming an equidistant decomposition as in the example above {numref}`fig-FEM1dBasisFct`, we get

$$\varphi_i'(x) = \begin{cases}
1/h\quad \text{für}\ x\in (x_{i-1},x_i)\\
-1/h\quad \text{für}\ x\in (x_{i},x_{i+1})\\
0 \quad \text{sonst.}\end{cases}$$

```{code-cell} ipython3
:tags: [hide-cell, remove-output]

def dphi(i,x,xi=xi):
    """
    Derivative 1d first order FEM basis functionen
    
    Parameters
    ----------
    i : int, nummer of the basis function
    x : nparray, coordinates to compute
    xi : nparray, knots

    Return
    ------
    y : nparray, function values for x
    """
    y = np.zeros_like(x)
    if i > 0:
        h = xi[i]-xi[i-1]
        ind = (xi[i-1]<=x)*(x<=xi[i])
        y[ind] = np.ones_like(x[ind])/h
    if i < n:
        h = xi[i+1]-xi[i]
        ind = (xi[i]<=x)*(x<xi[i+1]) 
        y[ind] = -np.ones_like(x[ind])/h
    return y

fig, ax = plt.subplots(figsize=(6, 2))
for i in range(n+1):
    ax.plot(xp,dphi(i,xp),label=r'$\varphi_'+str(i)+'\'(x)$')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
glue("FEM_1d_p1_deriv_fig", fig, display=False)
```

```{glue:figure} FEM_1d_p1_deriv_fig
---
figwidth: 400px
name: fig-FEM1dDerivBasisFkt
---

Derivative 1d first order FEM basis functions
```

For the specific example, we obtain the matrix $A$:

```{code-cell} ipython3
:tags: [hide-input]

from scipy.integrate import fixed_quad

A = []
for i in range(0,n+1):
    ai = []
    for j in range(0,n+1):
        # integration over the elements using the Gaussian quadrature
        aij = np.sum([fixed_quad(lambda x:dphi(i,np.array(x))*dphi(j,np.array(x)), xi[k], xi[k+1],n=2)[0]
                      for k in range(n)])
        ai.append(aij)
    A.append(ai)
A = np.array(A,dtype=float)
A
```

`````{margin}
````{note}
Notes, warnings, and other admonitions
```
```{admonition} Exercise
Compute the coefficients by hand.
```
````
`````
```{admonition} Exercise
Compute the coefficients by hand.
```

Analog follows for the right-hand side with concrete function $f(x) = 1$:

```{code-cell} ipython3
:tags: [hide-input]

def f(x):
    x = np.array(x)
    return np.ones_like(x)

b = []
for j in range(0,n+1):
    b.append(np.sum([fixed_quad(lambda x:f(np.array(x))*phi(j,np.array(x)),
                                 xi[k], xi[k+1],n=2)[0]
                      for k in range(n)]))
b = np.array(b,dtype=float)
b
```

The FEM solution $u(x)$ is therefore given by the solution of the system of equations {eq}`eq:FEM1dLinOrderSys` with the calculated system matrix and right hand side vector. Since the boundary values are given, only the inner degrees of freedom are used. The result is

```{code-cell} ipython3
u = np.zeros_like(xi)

# We will deal with the choice of linear equation solvers later:
u[1:-1] = np.linalg.solve(A[1:-1,1:-1],b[1:-1])
u
```

```{code-cell} ipython3
:tags: [hide-cell, remove-output]

xp = np.linspace(0,1,400)
fig, ax = plt.subplots(figsize=(6, 2))
ax.plot(xi,u,label='FEM solution')
ax.plot(xp,uanalytic(xp),label='analytical solution')
ax.legend()
glue("FEM_1d_p1_solutionexmp_fig", fig, display=False)
```

```{glue:figure} FEM_1d_p1_solutionexmp_fig
---
figwidth: 400px
name: fig-FEM1dSolutionExmp
---

Solution of the boundary value problem using 1st order FEM
```

```{admonition} Exercise
* What is the system of equations if the trapezoidal rule is used to integrate the right-hand side?
* Calculate the solution using the finite difference method and compare the two systems.
* Why does the numerical solution in the nodes match the analytical solution exactly?
```

### NGSolve solution

Now let's do the same with NGSolve

* First, we create a one-dimensional mesh using `netgen`.

```{code-cell} ipython3
from netgen.meshing import Mesh as NGMesh # Caution: Mesh is also available in ngsolve!
from netgen.meshing import MeshPoint, Pnt, Element1D, Element0D
from ngsolve import *

m = NGMesh(dim=1)

# Number of subintervals
N = 5

# Points for the decomposition on the interval [0,1]
pnums = []
for i in range(0, N+1):
    pnums.append (m.Add (MeshPoint (Pnt(i/N, 0, 0))))

# Each 1D element (subinterval) can be assigned to a material.
# In our case, there is only one material.
idx = m.AddRegion("material", dim=1)
for i in range(0,N):
    m.Add (Element1D ([pnums[i],pnums[i+1]], index=idx))

# Left and right ends are boundary value points (0D elements)
idx_left = m.AddRegion("left", dim=0)
idx_right = m.AddRegion("right", dim=0)

m.Add (Element0D (pnums[0], index=idx_left))
m.Add (Element0D (pnums[N], index=idx_right))

# We have now defined the mesh
mesh = Mesh(m)
```

Now we create a $H^1$ function space using this 1D mesh and the Dirichlet boundary points `left` and `right`.

```{code-cell} ipython3
V = H1(mesh,order = 1, dirichlet='left|right')
u = V.TrialFunction()
v = V.TestFunction()

# short cut
# u,v = V.TnT()
```

$u,v$ are trial and test functions for the definition of the linear and bilinear function

```{code-cell} ipython3
a = BilinearForm(V)
a += grad(u)*grad(v)*dx

f = CoefficientFunction(1)
b = LinearForm(V)
b += f*v*dx
```

The two operators are now defined, but not yet calculated. The calculation is also called **assembling**. We will see later what this means in detail.

```{code-cell} ipython3
a.Assemble()
b.Assemble();
```

The matrix and the vector of the bilinear and linear form contain all degrees of freedom, in particular also the boundary points. However, these are given by the Dirichlet boundary values and do not need to be calculated (as mentioned above). We will take this into account when solving the system.

```{code-cell} ipython3
print(a.mat)
```

```{code-cell} ipython3
print(b.vec)
```

The `GridFunction` class is available in `NGSolve` for the solution itself. The trial and test functions are only used to define the problem and therefore have no memory for the solution vector, for example. The linear combination of the basis functions is automatically calculated in the evaluation of this. We need a GridFunction for the solution:

```{code-cell} ipython3
gfu = GridFunction(V)
```

Calculation of the FEM solution with NGSolve taking into account the free degrees of freedom:

```{code-cell} ipython3
gfu.vec.data = a.mat.Inverse(freedofs=V.FreeDofs())*b.vec
```

When the command is executed, the matrix is not inverted and multiplied to the vector from the left. That would be far too time-consuming numerically. Even if the notation claims otherwise, only the system of equations is solved.

Again, we get the same result as above:

```{code-cell} ipython3
:tags: [hide-input]

xp = np.linspace(0,1,400)
plt.plot(xp,[gfu(mesh(xi,0)) for xi in xp],label='numerical solution')
plt.plot(xp, uanalytic(xp),label='analytical solution')
plt.legend()
plt.grid()
plt.show()
```

## Extension into two-dimensional space

Now let us look at the two-dimensional case. We will see that apart from the mesh, hence the region, the implementation in ngsolve remains identical. We consider the analog boundary value problem in two dimensions

$$\begin{split}
-\Delta u & = 1 \quad\text{for}\ x\in \Omega\\
u & = 0 \quad\text{for}\ x\in \partial\Omega
\end{split}$$

on the unit square in $\mathbb{R}^2$. For the weak equation follows

$$\int_\Omega \nabla u \cdot \nabla \varphi dV = \int_\Omega f(x) \varphi dV \quad \forall\ \varphi\in V \subset H_0^1(\Omega),$$

where $V$ is a finite element space with $\dim V < \infty$.

```{code-cell} ipython3
from ngsolve import *
from ngsolve.webgui import Draw
```

From `netgen` we import the unit square. With the help of the netgen module we can define and mesh 2d and 3d geometries.

```{code-cell} ipython3
from netgen.geom2d import unit_square
```

We now generate an unstructured mesh with a maximum edge length of 0.25:

```{code-cell} ipython3
mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
```

The size of the mesh can be controlled via the `maxh` parameter.

`````{margin}
````{note}
`webgui_jupyter_widgets`

The js-file must be installed in _static/ in order to have the functionality available (see {ref}`chap:ngsolveinjb`).
````
`````

```{code-cell} ipython3
Draw(mesh);
```

What do the basis functions look like in two dimensions? To do this, we create an FEM function space and visualize the basis functions. The edge of the unit square consists of 4 lines, which have different labels:

```{code-cell} ipython3
mesh.GetBoundaries()
```

We define the `H1` FEM function space with the Dirichlet boundary condition and initialize a `GridFunction`, an FEM function from the function space:

```{code-cell} ipython3
V = H1(mesh,order=1,dirichlet='bottom|right|top|left')
gfu = GridFunction(V)
```

Using the grid function, we can visualize the global basic functions. To do this, we set one coefficient to 1 and the others to 0. In the following example we visualize the 20th global basis function:

```{code-cell} ipython3
gfu.vec.FV()[:] = 0 # Initialize all entries with 0
gfu.vec.FV()[20] = 1

Draw(gfu,mesh,'u');
```

Using $u(x) = \sum_i u_i\,\varphi_i(x) \in V$, as in the 1d example {eq}`eq:discweekequation`, the same system of equations follows

$$A(u,v) = f(v)\quad \forall \ v\in V,$$

with the bilinear form $A: V \times V \to \mathbb{R}$ 

$$\begin{split}
A : V \times V & \to \mathbb{R}\\
    (u,v) & \mapsto A(u,v) = \int_\Omega \nabla u\cdot \nabla v\, dV,
\end{split}$$

and linear form $f: V \to \mathbb{R}$

$$\begin{split}
b : V & \to \mathbb{R}\\
    v & \mapsto b(v) = \int_\Omega f(x)\cdot v\, dV.
\end{split}$$

For the definition in `ngsolve` we use again the proxy functions

```{code-cell} ipython3
u = V.TrialFunction()
v = V.TestFunction()
```

In two dimensions, we use `dx` to denote a surface integral and `ds` for a path integral (boundary integral). In our case we have for the bilinear form

```{code-cell} ipython3
A = BilinearForm(V)
A += grad(u) * grad(v)*dx
```

and analogously for the linear form

```{code-cell} ipython3
f = CoefficientFunction(1)
b = LinearForm(V)
b += f*v*dx
```

The right-hand side function onece again defined using a `CoefficientFunction`.

We have thus defined the bilinear form, which can be described in finite dimensions using a matrix, and the linear form, which can be described using a vector, but have not yet calculated them. The calculation of these is also called **assembling**.

```{code-cell} ipython3
A.Assemble()
b.Assemble();
```

The matrix $A$ is in a *sparse* format stored. This means that only the matrix entries for which we potentially receive an entry are saved. The memory is not allocated at all for the rest of the matrix.

```{code-cell} ipython3
print(A.mat)
```

We can also regard this matrix (at least as long as it is small!) as a *dense* matrix:

```{code-cell} ipython3
rows,cols,vals = A.mat.COO()

denseA = np.zeros((np.max(rows)+1,np.max(rows)+1))
k=0
for i,j in zip(rows,cols):
    denseA[i,j] = vals[k]
    k+=1

plt.spy(denseA)
plt.show()
```

We can also use `scipy.sparse` for further computations.

```{code-cell} ipython3
import scipy.sparse as sp
Asparse = sp.coo_matrix((vals,(rows,cols)))
plt.spy(Asparse)
```

The pattern of the matrix entries depends on the numbering of the nodes (from the meshing) and the FEM basic functions. It is also possible to draw the mesh directly with `matplotlib`; the mesh object contains all the information required for this.

```{code-cell} ipython3
for e in mesh.edges:
    line = np.array([mesh.vertices[v.nr].point for v in e.vertices])
    plt.plot(line[:,0],line[:,1],c='gray',alpha=0.75)
for v in mesh.vertices:
    plt.text(*v.point,v,color='red')
plt.gca().set_axis_off()
plt.gca().set_aspect(1)
plt.show()
```

We now consider the free degrees of freedom, for which we must keep the homogeneous Dirichlet boundary condition in mind. The number of free dof's is given by

```{code-cell} ipython3
freedofs = V.FreeDofs()
print(freedofs)
freedofsnp = np.array([i for i in freedofs])
```

As we can see, the order of numbering the nodes is via the corners, edges into the interior. Thus, the degrees of freedom available to us are at the end. The matrix pattern for solving the problem is given by

```{code-cell} ipython3
ind = np.arange(freedofsnp.shape[0])[freedofsnp]
plt.spy(denseA[np.ix_(ind,ind)])
plt.show()
```

Let's solve the system for the inner degrees of freedom:

`````{margin}
````{note}
I usually start writing with a Jupyter notebook and do the cosmetics in the exported Markdown file later.
````
`````

```{code-cell} ipython3
gfu.vec.data = A.mat.Inverse(freedofs=V.FreeDofs())*b.vec

Draw(gfu,mesh,'u');
```
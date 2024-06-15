---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Higher Analysis and Numerics

## Course participants

The [ZHAW](https://www.zhaw.ch/en/university/) is one of the leading **universities of applied sciences** (**Fachhochschule**) in Switzerland. It offers teaching, research, continuing education and other services that are both practice-oriented and science-based.

The participants of the course are students of the [**School of Engineering**](https://www.zhaw.ch/en/engineering/) and typically study

- [Computer Science](https://www.zhaw.ch/en/engineering/study/bachelors-degree-programmes/computer-science/)
- [Mechanical Engineering](https://www.zhaw.ch/en/engineering/study/bachelors-degree-programmes/mechanical-engineering/)
- [Electrical Engineering](https://www.zhaw.ch/en/engineering/study/bachelors-degree-programmes/electrical-engineering/)
- [Systems Engineering](https://www.zhaw.ch/en/engineering/study/bachelors-degree-programmes/systems-engineering/)
- [Aviation](https://www.zhaw.ch/en/engineering/study/bachelors-degree-programmes/aviation/)

The students have the basic mathematical knowledge in linear algebra, analysis and numerics with a focus on application. The course is intended as preparation for a Master's degree course (university, university of applied sciences) or as a specialization for students interested in mathematics.

---

## Organization of the course
In addition to an introduction to functional analysis and modeling with partial differential equations (PDE), numerical methods for solving PDEs are developed. Accompanying applications from engineering practice are presented.

```{figure} ./images/HANAAufbau.png
:height: 300px
:name: Organization

Organization of the course
```

---

## Content of the course

```{dropdown} Analysis
- Introduction to functional analysis
    - Basic concepts
    - Function spaces
    - Calculus of variations
        - Differentiability in function spaces
        - Weak solutions
- Introduction to partial differential equations
    - What is a partial differential equation?
    - Linear partial differential equations
    - Classification PDE 2nd order
    - Modeling
        - Stationary diffusion processes
        - Transient diffusion processes
        - Hyperbolic equation
```

```{dropdown} Numerics
- Finite element method
    - Introduction in one dimension
    - Finite element procedure
    - Nodal finite elements (2d, 3d)
    - Finite elements of higher order
- Linear equation solver
    - Matrix structure examples
    - Direct equation solvers
    - Iterative equation solvers
```

```{dropdown} Practical applications (projects)
- Stationary temperature field
- Electrostatic field
- Magnetostatic field (2d)
- Mechanical field
- Flow field
- Transient temperature field
- Acoustic field
```
    

---

## Software

The program includes the implementation of numerical methods on the computer. In particular, students also learn about modern digital techniques for product optimization.

::::{grid} 1 1 2 3
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: https://www.python.org/downloads/
:class-header: bg-light

<img src="./images/python-logo@2x.png" alt="Python" width="150px">
:::

:::{grid-item-card}
:link: https://numpy.org/
:link-type: url
:class-header: bg-light

<img src="./images/NumPy.png" alt="NumPy" width="150px">

:::

:::{grid-item-card}
:link: https://ngsolve.org/
:link-type: url
:class-header: bg-light

<img src="./images/logo_withname_retina.png" alt="NETGEN/NGSolve" width="150px">

:::

:::{grid-item-card}
:link: https://scipy.org/
:link-type: url
:class-header: bg-light

<img src="./images/SciPy.png" alt="SciPy" width="150px">

:::

:::{grid-item-card}
:link: https://www.sympy.org/
:link-type: url
:class-header: bg-light

<img src="./images/SymPy.png" alt="SymPy" width="150px">

:::

::::
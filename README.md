![Header](header.png)

<div align="center">



    
### Mesh Bayesian Optimisation for manifold-constrained black-box optimisation


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A0-eSWHt5LHLX-ual3sWgKXp9eLqvzLU?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-repository-181717?logo=github)](https://github.com/aum3/Mesh-Manifold-Bayesian-Optimisation)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Aum Dave** · University of Warwick

</div>

> **Summary::** a demo of Bayesian optimisation (BO) on meshes, for black-box functions that are only defined on a surface.

---
## 1. Introduction

> **How do you optimise a function you know nothing of?** No derivatives, closed form etc. and every evaluation is expensive? Bayesian optimisation handles this with well-defined, testable statistical assumptions, which is why it is so popular wherever evaluations cost real money or time (materials, molecules, engineering design).

### 1.1 Problem

$$
x^\star = \arg\min_{x \in \mathcal{M}} f(x)
$$

where $\mathcal{M}$ is a 2D manifold (surface) embedded in $\mathbb{R}^3$ and $f$ is an expensive black-box function.

### 1.2 Existing approaches

- **Closed-form kernels** exist where the Laplace-Beltrami eigendecomposition is known ($S^n$, $T^n$, etc.).
- **Approximations** cover general manifolds, e.g. a weighted graph built from a point cloud.

### 1.3 My approach

Use a **mesh** as the approximation. Mesh Laplacians are well studied (spectral geometry, FEA), and meshes carry more geometric information than graphs.

1. Sample 100 points from the manifold.
2. Fit a mesh with Poisson surface reconstruction, which comes with strong theoretical guarantees.
3. Use the Mesh matern kernel.
4. Since this is an approximation of an approximation, check the kernel against the exact one on a unit sphere.



---
## 2. Setup



---
## 3. Mesh and Kernel

**How does knowing $f$ at one point tell you about $f$ elsewhere?**

If you knew it was 22°C in Coventry, you'd guess nearby towns are close to 22°C too. Information spreads with distance. That is what a kernel does.

### The mesh

A mesh is a surface stitched together from triangles. I built mine from the 100-point cloud using Poisson surface reconstruction (PSR), which fits a smooth implicit surface to the points and extracts a triangle mesh from it.

<a title="Chrschn, Public domain, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Dolphin_triangle_mesh.png"><img width="330" alt="The polygon mesh of dolphin." src="https://thumb.wikimedia.org/wikipedia/commons/thumb/f/fb/Dolphin_triangle_mesh.png/330px-Dolphin_triangle_mesh.png?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail"></a>

### The Laplacian $\Delta$

<a title="Nicoguaro. Based on File:Heat eqn.gif by en:User:Oleg Alexandrov, CC BY 4.0 &lt;https://creativecommons.org/licenses/by/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Heat.gif"><img width="330" alt="Animation of the heat equation with a crescent moon as initial condition." src="https://thumb.wikimedia.org/wikipedia/commons/thumb/0/01/Heat.gif/330px-Heat.gif?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail"></a>

Heat spreads by averaging: a point hotter than its neighbours cools, a colder one warms. The Laplacian measures that gap (value at a point minus the average of its neighbours), so it gives the rate of change. On a surface we use the Laplace-Beltrami operator.

Matérn kernels are built from the eigenfunctions of this operator, so they inherit the geometry of the surface and spread information the way heat would.

### The Matérn kernel

- **Lengthscale** $\ell$: how far correlations reach across the surface.
- **Smoothness** $\nu$: how rough the sampled functions are.

That control over roughness is why Matérn kernels suit real-world data, where functions can be continuous but not differentiable.

---
## 4. Objective Function

> **This is what we are trying to minimise.** In practice it could be a temperature or the water solubility of a molecule. Both are costly to measure.

For demonstration I use $f(x, y, z) = x$. The method works for any continuous function, provided the kernel suits it.


    
![png](saladin_files/saladin_8_0.png)
    


### What you're looking at

The ground-truth objective on the mesh. The mesh isn't a perfect sphere because it was built from only 100 points. That is deliberate: the method should still work when the manifold is unknown or only hypothesised.

---
## 5. Bayesian Optimisation

> **In short:** treat $f(x)$ at every point as a Gaussian random variable. Together these give a distribution over plausible functions: a Gaussian process (GP).

The kernel ties the variables together. Once $f(x)$ is known, $k(x, y)$ constrains what $f(y)$ can be at every other point $y$.

**The loop**

0. Start with a prior mean $\mu(x)$ and standard deviation $\sigma(x)$ at every point.
1. Evaluate $f$ at a chosen point $x$.
2. Condition the GP on the data. The kernel updates $\mu$ and $\sigma$ at *every* point (now the posterior).
3. Pick the next $x$ using an acquisition function, and repeat while tracking the best value seen.

The only real information is the observed data, and by design there is very little of it. If $f$ were cheap, you wouldn't need BO.

### 5.1 Gaussian-process surrogate

The surrogate is a cheap, differentiable stand-in for $f$. With a zero-mean prior and kernel $k$, conditioning on $m$ observations $(X, \mathbf{y})$ gives

$$
\mu(\cdot) = k(\cdot, X)\,\big[K_{XX} + \sigma^2 I\big]^{-1}\mathbf{y},
\qquad
\Sigma(\cdot,\cdot') = k(\cdot,\cdot') - k(\cdot, X)\,\big[K_{XX} + \sigma^2 I\big]^{-1} k(X, \cdot').
$$

($\sigma^2$ is a small jitter for numerical stability.)

### 5.2 Acquisition function: expected improvement

The acquisition function scores each point by how promising it is to sample next. BO swaps the hard problem (optimise $f$) for an easy one (optimise this score, which is cheap and differentiable).

For minimisation with incumbent $f_{\min}$ and $Z = (f_{\min} - \mu - \xi)/\sigma$:

$$
\mathrm{EI}(x) = (f_{\min} - \mu - \xi)\,\Phi(Z) + \sigma\,\varphi(Z)
$$

$\xi$ trades off exploration against exploitation (larger $\xi$ favours exploring). I use $\xi = 0.9$ here.

---
## 6. Mesh BO in action: Posterior Evolution

> **Information spreads along the geometry of the mesh.**

### What you're looking at
- BO running over the mesh. Colour is the GP posterior mean: **dark = low** (what we're after), light = high.
- Top row: front of the mesh. Bottom row: back.
- Orange dots are points already sampled. The red diamond is where expected improvement wants to sample next.

### What to notice
- The search stays in one region for a few steps (exploiting), then jumps to a distant part of the mesh (exploring).
- The dark basin on the back sharpens with each sample, while the front needs no more evaluations after the first two.
- The mesh has about 4,000 vertices, yet a handful of evaluations gets close to the true minimum (numbers below).

> **Each observation is spread across the surface by the kernel, so the optimiser needs far fewer evaluations.**


    
![png](saladin_files/saladin_13_0.png)
    


    initial vertex value: 0.5475596189498901
    sampled values (sorted, incl. initial): [-0.97176754 -0.91938204 -0.91281438 -0.47099268 -0.26728225  0.00433171
      0.54755962]
    true minimum: -1.0502488613128662 at vertex 1976


---
## 7. Sample Path

> **Which route did the optimiser take?** (A slightly different set-up from section 6.)

### What you're looking at
Numbers show the order in which vertices were sampled, joined by curves that follow the surface. Left: ground-truth objective. Right: the GP posterior mean after 3 sample points.


    
![png](saladin_files/saladin_15_0.png)
    


---
## 8. Kernel Influence

> **What does one observation tell the model about the rest of the surface?**

### What you're looking at
Heat map of $k(x_0, \cdot)$ for a single source vertex $x_0$ (red diamond). Bright = strongly correlated with the source.

### What it means
Correlation fades with distance *along the surface*, as if $x_0$ were a heat source. That is the geometry we wanted the kernel to capture.


    
![png](saladin_files/saladin_17_0.png)
    











---
## 9. Error Analysis

> **How much do we lose by using a mesh instead of the smooth manifold?**

### Setup
The 100 points were sampled from a sphere, so a "true" kernel exists. I take the same points and compare (a) the mesh kernel at their nearest mesh vertices with (b) the continuous Matérn kernel on $S^2$, using identical hyperparameters. Both are plotted against geodesic distance.

### What you're looking at
Red: mesh kernel. Blue: continuous sphere kernel. Each dot is a pair of points (a random subset, to keep the plot readable).


    
![png](saladin_files/saladin_20_0.png)
    


---
## 10. Conclusions and Future Work

Bayesian optimisation on meshes is viable: a mesh built from just 100 points gave a kernel that tracks the true sphere kernel closely.

### Results
- The mesh kernel is very close to the continuous one (RMS difference 0.0055).
- That held even though the mesh is an approximation built from 100 random points.
- BO reached a value of -0.97 against a true minimum of -1.05, using 7 evaluations out of about 4,000 vertices (seed 0).

### Limitations
- One mesh, one manifold, one objective, one seed.
- Manifolds with small reach or other pathologies may not survive meshing, and mesh Laplacians are sensitive to mesh quality.

### Future work
- Repeat over many seeds and objectives, and compare against random search.
- Test on non-spherical manifolds and real data.

---
## 11. References and Links

### References
1. Borovitskiy et al., "Matérn Gaussian processes on Riemannian manifolds", NeurIPS 2020.
2. Kazhdan, Bolitho and Hoppe, "Poisson surface reconstruction", Eurographics SGP 2006.

### Links
- **Code:** [GitHub repository](https://github.com/aum3/Mesh-Manifold-Bayesian-Optimisation)
- **Author:** [LinkedIn](https://linkedin.com/in/aumd) · [Email](mailto:aumd39@gmail.com)

### Acknowledgements
Thanks to Dr Zhengang Zhong for supervising this project, and to the University of Warwick for funding it.

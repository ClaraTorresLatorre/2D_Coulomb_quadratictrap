# Numerical generation of the figure

The figure is a particle approximation of the constrained minimizer.

We approximate $\mu$ by an empirical measure

```math
\mu_n = \frac1n\sum_{i=1}^n \delta_{z_i}
```

of $n=1000$ points $z_1,\dots,z_n\in\mathbb R^2$ and minimize the corresponding energy

```math
\mathcal E[\mu_n]
=
-\frac{1}{n^2}\sum_{i\neq j}\log|z_i-z_j|
+\frac1n\sum_{i=1}^n |z_i|^2.
```

The sum runs over all ordered pairs $i\neq j$, so that each unordered pair is counted twice, consistent with

```math
-\iint \log|z-w|\,d\mu\,d\mu.
```

To avoid the singularity of the logarithm we replace $\log|z_i-z_j|$ by

```math
\frac12\log\!\bigl(|z_i-z_j|^2+\varepsilon\bigr)
```

with $\varepsilon=10^{-8}$. The gradient supplied to the optimizer is then exactly that of this regularized energy,

```math
\partial_{z_i}\mathcal E[\mu_n]
=
-\frac{2}{n^2}
\sum_{j\neq i}
\frac{z_i-z_j}{|z_i-z_j|^2+\varepsilon}
+\frac2n\,z_i,
```

and was cross-checked against finite differences.

The half-space constraint ${x_2\ge a}$ is imposed as a bound on each particle, with the first coordinate free and the second bounded below by $a$. Minimization is carried out using the L-BFGS-B quasi-Newton method with the analytic gradient (at most 600 iterations, with tolerances `ftol=1e-11` and `gtol=1e-7`).

For each value of $a$, the optimizer is initialized from a fresh uniform sample of the unit disk, projected into ${x_2\ge a}$. To further guard against finite-$n$ metastability, after each minimization the particles lying within

```math
0.5\,n^{-1/2}
```

of the constraint line are displaced into the interior and the energy is re-minimized. The perturbed configuration is retained only if it has strictly lower energy.

In the figure, a particle is classified as belonging to the singular layer when its second coordinate is within $0.5,n^{-1/2}$ of $a$, and as bulk otherwise. This cutoff is used only for visualization and enters neither the energy nor the optimization. The strip below each panel is a histogram of the first coordinate of the wall particles.

The computation is deterministic (fixed random seed). The values shown are

```math
a \in \{-1.0,\,-0.3,\,0.5,\,0.9,\,1.2,\,1.5\}.
```

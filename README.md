[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Proximal DC Algorithm for Sample Average Approximation of Chance Constrained Programming

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper [A Proximal DC Algorithm for Sample Average Approximation of Chance Constrained Programming](xx) by Peng Wang, Rujun Jiang, Qingyuan Kong, and Laura Balzano.

**Important: This code is being developed on an on-going basis at https://github.com/peng8wang/IJOC-CCP. Please go there if you would like to get a more recent version or would like support**

## cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/xx (**TODO**)

https://doi.org/xx (**TODO**) 

Below is the BibTex for citing this snapshot of the repository.

```
@misc{wang2025proximal,
  author =        {Peng Wang and Rujun Jiang and Qingyuan Kong and Laura Balzano},
  publisher =     {INFORMS Journal on Computing},
  title =         {{A Proximal DC Algorithm for Sample Average Approximation of Chance Constrained Programming}}, 
  year =          {2025},
  doi =           {10.1287/ijoc.2024.0648.cd},
  url =           {https://github.com/INFORMSJoC/2024.0648},
}
```

## Description
The goal of this software is to demonstrate the effectiveness of the proximal DC method proposed in this paper for the chance constrained programming, as compared to other methods, as well as the effectiveness of the model when applied to empirical study. 

## Code
In order to run this software, you must install Gurobi 9.5.2 from https://www.gurobi.com/downloads/gurobi-software/. This code can be run in MATLAB R2022b.

This directory contains the following folders, each of which corresponds to an experiment in the paper: 
* `VaR-Porfolio`: The VaR-constrained mean-variance portfolio selection problem.
* `PTP-Convex`: The Probabilistic transportation problem with a convex objective
* `PTP-Nonconvex`: The Probabilistic transportation problem with a nonconvex objective
* `Nonlinear-CCP`: Linear optimization with a joint convex nonlinear chance constraint

In each of the folders above, you will find some or all of the following functions:
* `main_xxxx.m`: main file to run the codes
* `MIP.m`: the implementation of the mixed-integer program (MIP) in the paper [Solving chance-constrained stochastic programs via sampling and integer programming](https://pubsonline.informs.org/doi/10.1287/educ.1080.0048).
* `CVaR.m`: the implementation of the CVaR in the paper [Convex approximations of chance constrained programs](https://epubs.siam.org/doi/10.1137/050622328).
* `BiCVaR.m`: the implementation of the bisection-based CVaR method in the paper [An augmented Lagrangian decomposition method for chance-constrained optimization problems](https://pubsonline.informs.org/doi/10.1287/ijoc.2020.1001).
* `DCA.m`: the implementation of Algorithm 1 without proximal term in our paper.
* `pDCA.m`: the implementation of Algorithm 1 in our paper.
* `SCA.m`:  the implementation of the successive convex approximation method (SCA) in the paper [Squential convex approximations to joint chance constrained programs: A Monte Carlo approach](https://pubsonline.informs.org/doi/10.1287/opre.1100.0910).
* `ALDM.m`, `ALDM_update_x.m`, `ALDM_update_y.m`: the implementation of the augmented Lagrangian decomposition method (ALDM) in the paper [An augmented Lagrangian decomposition method for chance-constrained optimization problems](https://pubsonline.informs.org/doi/10.1287/ijoc.2020.1001).
* `post_processing.m`: the code for calculating the maximum, minimum, and mean values of the indicators.
* `risk_level.m`: the code for calculating the risk level of each method.
* `Nonlinear-CCP/gensample.m`: the code for generating the $d \times m$ matrix of random variables $\xi$.



## Support

For support in using this software, submit an [issue](https://github.com/peng8wang/IJOC-CCP/issues/new). This code is being developed on an on-going basis at the author's [Github page](https://github.com/peng8wang/IJOC-CCP).




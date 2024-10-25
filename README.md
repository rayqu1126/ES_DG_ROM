# Entropy stable reduced order modeling (ES-ROM) of nonlinear conservation laws using discontinuous Galerkin (DG) methods
## 1D Domains

- **`FOM_1D.jl`**: Full order model (FOM) for 1D periodic domains  
- **`FOM_1D_weak.jl`**: FOM for 1D domains with weakly imposed boundary conditions (implements reflective wall)  
- **`ROM_1D.jl`**: Galerkin projection ROM for 1D domains (without hyper-reduction)  
- **`HR_1D.jl`**: Hyper-reduced ROM for 1D periodic domains  
- **`HR_1D_hybridized.jl`**: Hyper-reduced ROM for 1D domains with weakly imposed boundary conditions (implements reflective wall) using a hybridized operator

## 2D Domains
- **`FOM_2D.jl`**: Full order model (FOM) for 2D periodic domains  
- **`FOM_2D_weak.jl`**: FOM for 2D domains with weakly imposed boundary conditions (implements reflective wall)  
- **`ROM_2D.jl`**: Galerkin projection ROM for 2D domains (without hyper-reduction)  
- **`HR_2D.jl`**: Hyper-reduced ROM for 2D periodic domains  
- **`HR_2D_hybridized.jl`**: Hyper-reduced ROM for 2D domains with weakly imposed boundary conditions (implements reflective wall) using hybridized operators

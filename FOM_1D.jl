using Plots, LinearAlgebra
using SparseArrays, StaticArrays
using OrdinaryDiffEq, StartUpDG, Trixi

N = 3 # polynomial degree of the DG approximation
num_elements = 256
epsilon = 0.01 # viscosity coefficient

equations = InviscidBurgersEquation1D()
# equations = LinearScalarAdvectionEquation1D(1.0)
# equations = CompressibleEulerEquations1D(1.4)

Trixi.entropy2cons(u, ::LinearScalarAdvectionEquation1D) = u

Trixi.flux_ec(u_ll, u_rr, orientation, equations::LinearScalarAdvectionEquation1D) = 
    Trixi.flux_central(u_ll, u_rr, orientation, equations)

Trixi.flux_ec(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D) = 
    Trixi.flux_ranocha(u_ll, u_rr, orientation, equations)

rd = RefElemData(Line(), SBP(), N)
md = MeshData(uniform_mesh(Line(), num_elements), rd; is_periodic=true)

function initial_condition(x, ::Union{<:LinearScalarAdvectionEquation1D, InviscidBurgersEquation1D})
    return SVector(0.5 - sin(pi * x))
end

function initial_condition(x, equations::CompressibleEulerEquations1D)
    rho = 1.0 + exp(-100 * x^2)
    v_1 = 0.0
    p = rho^equations.gamma
    return Trixi.prim2cons(SVector(rho, v_1, p), equations)
end
x = vec(md.x)
u = initial_condition.(x, equations)

# Pre-compute global DG matrices M, Q, B
M = kron(I(num_elements), rd.M .* md.J[1,1])  # assume J is same over each element
M_inv = kron(I(num_elements), inv(rd.M .* md.J[1,1]))

Q = rd.M * rd.Dr
S = Q - Q'

Q = kron(I(num_elements), S)
for i = N+1 : N+1 : (num_elements-1) * (N+1)
    Q[i,i+1] = 1
    Q[i+1,i] = -1
end 
Q[end,1] = 1
Q[1,end] = -1

Q = 0.5 * sparse(Q)
B = Q - kron(I(num_elements), 0.5 * S)
B = sparse(abs.(B))

function rhs!(du, u, p, t)
    (; M_inv, Q, B, epsilon, equations) = p

    fill!(du, zero(eltype(du)))

    cols = rowvals(Q) 
    for i in eachindex(du)
        for id in nzrange(Q, i)
            j = cols[id]
            if i > j
                fij = flux_ec(u[i], u[j], 1, equations)
                QFij = 2 * Q[i,j] * fij
                du[i] += QFij
                du[j] -= QFij
            end
        end
    end

    # interface dissipation
    if params.use_interface_dissipation == true
        cols = rowvals(B) 
        for i in eachindex(du)
            for id in nzrange(B, i)
                j = cols[id]            
                if i > j
                    lambda = max_abs_speed_naive(u[i], u[j], 1, equations)
                    Dij = 0.5 * lambda * (u[i] - u[j])
                    BDij = B[i,j] * Dij
                    du[i] += BDij
                    du[j] -= BDij
                end
            end
        end
    end

    du .= -M_inv * du

    # viscous terms
    sigma = M_inv * (Q * u)
    du .-= epsilon * M_inv * Q' * sigma 
end

# our ROM currently doesn't incorporate interface dissipation 
# (e.g., upwinding), though it can be added to the FOM.
use_interface_dissipation=false

params = (; M, M_inv, Q, B, 
            use_interface_dissipation, 
            equations, epsilon)
tspan = (0.0, 0.75)
ode = ODEProblem(rhs!, u, tspan, params)

sol = solve(ode, Tsit5(), reltol = 1e-9, abstol = 1e-11,
            saveat=LinRange(tspan[1], tspan[2], 300), 
            callback=Trixi.AliveCallback(alive_interval=1000))

plot(x, getindex.(sol.u[end], 1), label="FOM")

using Plots, LinearAlgebra
using SparseArrays, StaticArrays
using OrdinaryDiffEq, StartUpDG, Trixi

N = 3 # polynomial degree of the DG approximation
num_elements = 512
epsilon = 0.0002 # viscosity coefficient

# equations = InviscidBurgersEquation1D()
# equations = LinearScalarAdvectionEquation1D(1.0)
equations = CompressibleEulerEquations1D(1.4)

Trixi.entropy2cons(u, ::LinearScalarAdvectionEquation1D) = u

Trixi.flux_ec(u_ll, u_rr, orientation, equations::LinearScalarAdvectionEquation1D) = 
    Trixi.flux_central(u_ll, u_rr, orientation, equations)

Trixi.flux_ec(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D) = 
    Trixi.flux_ranocha(u_ll, u_rr, orientation, equations)

rd = RefElemData(Line(), SBP(), N)

(VX,), EToV = uniform_mesh(Line(), num_elements)
@. VX = 0.5 * (1 + VX) # map to [0, 1]
# @. VX = 0.5 * VX # map to [-0.5, 0.5]
md = MeshData((VX,), EToV, rd)

function initial_condition(x, ::Union{<:LinearScalarAdvectionEquation1D, InviscidBurgersEquation1D})
    return SVector(0.5-sin(pi * x))
end

function initial_condition(x, equations::CompressibleEulerEquations1D)
    # Iinitial conditions for example 1
    rho = 2 + 0.5 * exp(-100 * (x-0.5)^2)
    v_1 = 0.1 * exp(-100 * (x-0.5)^2)
    p = rho^equations.gamma

    # Initial conditions for exmaple2 (sod shock tube)
    # rho = 0.125 + 0.875/(1+exp(100*x))
    # p = 0.1 + 0.9/(1+exp(100*x))
    # v_1 = 0

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

Q[1,1], Q[end,end] = -1,1

Q = 0.5 * sparse(Q)


function rhs!(du, u, p, t)
    (; M_inv, Q, epsilon, equations) = p

    fill!(du, zero(eltype(du)))
    u1P = SVector(u[1][1], -u[1][2], u[1][3])
    uNP = SVector(u[end][1], -u[end][2], u[end][3])

    cols = rowvals(Q) 
    # flux evaluation
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

    du[1] -= flux_ec(u[1], u1P, 1, equations)
    du[end] += flux_ec(u[end], uNP, 1, equations)

    du .= -M_inv * du

    # viscous terms
    sigma = M_inv * (Q * u)
    du .-= epsilon * M_inv * Q' * sigma
end


h = estimate_h(rd,md)
params = (; M, M_inv, Q, 
            equations, epsilon, h, N = rd.N)
tspan = (0.0, 0.75)
ode = ODEProblem(rhs!, u, tspan, params)

function eigen_est(integrator)
    (; epsilon, h, N) = integrator.p
    integrator.eigen_est = max((N+1)^2 * inv(h), epsilon * inv(h^2) * (N+1)^4)
end


sol = solve(ode, RK4(), reltol = 1e-9, abstol = 1e-11,
            saveat=LinRange(tspan[1], tspan[2], 400), 
            callback=Trixi.AliveCallback(alive_interval=1000))

plot(x, getindex.(sol.u[end], 1), label="FOM")
using Trixi
using NonNegLeastSquares
using LinearAlgebra

trixi_include("FOM_1D_weak.jl", N=3, num_elements=512)

trixi_include("ROM_1D.jl", Nmodes = 20,
              weight_matrix=sqrt(M))

# pass in weight_matrix in a hopefully type stable fashion via closure
function create_trim_basis(weight_matrix=I)
    function trim_basis(V; tol=1e-14)
        U, s, _ = svd(weight_matrix * V)
        return weight_matrix \ U[:, findall(s .> tol)]
    end
    return trim_basis
end

trim_basis = create_trim_basis(weight_matrix)

function greedy_hyperreduction(Vtarget, b, tol)
    r = b
    w = eltype(b)[]
    I_index = Int[]
    Vtarget_norm = zeros(size(Vtarget))
    for i in axes(Vtarget, 1)
        Vtarget_norm[i,:] = Vtarget[i,:] / norm(Vtarget[i,:])
    end
    while norm(r) / norm(b) > tol
        tmpmax = -Inf
        imax = 1
        for i = 1:size(Vtarget, 1)
            if !(i in I_index)
                val = sum(Vtarget_norm[i,:] .* r) / norm(r)
                if (val > tmpmax)
                    tmpmax = val
                    imax   = i
                end
            end
        end
        append!(I_index, imax)
        VI = copy(Vtarget[I_index, :]')

        # Least squares solve
        w = VI \ b
        if any(w .<= 0)
            w = nonneg_lsq(VI, b; alg=:nnls)             # nonnegative least square
        end

        r = b - VI * w
        #@show norm(r), size(I_index)
    end

    VI = copy(Vtarget[I_index, :]')
    w = VI \ b
    if any(w .<= 0)
        w = vec(nonneg_lsq(VI, b; alg=:nnls))             # nonnegative least square
    end
    return I_index, w
end

# Caratheodory pruning
function caratheodory_pruning(V, w_in)

    if length(w_in) <= size(V, 2)
        return w_in, eachindex(w_in)
    end
    w = copy(w_in)
    M, N = size(V)
    inds = collect(1:M)
    m = M-N
    Q, _ = qr(V)
    Q = copy(Q)
    for _ in 1:m
        kvec = Q[:,end]

        # for subtracting the kernel vector
        idp = findall(@. kvec > 0)
        alphap, k0p = findmin(w[inds[idp]] ./ kvec[idp])
        k0p = idp[k0p]

        # for adding the kernel vector
        idn = findall(@. kvec < 0);
        alphan, k0n = findmax(w[inds[idn]] ./ kvec[idn])
        k0n = idn[k0n];

        alpha, k0 = abs(alphan) < abs(alphap) ? (alphan, k0n) : (alphap, k0p)
        w[inds] = w[inds] - alpha * kvec
        deleteat!(inds, k0)
        Q, _ = qr(V[inds, :])
        Q = copy(Q)
    end
    return w[inds], inds
end


V_test = trim_basis([ones(size(V_ROM, 1)) V_ROM (M \ (Q' * V_ROM))])

# hyperreduction
# this tolerance is purely used for HR
tol = sqrt(sum(svd_values[Nmodes+1:end].^2) / sum(svd_values.^2))
create_target_basis(V1, V2; tol) = 
    trim_basis(hcat([V1[:,i] .* V2[:,j] for i in axes(V1,2), j in axes(V2,2)]...); tol)

e = ones(size(V_ROM, 1))
V_target = trim_basis([e create_target_basis(V_ROM, V_ROM; tol)])
ids_HR, w_HR = greedy_hyperreduction(V_target, V_target' * M.diag, tol)
# w_HR, ids_HR = caratheodory_pruning(V_target, M.diag)

## Ideal hyper-reduction
# w_HR = M.diag
# ids_HR = eachindex(M.diag)

# hyperreduced operators
V_HR = V_ROM[ids_HR, :]
M_HR = V_HR' * Diagonal(w_HR) * V_HR
P_HR = M_HR \ (V_HR' * Diagonal(w_HR))

M_test_HR = V_test[ids_HR,:]' * Diagonal(w_HR) * V_test[ids_HR,:]


# add stabilization points
eigenTol = 1e-8
lambda , Mtvectors= eigen(Symmetric(M_test_HR))
if  minimum(lambda) < eigenTol 
    Z = V_test * Mtvectors[:,findall(lambda .< eigenTol)]
    Z_target = hcat((Z[:,i] .* Z[:,j] for i in axes(Z,2) for j in axes(Z,2))...)
    alphaC = 1e-2
    Vtarget_stabilized = [V_target sqrt(alphaC) * Z_target]
    b_stabilized = [V_target' * M.diag; sqrt(alphaC) * Z_target' * vec(md.wJq)]
    ids_HR, w_HR= greedy_hyperreduction(Vtarget_stabilized, b_stabilized, tol)
    V_HR = V_ROM[ids_HR, :]
    M_HR = V_HR' * Diagonal(w_HR) * V_HR
    P_HR = M_HR \ (V_HR' * Diagonal(w_HR))
    M_test_HR = V_test[ids_HR,:]' * Diagonal(w_HR) * V_test[ids_HR,:]
end


if cond(M_test_HR) > 1e8
    @warn "Condition number of test mass matrix is $(cond(M_test_HR))"
end
P_test = M_test_HR \ (V_test[ids_HR,:]' * Diagonal(w_HR))
Vf_test = V_test[[1,end], :]

# hybridized "hyperreduced" operators
Qt = P_test' * V_test' * Q * V_test * P_test
Bf = [-1 0; 0 1]
E = V_test[[1,end], :] * P_test
Q_HR = 0.5 * [Qt-Qt' E' * Bf;
              -Bf * E   Bf]
Q_HR = 0.5 * (Q_HR - Q_HR')

Vf = V_ROM[[1,end],:]
Vh = [V_HR; Vf] 

# advective and viscous operator (use HR mass matrix)
invM_K_ROM = M_HR \ (V_ROM' * Q' * inv(M) * Q * V_ROM)

function rhs_HR!(du_ROM, u_ROM, p, t)
    (; du_HR, invM_VTr_HR, P_HR, V_HR, Q_HR, Vh, invM_VfTr, epsilon, equations) = p

    u = entropy2cons.(Vh * (P_HR * cons2entropy.(V_HR * u_ROM, equations)), equations)

    fill!(du_HR, zero(eltype(du_HR)))
    for i in axes(Q_HR, 1)
        for j in axes(Q_HR, 2)
            fij = flux_ec(u[i], u[j], 1, equations)
            QFij = 2 * Q_HR[i,j] * fij
            du_HR[i] += QFij
        end
    end

    uM = SVector(SVector(u[end-1]),SVector(u[end]))
    uP = SVector(SVector(u[end-1][1],-u[end-1][2],u[end-1][3]), SVector(u[end][1],-u[end][2],u[end][3]))
    nxJ = SVector(-1., 1.)
    interface_flux = @. flux_ec(uP, uM, 1, equations) .* nxJ
    
    du_ROM .= -((invM_VTr_HR * du_HR) + invM_VfTr * (interface_flux) + epsilon * invM_K_ROM * u_ROM)
end

params = (; ode.p..., ode_ROM.p..., 
            invM_VTr_HR = M_HR \ Vh', invM_VfTr = M_HR \ Vf', P_HR, V_HR, Vh, Vf, Q_HR,             
            du_HR = similar(ode.u0, size(Q_HR, 2)))

ode_HR = ODEProblem(rhs_HR!, pinv(V_ROM) * ode.u0, ode.tspan, params)
sol_HR = solve(ode_HR, RK4(), reltol = 1e-8, abstol = 1e-10,
               saveat=LinRange(tspan[1], tspan[2], 400), 
               callback=Trixi.AliveCallback(alive_interval=50))

L2_error_HR = sqrt(sum(M * norm.(sol.u[end] - V_ROM * sol_HR.u[end]).^2)) / sqrt(sum(M*norm.(sol.u[end]).^2))

println("ROM L2_error = $L2_error with $Nmodes modes")
println("HR L2 error = $L2_error_HR with $(length(w_HR)) HR points")

plot(x, getindex.(sol.u[end], 1), label="FOM")
plot!(x, getindex.(V_ROM * sol_ROM.u[end], 1), linestyle=:dash, label="ROM")
plot!(x, getindex.(V_ROM * sol_HR.u[end], 1), label="HR-ROM")
scatter!(x[ids_HR], getindex.(V_ROM[ids_HR,:] * sol_HR.u[end], 1), label="", ms=2)




using Trixi
using NonNegLeastSquares
using LinearAlgebra, Plots

trixi_include("FOM_2D.jl")

trixi_include("ROM_2D.jl", Nmodes = 50,
              weight_matrix=sqrt(kron(Diagonal(md.J[1,:]), rd.M)))

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
        @show norm(r), size(I_index)
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

# this tolerance is purely used for HR
tol = sqrt(sum(svd_values[Nmodes+1:end].^2) / sum(svd_values.^2))
Vx_test = trim_basis([ones(size(V_ROM, 1)) [V_ROM M \ (Qx' * V_ROM)]])
Vy_test = trim_basis([ones(size(V_ROM, 1)) [V_ROM M \ (Qy' * V_ROM)]])

# hyperreduction
create_target_basis(V1, V2; tol) = 
    trim_basis(hcat([V1[:,i] .* V2[:,j] for i in axes(V1,2), j in axes(V2,2)]...); tol)

e = ones(size(V_ROM, 1))
V_target = create_target_basis(V_ROM, V_ROM; tol)
ids_HR, w_HR = greedy_hyperreduction(V_target, V_target' * M.diag, tol)

Mx_test_HR = Vx_test[ids_HR,:]' * Diagonal(w_HR) * Vx_test[ids_HR,:]
My_test_HR = Vy_test[ids_HR,:]' * Diagonal(w_HR) * Vy_test[ids_HR,:]

if cond(Mx_test_HR) > 1e8 || cond(My_test_HR) > 1e8  
    @warn "Condition number of test mass matrix is $(cond(Mx_test_HR)), $(cond(My_test_HR))"
end
Px_test = Mx_test_HR \ (Vx_test[ids_HR,:]' * Diagonal(w_HR))
Py_test = My_test_HR \ (Vy_test[ids_HR,:]' * Diagonal(w_HR))

Qx_HR = Px_test' * (Vx_test' * Qx * Vx_test) * Px_test
Qy_HR = Py_test' * (Vy_test' * Qy * Vy_test) * Py_test

# advective and viscous operator (use HR mass matrix)
V_HR = V_ROM[ids_HR, :]
M_HR = V_HR' * Diagonal(w_HR) * V_HR
P_HR = M_HR \ (V_HR' * Diagonal(w_HR))
invM_K_ROM = M_HR \ (V_ROM' * Qx' * inv(M) * Qx * V_ROM) + M_HR \ (V_ROM' * Qy' * inv(M) * Qy * V_ROM)


function rhs_HR!(du_ROM, u_ROM, p, t)
    (; du_HR, invM_VTr, P_HR, V_HR, Qx_HR, Qy_HR, epsilon, equations) = p

    u = entropy2cons.(V_HR * (P_HR * cons2entropy.(V_HR * u_ROM, equations)), equations)
    # u = V_HR * u_ROM

    fill!(du_HR, zero(eltype(du_HR)))
    for i in axes(Qx_HR, 1)
        for j in axes(Qx_HR, 2)
            if i > j
                fij = flux_ec(u[i], u[j], 1, equations)
                QFij = 2 * Qx_HR[i,j] * fij
                du_HR[i] += QFij
                du_HR[j] -= QFij
            end
        end
    end

    for i in axes(Qy_HR, 1)
        for j in axes(Qy_HR, 2)
            if i > j
                fij = flux_ec(u[i], u[j], 2, equations)
                QFij = 2 * Qy_HR[i,j] * fij
                du_HR[i] += QFij
                du_HR[j] -= QFij
            end
        end
    end

    du_ROM .= -((invM_VTr * du_HR)  + epsilon * invM_K_ROM * u_ROM)
end

params = (; ode.p..., 
            invM_VTr = M_HR \ V_HR', P_HR, V_HR, Qx_HR, Qy_HR,
            du_HR = similar(ode.u0, size(w_HR)))
ode_HR = ODEProblem(rhs_HR!, pinv(V_ROM) *  reshape(ode.u0,FOM_dim,:), ode.tspan, params)
sol_HR = solve(ode_HR, Tsit5(), reltol = 1e-8, abstol = 1e-8,
               saveat=LinRange(tspan[1], tspan[2], 400), 
               callback=Trixi.AliveCallback(alive_interval=10))
L2_error_HR = sqrt(sum(M * norm.(reshape(sol.u[end],FOM_dim,:)- V_ROM * sol_HR.u[end]).^2))  / sqrt(sum(M*norm.(reshape(sol.u[end],FOM_dim,:)).^2))

# println("ROM L2_error = $L2_error with $Nmodes modes")
println("HR L2 error = $L2_error_HR with $(length(w_HR)) HR points")

xp, yp = vec(rd.Vp * md.x), vec(rd.Vp * md.y)
scatter(xp, yp, zcolor=vec(rd.Vp * reshape(StructArrays.component(V_ROM * sol_HR.u[end],1), (N+1)^2, md.num_elements)) , 
markersize=2, markerstrokewidth=0, axis=([], false), 
ticks = false, aspect_ratio=:equal, clims=(0.95, 2.1), colorbar = false)


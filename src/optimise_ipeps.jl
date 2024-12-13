@kwdef mutable struct iPEPSOptimize
    boundary_alg::VUMPS
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    optimizer = Defaults.optimizer
    folder::String = Defaults.folder
    show_every::Int = Defaults.show_every
    save_every::Int = Defaults.save_every
end

"""
    indexperm_symmetrize(ipeps)

return the symmetrized `ipeps` by permuting the indices of the `ipeps` to ensure
```
        4
        │
 1 ── ipeps ── 3
        │
        2
```
"""
function indexperm_symmetrize(ipeps)
    # ipeps += permutedims(ipeps, (3,2,1,4,5)) 
    # ipeps += permutedims(ipeps, (1,2,4,3,5)) 
    # ipeps += permutedims(ipeps, (4,2,3,1,5)) 
    # ipeps += permutedims(ipeps, (3,2,4,1,5)) 
    return ipeps / norm(ipeps)
end

function bulid_ABBA(A)
    D, d, Ni, Nj = size(A)[[1,5,6,7]]
    # B = permutedims(A, (3,4,1,2,5))
    ap = [((i+j) % 2 == 0 ? reshape(ein"abcde,fghmn->afbgchdmen"(A[:,:,:,:,:,i,j], conj(A[:,:,:,:,:,i,j])), D^2,1,D^2,D^2, d,d) : reshape(ein"abcde,fghmn->afbgchdmen"(permutedims(A[:,:,:,:,:,i,j], (3,4,1,2,5)), conj(permutedims(A[:,:,:,:,:,i,j], (3,4,1,2,5)))), D^2,D^2,D^2,1, d,d)) for i in 1:Ni, j in 1:Nj]
    M = [ein"abcdee->abcd"(ap) for ap in ap]
    # M = [(i==j ? A : B) for i in 1:2, j in 1:2]
    return ap, M
end

"""
    init_ipeps(;atype = Array, Ni::Int, Nj::Int, D::Int)

return a random `ipeps` with bond dimension `D` and physical dimension 2.
"""
function init_ipeps(;atype = Array, file=nothing, D::Int, d::Int, Ni::Int, Nj::Int)
    if file !== nothing
        A = load(file, "bcipeps")
    else
        A = rand(ComplexF64, D,1,D,D,d, Ni,Nj)
        A /= norm(A)
        # _, M = bulid_ABBA(A)
        # rt = VUMPSRuntime(M, χ, params.boundary_alg)
        # rt′ = leading_boundary(rt, M, params.boundary_alg)
        # Zygote.@ignore params.reuse_env && update!(rt, rt′)
        # n = 1
        # Zygote.@ignore begin
        #     env = VUMPSEnv(rt′, M)
        #     @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
        #     # n, _ = rightenv(ARu, conj(ARd), M, FLo; ifobs=true) 
        #     λFLo, _ =  rightenv(ARu, conj.(ARd), M; ifobs=true)
        #     λC, _ = rightCenv(ARu, conj.(ARd);    ifobs=true)
        #     n = prod(λFLo./λC)
        # end
        # A /= sqrt(n)
    end
    return atype(A)
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(A, h, rt, oc, params::iPEPSOptimize)
    # n = 1
    # _, M = bulid_ABBA(A)
    # Zygote.@ignore begin
    #     @unpack AR = rt
    #     λFLo, _ =  rightenv(AR, conj.(AR), M; ifobs=true)
    #     λC, _ = rightCenv(AR, conj.(AR);    ifobs=true)
    #     n = prod(λFLo./λC)
    #     @show  "*************************" n "*************************"
    # end
    # A /= n^(1/8)
    ap, M = bulid_ABBA(A)
    params.verbosity >= 4 && println("for convergence")
    Zygote.@ignore update!(rt, leading_boundary(rt, M, params.boundary_alg))

    params.verbosity >= 4 && println("real AD calculation")
    params_diff = deepcopy(params)
    params_diff.boundary_alg.maxiter = 10
    rt′ = leading_boundary(rt, M, params_diff.boundary_alg)

    Zygote.@ignore params.reuse_env && update!(rt, rt′)
    env = VUMPSEnv(rt′, M)
    # return TeneT.checkpoint(expectation_value, h, ap, env, oc, params)
    
    return expectation_value(h, ap, M, env, oc, params)
end

"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A::AbstractArray, h, χ::Int, params::iPEPSOptimize, file=nothing)
    D = size(A, 1)
    oc = optcont(D, χ)
    # A = indexperm_symmetrize(A)
    if file !== nothing
        atype = TeneT._arraytype(A)
        rt = atype(load(file, "rt"))
        params.verbosity >= 4 && @info "load env from $file"
    else
        _, M = bulid_ABBA(A)
        rt = VUMPSRuntime(M, χ, params.boundary_alg)
        params.verbosity >= 4 && @info "random initial env"
    end

    function f(A) 
        return real(energy(A, h, rt, oc, params))
    end
    function g(A)
        grad = Zygote.gradient(f,A)[1]
        return grad
    end
    res = optimize(f, g, 
        A, params.optimizer, inplace = false,
        Optim.Options(f_tol=params.tol, 
                      iterations=params.maxiter,
                      extended_trace=true,
                      callback=os->writelog(os, params, rt, D, χ)
        )
    )
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, params::iPEPSOptimize, rt, D::Int, χ::Int)
    @unpack folder = params

    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy = %.15f \tgnorm = %.3e\n", os.iteration, os.metadata["time"], os.value, os.g_norm)

    maxiter = params.boundary_alg.maxiter
    folder = joinpath(folder, "D$(D)_χ$(χ)_maxiter$(maxiter)")
    !(ispath(folder)) && mkpath(folder)
    if params.verbosity >= 3 && os.iteration % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end
    if params.save_every != 0 && os.iteration % params.save_every == 0
        save(joinpath(folder, "ipeps", "ipeps_No.$(os.iteration).jld2"), "bcipeps", Array(os.metadata["x"]))
        save(joinpath(folder, "env", "env.jld2"), "rt", Array(rt))
    end

    return false
end
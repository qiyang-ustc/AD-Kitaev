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
    D, d = size(A)[[1,5]]
    B = permutedims(A, (3,4,1,2,5))
    ap = [(i==j ? reshape(ein"abcde,fghmn->afbgchdmen"(A, conj(A)), D^2,1,D^2,D^2, d,d) : reshape(ein"abcde,fghmn->afbgchdmen"(B, conj(B)), D^2,D^2,D^2,1, d,d)) for i in 1:2, j in 1:2]
    M = [ein"abcdee->abcd"(ap) for ap in ap]
    # M = [(i==j ? A : B) for i in 1:2, j in 1:2]
    return ap, M
end

"""
    init_ipeps(;atype = Array, Ni::Int, Nj::Int, D::Int)

return a random `ipeps` with bond dimension `D` and physical dimension 2.
"""
function init_ipeps(;atype = Array, file=nothing, D::Int, d::Int)
    if file !== nothing
        A = load(file, "bcipeps")
    else
        A = rand(ComplexF64, D,1,D,D,d)
        A /= norm(A)
    end
    return atype(A)
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(A, h, rt, oc, params::iPEPSOptimize)
    # A = indexperm_symmetrize(A)
    ap, M = bulid_ABBA(A)
    rt′ = leading_boundary(rt, M, params.boundary_alg)
    Zygote.@ignore params.reuse_env && update!(rt, rt′)
    env = VUMPSEnv(rt′, M)
    return expectation_value(h, ap, env, oc, params)
end

"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A::AbstractArray, h, χ::Int, params::iPEPSOptimize)
    D = size(A, 1)
    oc = optcont(D, χ)

    # A = indexperm_symmetrize(A)
    _, M = bulid_ABBA(A)
    rt = VUMPSRuntime(M, χ, params.boundary_alg)

    function f(A) 
        return real(energy(A, h, rt, oc, params))
    end
    function g(A)
        # f(x)
        grad = Zygote.gradient(f,A)[1]
        return grad
    end
    res = optimize(f, g, 
        A, params.optimizer, inplace = false,
        Optim.Options(f_tol=params.tol, 
                      iterations=params.maxiter,
                      extended_trace=true,
                      callback=os->writelog(os, params, D, χ)
        )
    )
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, params::iPEPSOptimize, D::Int, χ::Int)
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
        save(joinpath(folder, "ipeps_No.$(os.iteration).jld2"), "bcipeps", Array(os.metadata["x"]))
    end

    return false
end
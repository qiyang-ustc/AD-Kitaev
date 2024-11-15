using AD_Kitaev
using Random
using CUDA
using TeneT

Random.seed!(100)
atype = CuArray
D, χ = 3, 40
No = 0
S = 0.5
model = Kitaev(S,-1.0,-1.0,-1.0)
maxiter = 10
if No == 0
    file = nothing
else
    file = "data/fold/$model/D$(D)_χ$(χ)_maxiter$(maxiter)/ipeps_No.$(No).jld2"
end
h = hamiltonian(model)
A = init_ipeps(;atype, D=D, file=file, d=Int(2*S+1)^2)
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     maxiter=maxiter, 
                     miniter=1,
                     verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg, 
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=100,
                       tol=1e-10,
                       folder="data/fold/$model/"
)
optimise_ipeps(A, h, χ, params)
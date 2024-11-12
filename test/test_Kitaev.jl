using AD_Kitaev
using Random
using CUDA
using TeneT

Random.seed!(100)
atype = Array
D, χ = 3, 20
No = 0
S = 1//2
model = Kitaev(S,-1.0,-1.0,-1.0)
if No == 0
    file = nothing
else
    file = "data/$model/D$(D)_χ$(χ)_maxiter$(maxiter)/ipeps_No.$(No).jld2"
end
h = atype.(hamiltonian(model))
A = init_ipeps(;atype, D=D, file=file, d=Int(2*S+1))
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     maxiter=10, 
                     miniter=1,
                     verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg, 
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=100,
                       tol=1e-10,
                       folder="data/$model/"
)
optimise_ipeps(A, h, χ, params)
using AD_Kitaev
using Random
using CUDA
using TeneT

Random.seed!(100)
atype = CuArray
D, χ = 4, 30
No = 30
S = 0.5
model = Kitaev(S,-1.0,-1.0,-1.0)
maxiter = 10
if No == 0
    file = nothing
else
    file = "data/brickwall/$model/D$(D)_χ$(χ)_maxiter$(maxiter)/ipeps_No.$(No).jld2"
end
h = atype.(hamiltonian(model))
A = init_ipeps(;atype, D=D, file=file, d=Int(2*S+1))
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     maxiter=maxiter, 
                     miniter=3,
                     verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg, 
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=100,
                       tol=1e-10,
                       folder="data/brickwall/$model/"
)
optimise_ipeps(A, h, χ, params)
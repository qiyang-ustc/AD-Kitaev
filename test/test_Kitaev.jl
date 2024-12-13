using AD_Kitaev
using Random
using CUDA
using TeneT

Random.seed!(100)
atype = Array
Ni, Nj = 2, 2
D, χ = 2, 10
No = 71
S = 0.5
model = Kitaev(S,-1.0,-1.0,-1.0)
maxiter = 20
if No == 0
    ipeps_file = nothing
    env_file = nothing
else
    ipeps_file = "data/brickwall/$model/$(Ni)x$(Nj)/D$(D)_χ$(χ)_maxiter$(maxiter)/ipeps/ipeps_No.$(No).jld2"
    # env_file = "data/brickwall/$model/$(Ni)x$(Nj)/D$(D)_χ$(χ)_maxiter$(maxiter)/env/env.jld2"
    env_file = nothing
end
A = init_ipeps(;atype, D=D, file=ipeps_file, d=Int(2*S+1), Ni=Ni,Nj=Nj)
h = atype.(hamiltonian(model))
ind = h[1] .!= 0
sum(ind)
# boundary_alg = VUMPS(ifupdown=true,
#                      ifdownfromup=true, 
#                      maxiter=maxiter,  
#                      miniter=3,
#                      verbosity=2
# )
# params = iPEPSOptimize(boundary_alg=boundary_alg, 
#                        reuse_env=true, 
#                        verbosity=4, 
#                        maxiter=0,
#                        tol=1e-10,
#                        folder="data/brickwall/$model/$(Ni)x$(Nj)/"
# )
# optimise_ipeps(A, h, χ, params, env_file)
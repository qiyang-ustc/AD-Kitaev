using AD_Kitaev
using Random
using CUDA
using TeneT

Random.seed!(100)
atype = Array
Ni, Nj = 1, 1
D, χ = 2, 10
No = 100
S = 1.5
# model_old = Kitaev(0.5,-1.0,-1.0,-1.0/4)
model = Kitaev(S,-1.0,-1.0,-1.0)
Dz = 12.0
if No == 0
    file = nothing
else
    file = "data/merge/Dz$Dz/$model/$(Nj)x$(Nj)/D$(D)_χ$(χ)/ipeps/ipeps_No.$(No).jld2"
end
h = hamiltonian(model)
# A = AD_Kitaev.init_ipeps_form_small_spin(;atype, D=D, file=file, d=Int(2*S+1)^2, Ni=Ni, Nj=Nj)
A = AD_Kitaev.init_ipeps(;atype, D=D, file=file, d=Int(2*S+1)^2, Ni=Ni, Nj=Nj)
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=1,
                     verbosity=2
)
boundary_alg_AD = VUMPS(ifupdown=true,
                        ifdownfromup=false, 
                        ifsimple_eig=true,
                        maxiter=3, 
                        miniter=3,
                        verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg,
                       boundary_alg_AD=boundary_alg_AD,
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=0,
                       tol=1e-10,
                       folder="data/merge/Dz$Dz/$model/$(Ni)x$(Nj)/"
)
optimise_ipeps(A, model, χ, params; ifWp=false, Dz)
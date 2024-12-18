using AD_Kitaev
using Random
using CUDA
using TeneT

#####################################    parameters      ###################################
Random.seed!(42)
atype = Array
Ni, Nj = 1, 1
D, χ = 2, 20
No = 62
S = 1.0
model = Kitaev(S,1.0,1.0,1.0)
Dz = 0.0
method = :merge
# method = :brickwall
folder = "data/$method/Dz$Dz/$model/$(Ni)x$(Nj)/"
############################################################################################

boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=1,
                     maxiter_ad=3,
                     miniter_ad=3,
                     verbosity=2
)
params = iPEPSOptimize{method}(boundary_alg=boundary_alg,
                               reuse_env=true, 
                               verbosity=4, 
                               maxiter=100,
                               tol=1e-10,
                               folder=folder
)
A = init_ipeps(;atype, model, params, No, ifWp=true, D, χ=10, Ni, Nj)
optimise_ipeps(A, model, χ, params; Dz, ifWp=false)
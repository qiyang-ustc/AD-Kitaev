using AD_Kitaev
using Random
using CUDA
using TeneT

#####################################    parameters      ###################################
Random.seed!(42)
atype = Array
Ni, Nj = 1, 1
D, χ = 2, 10
No = 62
S = 1.0
ifWp = false
model = Kitaev(S,1.0,1.0,1.0)
Dz = 0.0
# method = :brickwall
method = :merge
folder = "data/$method/Dz$Dz/$model/$(Ni)x$(Nj)/"
############################################################################################

boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     ifsimple_eig=true,
                     maxiter=30, 
                     miniter=1,
                     maxiter_ad=0,
                     miniter_ad=0,
                     verbosity=2
)
params = iPEPSOptimize{method}(boundary_alg=boundary_alg,
                               reuse_env=true, 
                               verbosity=4, 
                               maxiter=1000,
                               tol=1e-10,
                               folder=folder
)
A = init_ipeps(;atype, model, params, No, ifWp, D, χ, Ni, Nj)
e, mag = observable(A, model, Dz, χ, params)

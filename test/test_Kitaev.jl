using AD_Kitaev
using Random
using CUDA
using TeneT

#####################################    parameters      ###################################
Random.seed!(42)
atype = Array
Ni, Nj = 2, 6
D, χ = 3, 10
No = 0
S = 1.5
model = Kitaev(S,1.0,1.0,1.0)
Dz = 0.0
# method = :merge
method = :brickwall
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
# A = init_ipeps(;atype, model, params, No, ifWp=false, ϵ = 5*1e-2, D, χ=20, Ni, Nj)
# A = AD_Kitaev.init_ipeps_spin111(;atype, model, params, No, ifWp=true, ϵ = 0, χ, Ni, Nj)
A = AD_Kitaev.init_ipeps_h5(;atype = Array, model, D, Ni, Nj)
optimise_ipeps(A, model, χ, params; Dz, ifWp=false)
using AD_Kitaev
using Random
using CUDA
using TeneT

#####################################    parameters      ###################################
Random.seed!(42)
atype = Array
Ni, Nj = 2, 6
D, χ = 3, 20
No = 0
S = 1.0
ifWp = false
ϵ = 5*1e-1
model = Kitaev(S,1.0,1.0,1.0)
Dz = 0.0
method = :brickwall
# method = :merge
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
# A = init_ipeps(;atype, model, params, No, ifWp, ϵ, D, χ=20, Ni, Nj)
A = AD_Kitaev.init_ipeps_h5(;atype = Array, model, file="./data/kitsShf_sikh2nfcr7D3D3.h5", D, Ni, Nj)
e, mag = observable(A, model, Dz, χ, params)

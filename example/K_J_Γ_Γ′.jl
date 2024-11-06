using AD_Kitaev
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(42)
folder = "./example/data/QQ/"
# folder = "./../../../../data/xyzhang/AD_Kitaev/"
bulk, key = init_ipeps(K_J_Γ_Γ′(0.5, -1.0, 0.0, 0.0, 0.0), 
                       [1.0,1.0,1.0], 0.0; 
                       Ni=1, Nj=1, 
                       D=2, χ=10, 
                       tol=1e-10, maxiter=50, miniter=1,
                       folder=folder, 
                       type = "_random",
                       atype = Array,
                       verbose = true
                       )

optimiseipeps(bulk, key; 
              f_tol = 1e-10, 
              opiter = 100, 
              verbose = true,
              miniter_ad = 3,
              )
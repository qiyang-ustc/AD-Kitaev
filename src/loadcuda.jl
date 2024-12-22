using CUDA
export trycuda

function trycuda()
    if CUDA.has_cuda()
        println("GPU is available!")
        return true
    else
        println("No GPU available.")
        return false
    end
end
using CUDA
export trycuda

function trycuda()
    if CUDA.has_cuda()
        println("GPU is available!")
        println("Device name: ", CUDA.device_name())
        return true
    else
        println("No GPU available.")
        return false
    end
end
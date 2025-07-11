
export bupdate!

function bupdate!(layer::BConvBias, lr::Float32)
    layer.kernel .-= layer.kernel_grad * lr
    layer.bias .-= layer.bias_grad * lr
    layer.kernel_grad = nothing
    layer.bias_grad = nothing
end


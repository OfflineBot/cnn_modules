
export bupdate!

function bupdate!(layer::BConv, lr::Float32)
    layer.kernel .-= layer.kernel_grad * lr
    layer.kernel_grad = nothing
end

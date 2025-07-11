
export update!

function update!(layer::DenseLayer, lr::Float32)
    layer.weight .-= layer.grad_weight .* lr
    layer.bias .-= layer.grad_bias .* lr
    layer.grad_bias = nothing
    layer.grad_weight = nothing
end



export backward!

function backward!(layer::DenseLayer, delta::Matrix{Float32}, dactivation::Function)::Matrix{Float32}

    delta .*= dactivation(layer.z)

    layer.grad_weight = layer.input_data' * delta
    layer.grad_bias = sum(delta, dims=1)
    new_delta = delta * layer.weight'
    layer.delta = new_delta
    return new_delta

end


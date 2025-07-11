
export forward!

function forward!(layer::DenseLayer, input::Matrix{Float32}, activation::Function)::Matrix{Float32}
    z = input * layer.weight .+ layer.bias
    a = activation(z)
    layer.z = z
    layer.a = a
    layer.input_data = input
    return a
end



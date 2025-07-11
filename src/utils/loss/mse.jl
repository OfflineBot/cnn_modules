
export mse_loss, mse_backward

mse_loss(pred::Matrix{Float32}, truth::Matrix{Float32})::Float32 = 
    sum((pred .- truth).^2) / length(truth)

function mse_backward(layer::DenseLayer, pred::Matrix{Float32}, truth::Matrix{Float32})::Matrix{Float32} 
    new_delta = 2f0 .* (pred .- truth) ./ length(pred)
    layer.delta = new_delta
    grad_weight = layer.input_data' * new_delta
    grad_bias = sum(new_delta, dims=1)
    layer.grad_weight = grad_weight
    layer.grad_bias = grad_bias

    return new_delta
end

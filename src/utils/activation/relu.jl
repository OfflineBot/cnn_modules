
export relu, deriv_relu

relu(x::AbstractArray{Float32})::AbstractArray{Float32} = ifelse.(x .<= 0.f0, 0.f0, x)

deriv_relu(x::AbstractArray{Float32})::AbstractArray{Float32} = ifelse.(x .<= 0.f0, 0.f0, 1.f0)

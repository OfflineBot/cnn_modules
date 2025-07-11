module cnn_modules


include("./conv/basic_conv/basic_conv.jl")
include("./conv/basic_conv_bias/basic_conv_bias.jl")
include("./conv/full_conv/full_conv.jl")

include("./fully_connected/fully_connected.jl")
include("./utils/utils.jl")

include("./ai/ai.jl")


end

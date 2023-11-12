using CSV, DecisionTree, DataFrames, MLJ, BetaML

function somefunc(a, b::Float64, c::Float64)
    println(a,b,c)
end

function init()
    inputs = [:SepalLengthCm, :SepalWidthCm]
    output = :Species
    
    model = DecisionTree.DecisionTreeClassifier(max_depth=3)
    
    df = CSV.read("data/Iris.csv", DataFrame)
    
    training = hcat([df[!, i] for i in inputs]...) # OBS using all data for training
    DecisionTree.fit!(model, training, df[!, output])
    
    param_tuple_expr = Expr(:tuple, Float64[:($(Symbol(p))::Float64 ) for p in inputs]...)
    
    # TODO change to take in the array in the function?
    function_body_expr = :(somefunc(model, $(inputs...)))
    # function_body_expr = :(println(model, $(join(inputs, ", "))))
    # function_body_expr = :(DecisionTree.predict(model, $(join(inputs, ", "))))
    
    # function_body_expr = :(DecisionTree.predict(model, [$(inputs)]))
    
    DecisionTree.predict(model, [10.2, 20.2])

    anon_func_expr = Expr(:->, param_tuple_expr, function_body_expr)
    
    return anon_func_expr
end

# anon_func = eval(init())

# anon_func([10.2, 20.3])

inputs = [:SepalLengthCm, :SepalWidthCm]
output = :Species

model = DecisionTree.DecisionTreeClassifier(max_depth=3)

df = CSV.read("data/Iris.csv", DataFrame)

training = hcat([df[!, i] for i in inputs]...) # OBS using all data for training
DecisionTree.fit!(model, training, df[!, output])

param_tuple_expr = Expr(:tuple, Float64[(p::Float64 ) for p in inputs]...)
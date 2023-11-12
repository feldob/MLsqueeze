using CSV, DataFrames, DecisionTree

df = CSV.read("data/Iris.csv", DataFrame)
inputs = [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]
output = :Species

a_inp_expr = Expr(:tuple, [:( $(Symbol(name))::Float64 ) for name in inputs]...)

# train the model
model = DecisionTree.DecisionTreeClassifier(max_depth=3)
training = hcat([df[!, i] for i in inputs]...) # OBS using all data for training
DecisionTree.fit!(model, training, df[!, output])

# extract the expression
exprstring = ":(DecisionTree.predict(model, [" * join(string.(inputs), ", ") * "]))"
function_body_expr = eval(Meta.parse(exprstring))

a_wrap_expr = Expr(:->, a_inp_expr, function_body_expr)

a = eval(a_wrap_expr)

a(.2,.2,.2,.2) |> println
a(400.2,400.2,400.2,40.2)  |> println
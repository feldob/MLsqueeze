using MLsqueeze, CSV, DataFrames, MLJ, MLJDecisionTreeInterface

#iris = load_iris();
# iris = DataFrames.DataFrame(iris);

# y, X = unpack(iris, ==(:target); rng=123);  

# Tree = @load DecisionTreeClassifier pkg=DecisionTree

# tree = Tree()

# mach = machine(tree, X, y)
# train, test = partition(eachindex(y), 0.7); # 70:30 split

# MLJ.fit!(mach, rows=train);

# yhat = predict_mode(mach, X[[1],:]);

df = CSV.read("data/Iris.csv", DataFrame)
    
inputs = [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]
output = :Species

ranges = deriveranges(df, inputs)

td = TrainingData("iris", df; inputs, output)

X = df[!, inputs]

df[!, output] = categorical(df[!, output])
y = df[!, output]

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

train, test = partition(eachindex(y), 0.7); # 70:30 split

mach = machine(tree, X, y)
MLJ.fit!(mach);

entry = X
entry = X[[3],:]

yhat = predict_mode(mach, entry);

# # create a classifier function that is compatible with BoundaryExposer
# modelsut = (SepalLengthCm::Float64, SepalWidthCm::Float64, PetalLengthCm::Float64, PetalWidthCm::Float64) -> MLJ.predict_mode(model, [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm])
# # create a model that can be used for classification

# be = BoundaryExposer(td, modelsut, BoundarySqueeze(MLsqueeze.ranges(td)))
# candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=false)
# df = todataframe(candidates, modelsut; output)

# plots(df, MLsqueeze.ranges(td); output)
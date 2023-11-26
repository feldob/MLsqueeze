using MLsqueeze, CSV, DataFrames

df_iris = CSV.read("data/Iris.csv", DataFrame)

inputs = [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]
output = :Species

using DecisionTree
td = TrainingData("iris", df_iris; inputs, output)
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
bs = BoundarySqueeze(td)
be = BoundaryExposer(td, modelsut, bs)
iterations=38
initial_candidates=10

candidates = apply(be; iterations, initial_candidates, optimizefordiversity=false)
df = todataframe(candidates, modelsut; output)
CSV.write("data/iris_bcs_random.csv", df)

iterations=2000

candidates = apply(be; iterations, initial_candidates)
df = todataframe(candidates, modelsut; output)
CSV.write("data/iris_bcs_div.csv", df)

p = plots(df, MLsqueeze.ranges(td); output)

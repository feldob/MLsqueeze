using MLsqueeze, CSV, DataFrames

df = CSV.read("data/titanic.csv", DataFrame)

filter!(r -> !ismissing(r.Age), df)
df.Age = convert(Vector{Float64}, df.Age)

filter!(r -> !ismissing(r.Pclass), df)
df.Pclass = convert(Vector{Float64}, df.Pclass)

df.Sex = map(d -> d == "male" ? 0.0 : 1.0, df.Sex)
df.Sex = convert(Vector{Float64}, df.Sex)

inputs = [:Pclass, :Age, :Sex, :Fare, :Parch, :SibSp]
output = :Survived


td = TrainingData("titanic", df; inputs, output)

# TODO do even for categorical, such as :Sex (setup another test)
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)
ranges = deriveranges(df, inputs)
Delta = abs.(map(r -> r[2] - r[1], ranges)) ./ 1000 # create some reasonable small delta for acceptance depending on size of the range
bs = BoundarySqueeze(MLsqueeze.ranges(td); Delta)
be = BoundaryExposer(td, modelsut, bs)

candidates = apply(be; iterations=200, initial_candidates=10)
df2 = df
df = todataframe(candidates, modelsut; output)

p = plots(df, MLsqueeze.ranges(td); output)

df.Pclass = round.(Int, df.Pclass)
df.Age = round.(Int, df.Age)
df.Parch = round.(Int, df.Parch)
df.Fare = round.(df.Fare, digits=2)
df.SibSp = round.(Int, df.SibSp)
df.Sex = map(s -> s < .5 ? "male" : "female", df.Sex)

df.n_Pclass = round.(Int, df.n_Pclass)
df.n_Age = round.(Int, df.n_Age)
df.n_Parch = round.(Int, df.n_Parch)
df.n_Fare = round.(df.n_Fare, digits=2)
df.n_SibSp = round.(Int, df.n_SibSp)
df.n_Sex = map(s -> s < .5 ? "male" : "female", df.n_Sex)

sort!(df, [:Sex, :Pclass, :Age, :Parch, :SibSp])

df
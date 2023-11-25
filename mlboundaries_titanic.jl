using CSV, DataFrames

function titanic_df()
    df = CSV.read("data/titanic.csv", DataFrame)

    filter!(r -> !ismissing(r.Age), df)
    df.Age = float.(df.Age)
    
    filter!(r -> !ismissing(r.Pclass), df)
    df.Pclass = float.(df.Pclass)
    
    df.Sex = map(d -> d == "male" ? 0.0 : 1.0, df.Sex)
    df.Sex = float.(df.Sex)

    return df
end

# define used inputs and output (OBS output must be ordinal/categorical)
inputs = [:Pclass, :Age, :Sex, :Fare, :Parch, :SibSp]

output = :Survived

using MLsqueeze

td = TrainingData("titanic", titanic_df(); inputs, output)

CSV.write("data/titanic_preprocessed.csv", td.df)

bs = BoundarySqueeze(td)

using DecisionTree

# create a ML model based on the data
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)

be = BoundaryExposer(td, modelsut, bs) # instantiate search alg
candidates = apply(be; iterations=2000, initial_candidates=10) # search and collect candidates

df = todataframe(candidates, modelsut; output)

CSV.write("data/titanic_bcs_raw.csv", df)

candidates = apply(be; iterations=20, initial_candidates=10, optimizefordiversity=false) # search and collect candidates

df = todataframe(candidates, modelsut; output)

CSV.write("data/titanic_bcs_raw_random.csv", df)

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

CSV.write("data/titanic_bcs_postprocessed.csv", df)
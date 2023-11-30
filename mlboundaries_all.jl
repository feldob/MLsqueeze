using MLsqueeze, CSV, DataFrames, DecisionTree

struct BoundaryExperiment
    name
    df
    inputs
    output

    BoundaryExperiment(name, file, inputs, output) = new(name, CSV.read(file, DataFrame), inputs, output)
end

inputstotal(exp::BoundaryExperiment) = ncol(exp.df)-1
inputsused(exp::BoundaryExperiment) = length(exp.inputs)
outputalternatives(exp::BoundaryExperiment) = length(unique(exp.df[!, exp.output]))

synth_exp = BoundaryExperiment("synth", "data/synth.csv", [:x1, :x2], :class)
iris_exp = BoundaryExperiment("iris", "data/Iris.csv", [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm], :Species)
heart_exp = BoundaryExperiment("heart", "data/heart.csv", [:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang, :oldpeak, :slope,:ca, :thal], :target)
wine_exp = BoundaryExperiment("wine", "data/wine.csv", [:Alcohol,:Malicacid,:Ash,:Acl,:Mg,:Phenols,:Flavanoids,:Nonflavanoidphenols,:Proanth,:Colorint,:Hue,:OD,:Proline], :Wine)
titanic_exp = BoundaryExperiment("titanic", "data/titanic_pp.csv", [:Pclass,:Sex,:Age,:SibSp,:Parch,:Fare], :Survived)
student_exp = BoundaryExperiment("student", "data/student_data_pp.csv", [:school, :sex, :age, :address,:famsize,:Pstatus,:Medu,:Fedu,:traveltime,:studytime,:failures,:schoolsup,:famsup,:paid,:activities,:nursery,:higher,:internet,:romantic,:famrel,:freetime,:goout,:Dalc,:Walc,:health,:absences, :G1, :G2], :G3)
car_exp = BoundaryExperiment("car", "data/car_evaluation_pp.csv", [:buyingprice,:maintenancecost,:doors,:capacity,:luggageboot,:safety], :decision)
adult_exp = BoundaryExperiment("adult", "data/adult_pp.csv", [:age,:fnlwgt,:educationalnum,:gender,:capitalgain,:capitalloss,:hoursperweek], :income)

exps = [ adult_exp, car_exp, student_exp, titanic_exp, wine_exp, synth_exp, iris_exp, heart_exp ]

# for exp in exps
#     td = TrainingData(exp.name, exp.df; inputs=exp.inputs, output=exp.output)
#     modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
#     be = BoundaryExposer(td, modelsut)
    
#     candidates = apply(be; iterations =10, initial_candidates=10, optimizefordiversity=false)
#     df = todataframe(candidates, modelsut; output = exp.output)
#     # CSV.write("data/expresults/$(exp.name)_bcs_random.csv", df)
    
#     # candidates = apply(be; MaxCandidates, iterations=2000, initial_candidates=10)
#     # df = todataframe(candidates, modelsut; output = exp.output)
#     # CSV.write("data/expresults/$(exp.name)_bcs_div.csv", df)
# end

# create latex table head
println("\\begin{tabular}{l|c|c|c}")
println("\\textbf{Dataset} & \\textbf{Features} & \\textbf{Used Features} & \\textbf{Output} \\\\")

for exp in exps
    println("\\textbf{$(exp.name)} & $(inputstotal(exp)) & $(inputsused(exp)) & $(outputalternatives(exp)) \\\\")
end
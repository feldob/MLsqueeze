using CSV, DataFrames

#DataDir = "/Users/feldt/Companies/OpenData/kaggle/compas/propublicaCompassRecividism_data_fairml.csv"
DataDir = read("data/CompassRawDirPath.txt", String)

df = CSV.read(joinpath(DataDir, "propublica_data_for_fairml.csv"), DataFrame)

describe(df)

# Output = Two_yr_Recidivism (binary)
# Continouos variables:
#  - Number_of_Priors (integer in range 0-38)
# Categorical variables:
#  - Misdemeanor (binary), called CrimeDegree in Mothilal2020
#  - Race (categorical with onehot encoding in binary vars: African_American, Asian, Hispanic, Native_American, Other)
#  - Age (binary), indicates if age > 45, skip Age_Below_TwentyFive for now
#  - Sex (binary), Female = 1 or male if 0

dfo = DataFrame(
    PriorsCount = df[:, :Number_of_Priors],
    CrimeDegree = df[:, :Misdemeanor],
    Race_AA = df[:, :African_American],
    Race_A = df[:, :Asian],
    Race_H = df[:, :Hispanic],
    Race_N = df[:, :Native_American],
    Race_O = df[:, :Other],
    Age = df[:, :Age_Above_FourtyFive],
    Sex = df[:, :Female],
    Class = df[:, :Two_yr_Recidivism]
)

if !isdir(joinpath(DataDir, "derived"))
    mkdir(joinpath(DataDir, "derived"))
end

dfo |> CSV.write(joinpath(DataDir, "derived", "compas_mothilal.csv"))
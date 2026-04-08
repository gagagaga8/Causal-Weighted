# eICU external validation data description script
# Generate baseline Table 1 from multiply imputed eICU validation set (val_dat)
#       and plot missing patterns, comparable to validation set description.
# records inwhen DirectoryRun eicu_1_2_preprocessing.R Generate imputed_val_dat_bunfix_hmor.RData

library(dplyr)
library(Hmisc)
library(naniar)

# Loading eICU ExternalValidationMultiple ImputationResults
# File eicu_1_2_preprocessing.R Generate Package mice object val_dat
if (!file.exists("imputed_val_dat_bunfix_hmor.RData")) {
  stop("not to imputed_val_dat_bunfix_hmor.RData Run eicu_1_2_preprocessing.R")
}
load("imputed_val_dat_bunfix_hmor.RData") # toobject val_dat

if (!exists("val_dat")) {
  stop("imputed_val_dat_bunfix_hmor.RData  val_dat object not found in ")
}

# Merge processed Data mice objectInternal data use descriptionstatistics
val_data_des <- val_dat$data %>% select(
  kdigo_creat,
  admission_age,
  gender,
  weight,
  immunosuppressant,
  sofa_24hours,
  creat_k1,
  bun_k1, bun_k2, bun_k3,
  pot_k1, pot_k2, pot_k3,
  ph_k1, ph_k2, ph_k3,
  uo_k1, uo_k2, uo_k3,
  study
)

# Missing Visualization optional 
# gg_miss_upset(val_data_des, nsets = n_var_miss(val_data_des), nintersects = 15)
# naplot(naclus(val_data_des), which = c("na per var"))

# will VariableandSexVariableStandardizeas 0/1 / M/F with MIMIC ResultsComparison
val_data_des$immunosuppressant <- as.character(val_data_des$immunosuppressant)
val_data_des$immunosuppressant[val_data_des$immunosuppressant %in% c("NON", "0")] <- "0"
val_data_des$immunosuppressant[val_data_des$immunosuppressant %in% c("OUI", "1")] <- "1"

val_data_des$gender <- as.character(val_data_des$gender)
val_data_des$gender[val_data_des$gender %in% c("Masculin", "M", "Male")] <- "M"
val_data_des$gender[val_data_des$gender %in% c("Feminin", "F", "Female")] <- "F"

# Generatebystudy Table 1 eICU1 vs eICU2 
vars_for_tab <- colnames(val_data_des)[1:(length(colnames(val_data_des)) - 1)]

table_1_by_study <- tableone::CreateTableOne(
  vars   = vars_for_tab,
  data   = val_data_des,
  strata = c("study"),
  test   = FALSE
)
printed_t_1_by_study <- print(table_1_by_study, catDigits = 1, contDigits = 2, noSpaces = TRUE)

# GenerateMerge processed Overall Table 1 not study ValidationsetFeature 
table_1_overall <- tableone::CreateTableOne(
  vars = vars_for_tab,
  data = val_data_des
)
printed_t_1_overall <- print(
  table_1_overall,
  nonnormal = vars_for_tab,
  catDigits = 1,
  contDigits = 2,
  noSpaces = TRUE
)

cat("eICU ExternalValidationBaselineFeaturedescriptionalreadyGenerate canaccording toneedtowill printed_t_1_by_study / printed_t_1_overall as CSV \n")

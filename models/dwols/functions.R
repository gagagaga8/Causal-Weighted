# will VariableConvertto numericVariableFunction
# if can Convertto numeric thenConvert else 
defactorize <- function(x)
{
  if(class(x)[1]=="factor" & !is.na(as.numeric(as.character(x))[1])) {
      temp <- as.numeric(as.character(x))
  } else { 
      temp <- x
  }
        return(temp)
}


# Rubinrule MergeMultiple ImputationResultsvarianceestimate
# thetas: each ImputationDatasetestimatevalue
# vars: each ImputationDatasetvariance
# ReturnMerge processed varianceestimate Imputation andImputation 
rubinr <-  function(thetas, vars) {
  theta <- mean(thetas) # Merge processed estimate value 
  w <- mean(vars) # Imputation variance within-imputation variance 
  b <- var(thetas) # Imputation variance between-imputation variance 
  return(w+(1+1/length(vars))*b) # Merge processed variance
}

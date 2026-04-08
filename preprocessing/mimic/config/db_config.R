# MIMIC-IV PostgreSQL database connection config
# Do not commit this file to version control

library(RPostgreSQL)

# Create database connection
create_mimic_connection <- function() {
  drv <- dbDriver("PostgreSQL")
  con <- dbConnect(drv,
                   host = Sys.getenv("DB_HOST", "localhost"),
                   port = as.integer(Sys.getenv("DB_PORT", "5432")),
                   dbname = Sys.getenv("MIMIC_DB_NAME", "MIMIC"),
                   user = Sys.getenv("DB_USER", "postgres"),
                   password = Sys.getenv("DB_PASSWORD", ""))
  return(con)
}

# Close database connection
close_mimic_connection <- function(con) {
  dbDisconnect(con)
}

# Safe SQL query execution
query_mimic <- function(con, query) {
  tryCatch({
    result <- dbGetQuery(con, query)
    return(result)
  }, error = function(e) {
    message("Query error: ", e$message)
    return(NULL)
  })
}

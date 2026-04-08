# eICU PostgreSQL database connection config
# Do not commit this file to version control

library(RPostgreSQL)

# Create database connection
create_eicu_connection <- function() {
  drv <- dbDriver("PostgreSQL")
  con <- dbConnect(drv,
                   host = Sys.getenv("DB_HOST", "localhost"),
                   port = as.integer(Sys.getenv("DB_PORT", "5432")),
                   dbname = Sys.getenv("EICU_DB_NAME", "eICU"),
                   user = Sys.getenv("DB_USER", "postgres"),
                   password = Sys.getenv("DB_PASSWORD", ""))
  return(con)
}

# Close database connection
close_eicu_connection <- function(con) {
  dbDisconnect(con)
}

# Safe SQL query execution
query_eicu <- function(con, query) {
  tryCatch({
    result <- dbGetQuery(con, query)
    return(result)
  }, error = function(e) {
    message("Query error: ", e$message)
    return(NULL)
  })
}

library(dplyr)
library(mice)

##################################################################################################################################
##################################################################################################################################
############################################################ eICU 1 ##############################################################
##################################################################################################################################
##################################################################################################################################

eicudbsg <- read.csv("~/Desktop/TEAM METHODS/phd/Trials Data/eicu data/SGeicu.csv", sep=";", comment.char="#")
eicu_dat_1<-data.frame(
        study="eICU1",
        SUBJECT_ID=substring(eicudbsg$subject_id, 5, nchar(eicudbsg$subject_id)-3),
        DATERANDO=as.POSIXct(eicudbsg$inc_daterando, tryFormats = "%d%b%Y:%H:%M:%OS"),
        BRASRANDO=eicudbsg$inc_bras, 
        DATEINC=as.POSIXct(eicudbsg$inc_datecent22, tryFormats = "%d%b%Y:%H:%M:%OS"),
        DATENAISS=as.Date(format(as.Date(eicudbsg$inc_datenaiss, "%d/%m/%Y"), "19%y-%m-%d")), 
        SEXEPAT=eicudbsg$inc_sexepat,
        POIDS=eicudbsg$inc_poids,
        COMTTIMMUNOSUP=eicudbsg$adm_com_ttimmunosup,
        J0_SCOSOFASUM=eicudbsg$rando_car_sofa_sum,
        EER_SEADATE=as.POSIXct(eicudbsg$eer_date, tryFormats = "%d%b%Y:%H:%M:%OS"),
        creat_k1=eicudbsg$rando_car_creat,
        creat_k2=eicudbsg$j0_car_creatmax,
        creat_k3=eicudbsg$j1_car_creatmax,
        bun_k1=eicudbsg$rando_car_uree,
        bun_k2=eicudbsg$j0_car_ureemax,
        bun_k3=eicudbsg$j1_car_ureemax,
        pot_k1=eicudbsg$rando_car_pot,
        pot_k2=eicudbsg$j0_car_potmax,
        pot_k3=eicudbsg$j1_car_potmax,
        ph_k1=eicudbsg$rando_car_ph,
        ph_k2=eicudbsg$j0_car_phmin,
        ph_k3=eicudbsg$j1_car_phmin,
        uo_k1=eicudbsg$j0_car_diures,
        uo_k2=eicudbsg$j1_car_diures,
        uo_k3=eicudbsg$j2_car_diures,
        HOP_SORTIEDATE=as.Date(format(as.Date(eicudbsg$bil_sortiedate, "%d/%m/%Y"), "20%y-%m-%d")),
        hospital_mortality=eicudbsg$bil_sortieetat=="D̩c̩d̩",
        kdigo_creat=eicudbsg$inc_ci4a=="OUI" | is.na(eicudbsg$inc_ci4a)
)

## Create age
eicu_dat_1$AGE <- with(eicu_dat_1, as.numeric(difftime(DATERANDO, DATENAISS , unit="days")))/365.25

###################################################
###### Get survival data as in NEJM Figure 1 ###### 
###################################################

load("eicu1_survival.RData")
temp <- data.frame(SUBJECT_ID=substring(dat$id, 5, nchar(dat$id)-3), t_d60=dat$derniere.nouvelles.censureJ60, status_d60=dat$etat.censureJ60)

eicu_dat_1 <- merge(x = eicu_dat_1, temp, by="SUBJECT_ID")
eicu_dat_1$time_to_death <- NA
eicu_dat_1$time_to_death[temp$status_d60==1] <- temp$t_d60[temp$status_d60==1] 

#######################
###### Create As ###### 
#######################

eicu_dat_1$aki_to_rrt_hours <- as.numeric(with(eicu_dat_1, difftime(EER_SEADATE, DATERANDO, units = "hours")))

eicu_dat_1$a1 <- ifelse(eicu_dat_1$aki_to_rrt_hours < 24, 1, 0)
eicu_dat_1$a2 <- ifelse(eicu_dat_1$aki_to_rrt_hours >= 24 & eicu_dat_1$aki_to_rrt_hours < 48, 1, 0)
eicu_dat_1$a3 <- ifelse(eicu_dat_1$aki_to_rrt_hours >= 48 & eicu_dat_1$aki_to_rrt_hours < 72, 1, 0)

eicu_dat_1$a2 <- pmax(eicu_dat_1$a2, eicu_dat_1$a1)
eicu_dat_1$a3 <- pmax(eicu_dat_1$a3, eicu_dat_1$a2)

eicu_dat_1$a1[is.na(eicu_dat_1$a1)] <- 0
eicu_dat_1$a2[is.na(eicu_dat_1$a2)] <- 0
eicu_dat_1$a3[is.na(eicu_dat_1$a3)] <- 0  

######################################################
############### Create Terminal States ###############
######################################################

eicu_dat_1$PHI_1 <- 0
eicu_dat_1$PHI_2 <- 0
eicu_dat_1$PHI_3 <- 0

eicu_dat_1$PHI_2[!is.na(eicu_dat_1$time_to_death) & eicu_dat_1$time_to_death<1] <- 1
eicu_dat_1$PHI_3[!is.na(eicu_dat_1$time_to_death) & eicu_dat_1$time_to_death<1] <- 1

eicu_dat_1$PHI_3[!is.na(eicu_dat_1$time_to_death) & eicu_dat_1$time_to_death>=1 & eicu_dat_1$time_to_death<2] <- 1


#####################################################
######## Create Hospital Free Days at Day 60 ########
##################################################### 

# Create time_to_hospi_discharge variable
eicu_dat_1$time_to_hospi_discharge <- as.numeric(with(eicu_dat_1, difftime(HOP_SORTIEDATE, DATERANDO, unit="days")))
# fix the patients who appear to have left the hospital before randomization
eicu_dat_1$time_to_hospi_discharge[which(eicu_dat_1$time_to_hospi_discharge<0)] <- 0 

eicu_dat_1$HFD60 <- NA

# The patients who died after hospital discharge AND within 60 days 
# are allocated hospital free days of time_to_death-time_to_hospi_discharge 
# i.e., the time they spent alive and outside the hospital before their death
eicu_dat_1$death_within_60d_outside_h <- with(eicu_dat_1, (!is.na(time_to_hospi_discharge) & !is.na(time_to_death) & time_to_death > time_to_hospi_discharge) & time_to_death <= 60 )
eicu_dat_1$HFD60[eicu_dat_1$death_within_60d_outside_h] <- with(eicu_dat_1[eicu_dat_1$death_within_60d_outside_h,], time_to_death-time_to_hospi_discharge)

# The patients who died after 60 days or had not died at the end of follow up
# are allocated hospital free days of 60-time_to_hospi_discharge 
# i.e., the time they spent alive and outside the hospital before day 60
eicu_dat_1$death_after_d60 <- with(eicu_dat_1, (is.na(time_to_death) | time_to_death > 60))
eicu_dat_1$HFD60[eicu_dat_1$death_after_d60] <- with(eicu_dat_1[eicu_dat_1$death_after_d60,], 60 - time_to_hospi_discharge)

# The patients who left the hospital after 60 days 
# are allocated hospital free days of zero 
eicu_dat_1$HFD60[eicu_dat_1$time_to_hospi_discharge>60] <- 0

# The patients who left the hospital after 60 days (at an unknown date)
# are allocated hospital free days of zero 
eicu_dat_1$HFD60[is.na(eicu_dat_1$HFD60)] <- 0 

# The patients who died in the hospital or 
# had not left the hospital at the end of followup (i.e., had no hospital_mortality is NA)
# are allocated hospital free days of zero 

eicu_dat_1$HFD60[is.na(eicu_dat_1$hospital_mortality) | eicu_dat_1$hospital_mortality] <- 0 

# Convert urine output in ml/kg/h
eicu_dat_1[, c("uo_k1", "uo_k2", "uo_k3")] <- eicu_dat_1[, c("uo_k1", "uo_k2", "uo_k3")]/eicu_dat_1$POIDS/24

# Convert creatinine at baseline in ml/dL
eicu_dat_1$creat_k1 <- eicu_dat_1$creat_k1/88.4

# Give study name
eicu_dat_1$study <- "eICU1"

# Exclude the patients randomized to the "early RRT strategy"
eicu_1_prep <- eicu_dat_1 %>% filter(BRASRANDO != "STRATEGIE PRECOCE") %>%   
        # Select the relevant columns  
        select(SUBJECT_ID, study, kdigo_creat, BRASRANDO, AGE, SEXEPAT, POIDS, COMTTIMMUNOSUP, J0_SCOSOFASUM, 
               creat_k1, creat_k2, creat_k3, bun_k1, bun_k2, bun_k3,
               pot_k1, pot_k2, pot_k3, ph_k1, ph_k2, ph_k3, 
               uo_k1, uo_k2, uo_k3,
               a1, a2, a3, PHI_1, PHI_2, PHI_3, HFD60) %>% 
        # rename these columns
        rename(orig_id=SUBJECT_ID, gender = SEXEPAT, admission_age = AGE, weight=POIDS,
               immunosuppressant=COMTTIMMUNOSUP, hfd = HFD60, SOFA_24hours=J0_SCOSOFASUM) %>% rename_with(tolower)

eicu_1_prep$hmor <- as.numeric((eicu_dat_1 %>% filter(BRASRANDO != "STRATEGIE PRECOCE") %>% select(hospital_mortality))$hospital_mortality)
eicu_1_prep$hmor[is.na(eicu_1_prep$hmor)] <- 0

##################################################################################################################################
##################################################################################################################################
############################################################ eICU 2 ##############################################################
##################################################################################################################################
##################################################################################################################################

INC <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/INC_ANSI.csv",
                sep=";")

ADMISREA <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/ADMISREA_ANSI.csv",
                     sep=";")

DQR <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/DQR_ANSI.csv",
                sep=";")

DQJ <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/DQJ_ANSI.csv",
                sep=";")

EER_SEA <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/EER_ANSI.csv",
                    sep=";")

RDM <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/RDM_ANSI.csv",
                sep=";")

HOP <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/HOP_ANSI.csv",
                sep=";")

BFE <- read.csv("/Users/francois/Desktop/TEAM\ METHODS/phd/Trials\ Data/eicu\ 2\ data/Données\ eICU2/BFE_ANSI.csv",
                sep=";")

no_dateinc <- INC$IN_DATEINC=="" # Exclude 9 patients with no inclusion date ## NB: We will later exclude the 5 patients who withdrew consent (and have missing dat)
eicu_dat <- data.frame(SUBJECT_ID=INC$SUBJECT_ID[!no_dateinc])

eicu_dat$kdigo_creat <- (INC$IN_CI4A==1 | is.na(INC$IN_CI4A) )[!no_dateinc]
eicu_dat$DATERANDO <- RDM$RDM_DATERANDO[!no_dateinc]

eicu_dat$BRASRANDO <- RDM$RDM_BRASRANDO[!no_dateinc]
eicu_dat$BRASRANDO[eicu_dat$BRASRANDO==1 & !is.na(eicu_dat$BRASRANDO)] <- "standard"
eicu_dat$BRASRANDO[eicu_dat$BRASRANDO==2 & !is.na(eicu_dat$BRASRANDO)] <- "grande_attente"

eicu_dat$DATEINC <- as.POSIXct(INC$IN_DATEINC[!no_dateinc], tryFormats = "%d%b%Y:%H:%M:%OS") 
eicu_dat$DATENAISS <- as.POSIXct(with(INC[!no_dateinc,], paste0(IN_DATENAISS_Y, "-", IN_DATENAISS_M, "-01")), tryFormats = "%Y-%m-%d")
eicu_dat$AGE <-with(eicu_dat, as.numeric(difftime(DATEINC, DATENAISS , unit="days")))/365.25
eicu_dat$SEXEPAT <- ifelse(INC$IN_SEXEPAT[!no_dateinc]==1, "M", "F")
eicu_dat$POIDS <- ADMISREA$ADM_POIDS_V[!no_dateinc]
eicu_dat$COMTTIMMUNOSUP <- ADMISREA$ADM_COMTTIMMUNOSUP[!no_dateinc]

eicu_dat$J0_SCOSOFASUM <- DQJ$J0_SCOSOFASUM[!no_dateinc]

eicu_dat$J0_CCBCREATMAX <- DQJ$J0_CCBCREATMAX[!no_dateinc]
eicu_dat$J0_CCBUREEMAX <- DQJ$J0_CCBUREEMAX[!no_dateinc]
eicu_dat$J0_CCBPOTMAX <- DQJ$J0_CCBPOTMAX[!no_dateinc]
eicu_dat$J0_CCBPHMIN <- DQJ$J0_CCBPHMIN[!no_dateinc]
eicu_dat$J0_CCBDIURES <- DQJ$J0_CCBDIURES_V[!no_dateinc]

eicu_dat$J1_CCBCREATMAX <- DQJ$J1_CCBCREATMAX[!no_dateinc]
eicu_dat$J1_CCBUREEMAX <- DQJ$J1_CCBUREEMAX[!no_dateinc]
eicu_dat$J1_CCBPOTMAX <- DQJ$J1_CCBPOTMAX[!no_dateinc]
eicu_dat$J1_CCBPHMIN <- DQJ$J1_CCBPHMIN[!no_dateinc]
eicu_dat$J1_CCBDIURES <- DQJ$J1_CCBDIURES_V[!no_dateinc]

eicu_dat$J2_CCBCREATMAX <- DQJ$J2_CCBCREATMAX[!no_dateinc]
eicu_dat$J2_CCBUREEMAX <- DQJ$J2_CCBUREEMAX[!no_dateinc]
eicu_dat$J2_CCBPOTMAX <- DQJ$J2_CCBPOTMAX[!no_dateinc]
eicu_dat$J2_CCBPHMIN <- DQJ$J2_CCBPHMIN[!no_dateinc]
eicu_dat$J2_CCBDIURES <- DQJ$J2_CCBDIURES_V[!no_dateinc]

#####

eicu_dat$R0_SCOSOFASUM <- DQR$R0_SCOSOFASUM[!no_dateinc]
eicu_dat$R0_CCBCREATMAX <- DQR$R0_CCBCREATMAX[!no_dateinc]
eicu_dat$R0_CCBUREEMAX <- DQR$R0_CCBUREEMAX[!no_dateinc]
eicu_dat$R0_CCBPOTMAX <- DQR$R0_CCBPOTMAX[!no_dateinc]
eicu_dat$R0_CCBPHMIN <- DQR$R0_CCBPHMIN[!no_dateinc]
eicu_dat$R0_CCBDIURES <- DQR$R0_CCBDIURES_V[!no_dateinc]

eicu_dat$R1_CCBCREATMAX <- DQR$R1_CCBCREATMAX[!no_dateinc]
eicu_dat$R1_CCBUREEMAX <- DQR$R1_CCBUREEMAX[!no_dateinc]
eicu_dat$R1_CCBPOTMAX <- DQR$R1_CCBPOTMAX[!no_dateinc]
eicu_dat$R1_CCBPHMIN <- DQR$R1_CCBPHMIN[!no_dateinc]
eicu_dat$R1_CCBDIURES <- DQR$R1_CCBDIURES_V[!no_dateinc]

eicu_dat$R2_CCBCREATMAX <- DQR$R2_CCBCREATMAX[!no_dateinc]
eicu_dat$R2_CCBUREEMAX <- DQR$R2_CCBUREEMAX[!no_dateinc]
eicu_dat$R2_CCBPOTMAX <- DQR$R2_CCBPOTMAX[!no_dateinc]
eicu_dat$R2_CCBPHMIN <- DQR$R2_CCBPHMIN[!no_dateinc]
eicu_dat$R2_CCBDIURES <- DQR$R2_CCBDIURES_V[!no_dateinc]

#####

#####

eicu_dat$DATESORTIE <- as.Date(HOP$HOP_DATESORTIE[!no_dateinc], format = "%d/%m/%Y")
eicu_dat$HOP_SV <- HOP$HOP_SV[!no_dateinc]
eicu_dat$hospital_mortality <- NA
eicu_dat$hospital_mortality[which(eicu_dat$HOP_SV==2)] <- TRUE
eicu_dat$hospital_mortality[which(eicu_dat$HOP_SV==1)] <- FALSE

eicu_dat$EER_SEADATE <- EER_SEA$EER_SEADATE[!no_dateinc]
eicu_dat$EER_SEADATE[eicu_dat$EER_SEADATE==""] <- NA
eicu_dat$EER_SEADATE <- as.POSIXct(eicu_dat$EER_SEADATE, tryFormats = "%d%b%Y:%H:%M:%OS") 

#######################
###### Create As ###### 
#######################
eicu_dat$aki_to_rrt_hours <- NA
eicu_dat$aki_to_rrt_hours <- as.numeric(difftime(eicu_dat$EER_SEADATE, eicu_dat$DATEINC, unit="hours"))

eicu_dat$a1 <- ifelse(eicu_dat$aki_to_rrt_hours < 24, 1, 0)
eicu_dat$a2 <- ifelse(eicu_dat$aki_to_rrt_hours >= 24 & eicu_dat$aki_to_rrt_hours < 48, 1, 0)
eicu_dat$a3 <- ifelse(eicu_dat$aki_to_rrt_hours >= 48 & eicu_dat$aki_to_rrt_hours < 72, 1, 0)

eicu_dat$a2 <- pmax(eicu_dat$a2, eicu_dat$a1)
eicu_dat$a3 <- pmax(eicu_dat$a3, eicu_dat$a2)

eicu_dat$a1[is.na(eicu_dat$a1)] <- 0
eicu_dat$a2[is.na(eicu_dat$a2)] <- 0
eicu_dat$a3[is.na(eicu_dat$a3)] <- 0

### Get time from inclusion to randomization
eicu_dat$aki_to_rando_hours <- NA
eicu_dat$aki_to_rando_hours[eicu_dat$DATERANDO!=""] <- as.numeric(difftime(as.POSIXct(eicu_dat$DATERANDO[eicu_dat$DATERANDO!=""], tryFormats = "%d%b%Y:%H:%M:%OS"), eicu_dat$DATEINC[eicu_dat$DATERANDO!=""], unit="hours"))

eicu_dat$aki_to_rando_hours <- pmax(eicu_dat$aki_to_rando_hours, 0) # This fixes the one patient who appear to have been randomized 4 hours prior to ... inclusion

###########################
###### Formating k's ######
########################### 

# See document "xxx.png" attached 

condition_0_24 <- !is.na(eicu_dat$aki_to_rando_hours) & eicu_dat$aki_to_rando_hours < 24

eicu_dat$creat_k1 <- as.numeric(eicu_dat$J0_CCBCREATMAX)
eicu_dat$creat_k2 <- as.numeric(ifelse(condition_0_24, eicu_dat$R0_CCBCREATMAX, eicu_dat$J1_CCBCREATMAX))
eicu_dat$creat_k3 <- as.numeric(ifelse(condition_0_24, eicu_dat$R1_CCBCREATMAX, eicu_dat$J2_CCBCREATMAX))

eicu_dat$bun_k1 <- as.numeric(eicu_dat$J0_CCBUREEMAX)
eicu_dat$bun_k2 <- as.numeric(ifelse(condition_0_24, eicu_dat$R0_CCBUREEMAX, eicu_dat$J1_CCBUREEMAX))
eicu_dat$bun_k3 <- as.numeric(ifelse(condition_0_24, eicu_dat$R1_CCBUREEMAX, eicu_dat$J2_CCBUREEMAX))

eicu_dat$pot_k1 <- as.numeric(eicu_dat$J0_CCBPOTMAX)
eicu_dat$pot_k2 <- as.numeric(ifelse(condition_0_24, eicu_dat$R0_CCBPOTMAX, eicu_dat$J1_CCBPOTMAX))
eicu_dat$pot_k3 <- as.numeric(ifelse(condition_0_24, eicu_dat$R1_CCBPOTMAX, eicu_dat$J2_CCBPOTMAX))

eicu_dat$ph_k1 <- as.numeric(eicu_dat$J0_CCBPHMIN)
eicu_dat$ph_k2 <- as.numeric(ifelse(condition_0_24, eicu_dat$R0_CCBPHMIN, eicu_dat$J1_CCBPHMIN))
eicu_dat$ph_k3 <- as.numeric(ifelse(condition_0_24, eicu_dat$R1_CCBPHMIN, eicu_dat$J2_CCBPHMIN))

eicu_dat$uo_k1 <- as.numeric(eicu_dat$J0_CCBDIURES)
eicu_dat$uo_k2 <- as.numeric(ifelse(condition_0_24, eicu_dat$R0_CCBDIURES, eicu_dat$J1_CCBDIURES))
eicu_dat$uo_k3 <- as.numeric(ifelse(condition_0_24, eicu_dat$R1_CCBDIURES, eicu_dat$J2_CCBDIURES))

####

condition_24_48 <- !is.na(eicu_dat$aki_to_rando_hours) & eicu_dat$aki_to_rando_hours >= 24 & eicu_dat$aki_to_rando_hours < 48

eicu_dat$creat_k3[condition_24_48] <- as.numeric(eicu_dat$R0_CCBCREATMAX)[condition_24_48] 
eicu_dat$bun_k3[condition_24_48] <- as.numeric(eicu_dat$R0_CCBUREEMAX)[condition_24_48] 
eicu_dat$pot_k3[condition_24_48] <- as.numeric(eicu_dat$R0_CCBPOTMAX)[condition_24_48] 
eicu_dat$ph_k3[condition_24_48] <- as.numeric(eicu_dat$R0_CCBPHMIN)[condition_24_48] 
eicu_dat$uo_k3[condition_24_48] <- as.numeric(eicu_dat$R0_CCBDIURES)[condition_24_48] 


#####################################################
######## Create Hospital Free Days at Day 60 ########
##################################################### 

suivi_J60 <- as.numeric(BFE$BFE_SUIVI1)[!no_dateinc]
suivi_R60 <- as.numeric(BFE$BFE_SUIVI2)[!no_dateinc]
suivi_60 <- ifelse(is.na(suivi_J60), suivi_R60, suivi_J60)

date_fin <- BFE$BFE_DATEFIN[!no_dateinc]

temp <- data.frame(SUBJECT_ID=BFE$SUBJECT_ID[!no_dateinc], suivi_60=suivi_60, motif_arret=BFE$BFE_MOTIFARRET[!no_dateinc], date_inc=eicu_dat$DATEINC, date_sortie_hospi=eicu_dat$DATESORTIE,  hospital_mortality=eicu_dat$hospital_mortality, date_fin=date_fin)

temp$date_of_death <- NA
temp$date_of_death[which(temp$motif_arret==1)] <- temp$date_fin[which(temp$motif_arret==1)]
temp$date_of_death <- as.POSIXct(temp$date_of_death, format = "%d/%m/%Y") 

eicu_dat <- eicu_dat[which(!temp$motif_arret %in% c(3,4)),] # Exclude the 5 patients who withdrew consent (and have missing data)
temp <- temp[which(!temp$motif_arret %in% c(3,4)),]

# Create time_to death and time_to_hospi_discharge variables
temp$time_to_death <- as.numeric(with(temp, difftime(date_of_death, date_inc, unit="days")))
temp$time_to_hospi_discharge <- as.numeric(with(temp, difftime(date_sortie_hospi, date_inc, unit="days")))

temp$HFD60 <- NA

# The patients who died in the hospital or 
# had not left the hospital at the end of followup (i.e., had no hospital_mortality is NA)
# are allocated hospital free days of zero 

temp$HFD60[is.na(temp$hospital_mortality) | temp$hospital_mortality] <- 0 

# The patients who died after hospital discharge AND within 60 days 
# are allocated hospital free days of time_to_death-time_to_hospi_discharge 
# i.e., the time they spent alive and outside the hospital before their death
temp$death_within_60d_outside_h <- with(temp, (!is.na(time_to_hospi_discharge) & !is.na(time_to_death) & time_to_death > time_to_hospi_discharge) & time_to_death <= 60 )
temp$HFD60[temp$death_within_60d_outside_h] <- with(temp[temp$death_within_60d_outside_h,], time_to_death-time_to_hospi_discharge)

# The patients who died after 60 days or had not died at the end of follow up
# are allocated hospital free days of 60-time_to_hospi_discharge 
# i.e., the time they spent alive and outside the hospital before day 60
temp$death_after_d60 <- with(temp, (is.na(time_to_death) | time_to_death > 60))
temp$HFD60[temp$death_after_d60] <- with(temp[temp$death_after_d60,], 60 - time_to_hospi_discharge)

# The patients who left the hospital after 60 days 
# are allocated hospital free days of zero 
temp$HFD60[temp$time_to_hospi_discharge>60] <- 0

# The patients who left the hospital after 60 days 
# are allocated hospital free days of zero 
temp$HFD60[is.na(temp$HFD60)] <- 0 

# The patient who left the hospital alive at day 0
# is allocated hospital free days of 60 
temp$HFD60[temp$HFD60>60] <- 60 

# Get hospital mortality
eicu_dat$hmor <- as.numeric(is.na(eicu_dat$HOP_SV) | eicu_dat$HOP_SV==2)

### Merge temp and eicu_dat by SUBJECT_ID
eicu_dat <- merge(x = eicu_dat, y=temp[, c("SUBJECT_ID", "time_to_hospi_discharge", "time_to_death", "HFD60")], by="SUBJECT_ID")

######################################################
############### Create Terminal States ###############
######################################################

eicu_dat$PHI_1 <- 0
eicu_dat$PHI_2 <- 0
eicu_dat$PHI_3 <- 0

eicu_dat$PHI_2[!is.na(eicu_dat$time_to_death) & eicu_dat$time_to_death<1] <- 1
eicu_dat$PHI_3[!is.na(eicu_dat$time_to_death) & eicu_dat$time_to_death<1] <- 1

eicu_dat$PHI_3[!is.na(eicu_dat$time_to_death) & eicu_dat$time_to_death>=1 & eicu_dat$time_to_death<2] <- 1

# Convert urine output in ml/kg/h
eicu_dat[, c("uo_k1", "uo_k2", "uo_k3")] <- eicu_dat[, c("uo_k1", "uo_k2", "uo_k3")]/eicu_dat$POIDS/24

# Convert creatinine at baseline in ml/dL
eicu_dat$creat_k1 <- eicu_dat$creat_k1/88.4

# Give study name
eicu_dat$study <- "eICU2"

##########
##########

# Exclude the patients randomized to the "more delayed RRT strategy"
eicu_2_prep <- eicu_dat %>% filter(BRASRANDO != "grande_attente"| is.na(BRASRANDO)) %>%   
        # Select the relevant columns  
        select(orig_id=SUBJECT_ID, study, kdigo_creat, BRASRANDO, AGE, SEXEPAT, POIDS, COMTTIMMUNOSUP, J0_SCOSOFASUM, 
               creat_k1, creat_k2, creat_k3, bun_k1, bun_k2, bun_k3,
               pot_k1, pot_k2, pot_k3, ph_k1, ph_k2, ph_k3, 
               uo_k1, uo_k2, uo_k3,
               a1, a2, a3, PHI_1, PHI_2, PHI_3, HFD60, hmor) %>%
        # rename these columns
        rename(gender = SEXEPAT, admission_age = AGE, weight=POIDS,
               immunosuppressant=COMTTIMMUNOSUP, hfd = HFD60, SOFA_24hours=J0_SCOSOFASUM) %>% rename_with(tolower)

# Clone the patients randomized to the "standard RRT strategy"
eicu_2_prep <- rbind(eicu_2_prep, eicu_2_prep %>% filter(brasrando == "standard"))


######################################################
############ Merge & Impute the two bases ############
######################################################

val_dat_miss <- rbind(eicu_1_prep, eicu_2_prep)

# Give the each of these new patients an id
val_dat_miss <- val_dat_miss %>% mutate(subject_id=1:nrow(val_dat_miss))

val_dat_miss$study <- as.factor(val_dat_miss$study)
val_dat_miss$brasrando <- as.factor(val_dat_miss$brasrando)
val_dat_miss$immunosuppressant <- as.factor(val_dat_miss$immunosuppressant)

val_dat_miss$a1 <- as.factor(val_dat_miss$a1)
val_dat_miss$a2 <- as.factor(val_dat_miss$a2)
val_dat_miss$a3 <- as.factor(val_dat_miss$a3)

val_dat_miss$phi_1 <- as.factor(val_dat_miss$phi_1)
val_dat_miss$phi_2 <- as.factor(val_dat_miss$phi_2)
val_dat_miss$phi_3 <- as.factor(val_dat_miss$phi_3)

### convert BUN from mmol/L to mg/dL
val_dat_miss$bun_k1 <- val_dat_miss$bun_k1 / 0.3571
val_dat_miss$bun_k2 <- val_dat_miss$bun_k2 / 0.3571
val_dat_miss$bun_k3 <- val_dat_miss$bun_k3 / 0.3571
###

imp <- mice(val_dat_miss, maxit=0)
predM <- imp$predictorMatrix
meth <- imp$method

no_predictor_var <- c("brasrando")

predM[, no_predictor_var] <- 0 # do not use no_predictor_var as predictors of other missing variables

do_not_impute_var <- c("brasrando")

meth[do_not_impute_var] <- "" # do not impute the do_not_impute_var

val_dat <- mice(val_dat_miss, m=20, seed = 1, predictorMatrix = predM, method = meth) # Change No. of imputations here
val_dat_simp <- complete(val_dat, 1)

save(val_dat, file="imputed_val_dat_bunfix_hmor.RData")




#### Get hospital mortality in the validation set
hm_ak1 <- eicu_dat_1 %>% filter(BRASRANDO != "STRATEGIE PRECOCE") %>% select(hospital_mortality)
eicu1_deaths <- table(hm_ak1)["TRUE"]

hm_ak2 <- eicu_dat %>% filter(BRASRANDO != "grande_attente"| is.na(BRASRANDO)) 
hm_ak2 <- rbind(hm_ak2, eicu_dat[eicu_dat$BRASRANDO == "standard" & !is.na(eicu_dat$BRASRANDO), ])
eicu2_deaths <- sum(table(hm_ak2$HOP_SV, useNA = "ifany")[c(2, 3)])

eicu1_deaths + eicu2_deaths
nrow(hm_ak1) + nrow(hm_ak2)
(eicu1_deaths + eicu2_deaths) / ( nrow(hm_ak1) + nrow(hm_ak2) )
####





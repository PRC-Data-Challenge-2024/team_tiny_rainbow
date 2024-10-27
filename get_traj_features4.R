rm(list=ls())

library(tidyverse)
library(arrow)
library(geosphere)
library(airportr)
library(data.table)
library(REdaS)

# load flight data -----------------------------------------------------------------

submission_set <- read_csv("final_submission_set.csv")
challenge_set <- read_csv("challenge_set.csv")

dataset <- rbind(submission_set, challenge_set)

# airport data ----------------------------------------------------------------

airports_dep <- airportr::airports%>%
  rename(adep = ICAO,
         adep_lon = Longitude,
         adep_lat = Latitude,
         adep_altitude = Altitude)%>%
  select(adep, adep_lon,adep_lat,adep_altitude)

airports_arr <- airportr::airports%>%
  rename(ades = ICAO,
         ades_lon = Longitude,
         ades_lat = Latitude,
         ades_altitude = Altitude)%>%
  select(ades, ades_lon,ades_lat,ades_altitude)

# load trajectory data -------------------------------------------------------

processed_files <- list()

files <- list.files("Challenge Data", full.names = T)
files <- as.data.frame(files)

files$date <- list.files("Challenge Data")
files$date <- substr(files$date, 1,10)

files <- files %>% arrange(date)

for (i in 1:nrow(files)) {
    
  try({
  print(files[i,"files"])
 
    traj <- read_parquet(files[i,"files"])
    flight_ids <- c(unique(traj$flight_id))
  
  if (i > 1) {
    traj0 <- read_parquet(files[i-1,"files"])
    traj0 <- traj0[traj0$flight_id %in% flight_ids, ]
    
    traj <- rbind(traj, traj0)
  }  
    
  if (i < nrow(files)) {
    trajx <- read_parquet(files[i+1,"files"])
    trajx <- trajx[trajx$flight_id %in% flight_ids, ]
      
    traj <- rbind(traj, trajx)
  }  
  

#  --------------------------------------------------------------------------

traj1 <- traj%>%
  group_by(flight_id)%>%
  arrange(timestamp, .by_group = T)%>%
  mutate(vs = ((dplyr::lead(altitude,n = 10)-altitude)/as.numeric((dplyr::lead(timestamp, n = 10)-timestamp)))*60)%>%    # vertical speed 
  mutate(vertical_rate = ifelse(is.na(vertical_rate),vs,vertical_rate))%>%
  mutate(flight_id = as.numeric(flight_id))%>%
  left_join(dataset, by = "flight_id")%>%
  left_join(airports_dep, by = "adep")%>%
  left_join(airports_arr, by = "ades")%>%
    group_by(flight_id)%>%
    arrange(timestamp, .by_group = T)%>%
    mutate(ades_lon = ifelse(is.na(ades_lon),dplyr::last(longitude),ades_lon))%>%
    mutate(ades_lat = ifelse(is.na(ades_lat),dplyr::last(latitude),ades_lat))%>%
    mutate(adep_lon = ifelse(is.na(adep_lon),dplyr::first(longitude),adep_lon))%>%
    mutate(adep_lat = ifelse(is.na(adep_lat),dplyr::first(latitude),adep_lat))
    
# distance from adep e ades -------------------------------------------------------

traj1 <- traj1 %>%
  mutate(dist_adep = 0.000539957*distHaversine(cbind(longitude, latitude), cbind(adep_lon, adep_lat)))%>%
  mutate(dist_ades = 0.000539957*distHaversine(cbind(longitude, latitude), cbind(ades_lon, ades_lat)))%>%
  mutate(dist_airp = 0.000539957*distHaversine(cbind(adep_lon, adep_lat), cbind(ades_lon, ades_lat)))

# phases --------------
traj1 <- traj1 %>%
  ungroup()%>%
  mutate(phase = ifelse(dist_adep <= 1, "GND_DEP",0))%>%
  mutate(phase = ifelse(dist_adep > 1 & dist_adep <= 40, "DEP_40",phase))%>%
  mutate(phase = ifelse(dist_adep > 40 & dist_adep <= 100, "DEP_100",phase))%>%
  mutate(phase = ifelse(dist_adep > 100 & dist_ades > 100, "ENR",phase))%>%
  mutate(phase = ifelse(dist_ades <= 100, "ARR_100",phase))%>%
  mutate(phase = ifelse(dist_ades <= 1, "GND_ARR",phase))
      
change_phase <- function(comp4,text1,text2){
      comp4 <- comp4 %>% group_by(flight_id) %>% mutate(change = ifelse(lag(phase)==text2 & phase == text1,1,0)) %>% ungroup()
      comp4 <- comp4 %>% group_by(flight_id) %>% mutate(change = cumsum(replace_na(change, 0))) %>% ungroup()
      comp4 <- comp4 %>% mutate(phase = ifelse(phase == text1 & change != 0, text2, phase))
      comp4$change <- NULL
      return(comp4)
    }
comp4 <- traj1   
    comp4 <- change_phase(comp4,"GND_DEP","DEP_40")
    comp4 <- change_phase(comp4,"DEP_40","DEP_100")
    comp4 <- change_phase(comp4,"DEP_100","ENR")
    comp4 <- change_phase(comp4,"ENR","ARR_100")
    comp4 <- change_phase(comp4,"ARR_100","ARR_40")
    comp4 <- change_phase(comp4,"ARR_40","GND_ARR")
    comp4 <- comp4 %>% mutate(phase = ifelse(dist_airp <= 140 & (phase %in% c("DEP_100","ARR_100")),c("ENR"),phase))
    comp4 <- comp4 %>% mutate(phase = ifelse(dist_airp > 140 & dist_airp <= 200 & (phase %in% c("DEP_100")),c("ENR"),phase))
    comp4 <- comp4 %>% mutate(phase = ifelse(dist_airp >= 140 & dist_ades <= 100 & dist_ades > 40,c("ARR_100"),phase))
    
    # comp5 <- comp4 %>% group_by(flight_id) %>% filter(any(phase == c("ENR"))) %>% ungroup()
    comp5 <- complete(comp4, flight_id, phase, fill = list(lat = 0))
    comp5$phase <- as.factor(comp5$phase)
    rm(comp4)
    
    ordem <- c("GND_DEP","DEP_40","DEP_100","ENR","ARR_100","ARR_40","GND_ARR")
    ordem2 <- c("GND_DEP","DEP_40","ENR","ARR_40","GND_ARR")
    ordem3 <- c("GND_DEP","DEP_40","ENR","ARR_100","ARR_40","GND_ARR")
    
    comp5 <- comp5 %>% arrange(flight_id,timestamp, match(phase, ordem), desc(phase), desc(phase))
    
#comp5 <- traj1   
traj1 <- comp5   
rm(comp5)
#traj1x <- readjust_phase(traj1)
#summary(as.factor(comp5$phase))

# features -----------------------------------------------------------------------

traj2 <- traj1 %>%
  mutate(phase = as.character(phase))%>%
  subset(phase %in% c("DEP_40","DEP_100","ENR","ARR_100","ARR_40"))%>%
  mutate(phase = ifelse(phase == "DEP_100", "ENR",phase))%>%
  mutate(phase = ifelse(phase == "ARR_40", "ARR_100",phase))%>%
  group_by(flight_id, phase)%>%
  arrange(timestamp, .by_group = T)%>%
  mutate(flown_distance = distHaversine(cbind(longitude, latitude), cbind(dplyr::lag(longitude),dplyr::lag(latitude))))%>%
  
  mutate(track = deg2rad(track))%>%
  mutate(tailwind_speed = sin(track)*u_component_of_wind + cos(track)*v_component_of_wind)%>%
  mutate(timelag = as.numeric(timestamp)- as.numeric(dplyr::lag(timestamp)))%>%
  mutate(wind_distance = tailwind_speed*timelag)%>%
  
  mutate(airspeed = (groundspeed*0.514444) - tailwind_speed)%>%
  mutate(specific_energy = (airspeed^2 + 9.8*(altitude*0.3048)))%>%
  mutate(max_track = max(track, na.rm = T))%>%
  mutate(min_track = min(track, na.rm = T))%>%
  mutate(track_variation = max_track - (min_track))%>%
  mutate(track_variation2 = max_track - (min_track+360))%>%
  mutate(track_variation = min(abs(track_variation),abs(track_variation2)))%>%
  summarise(track_variation = track_variation[1],
            average_vertical_rate = mean(vertical_rate, na.rm = T, trim = 0.05),
            average_airspeed = mean(airspeed, na.rm = T, trim = 0.05),
            groundspeed = mean(groundspeed*0.514444, na.rm = T, trim = 0.05),
            wind_distance = sum(wind_distance, na.rm = T),
            average_temperature = mean(temperature, na.rm = T, trim = 0.05),
            average_humidity = mean(specific_humidity, na.rm = T, trim = 0.05),
            specific_energy = dplyr::last(specific_energy),
            flown_distance = sum(flown_distance, na.rm = T),
            average_altitude = mean(altitude, na.rm = T, trim = 0.05),
            max_altitude = max(altitude, na.rm = T),
            cruise_altitude = quantile(altitude,0.99, na.rm = T))%>%
  pivot_wider(id_cols = "flight_id", names_from = "phase", values_from = 3:14)%>%
  select(-cruise_altitude_DEP_40,-cruise_altitude_ARR_100 )
  

# energy 10 NM -----------------------------
energy_10 <-  traj1 %>%
  group_by(flight_id)%>%
  subset(phase == "DEP_40")%>%
  subset(vertical_rate > 0) %>%
  arrange(timestamp, .by_group = T)%>%
  mutate(first_dist = dplyr::first(dist_adep))%>%
  subset(first_dist < 2)%>%
  mutate(flown_distance = distHaversine(cbind(longitude, latitude), cbind(dplyr::lag(longitude),dplyr::lag(latitude))))%>%
  mutate(flown_distance = ifelse(is.na(flown_distance),0,flown_distance))%>%
  mutate(flown_distance = cumsum(flown_distance))%>%
  mutate(flown_distance = flown_distance*0.000539957) %>%
  subset(flown_distance <= 10)%>%
  mutate(track = deg2rad(track))%>%
  mutate(tailwind_speed = sin(track)*u_component_of_wind + cos(track)*v_component_of_wind)%>%
  mutate(timelag = as.numeric(timestamp)- as.numeric(dplyr::lag(timestamp)))%>%
  mutate(wind_distance = tailwind_speed*timelag)%>%
  mutate(airspeed = (groundspeed*0.514444) - tailwind_speed)%>%
  mutate(specific_energy = (airspeed^2 + 9.8*(altitude*0.3048)))%>%
  arrange(flown_distance, .by_group = T)%>%
  summarise(specific_energy = dplyr::last(specific_energy),
            tas_10NM = dplyr::last(airspeed),
            groundspeed_10NM = dplyr::last(groundspeed))

# velocity of lift off -------

vlof <- traj1%>%
  subset(dist_adep < 2)%>%
  mutate(adep_height = altitude - adep_altitude)%>%
  subset(adep_height > 0)%>%
  group_by(flight_id) %>% arrange(timestamp, .by_group = T)%>%
  mutate(track = deg2rad(track))%>%
  mutate(tailwind_speed = sin(track)*u_component_of_wind + cos(track)*v_component_of_wind)%>%
  mutate(timelag = as.numeric(timestamp)- as.numeric(dplyr::lag(timestamp)))%>%
  mutate(wind_distance = tailwind_speed*timelag)%>%
  mutate(airspeed = (groundspeed*0.514444) - tailwind_speed)%>%
  summarise(first_adep_height = dplyr::first(adep_height),
            vlof_tas = dplyr::first(airspeed),
            sqrd_vlof_tas = dplyr::first(airspeed)^2,
            vlof_groundspeed = dplyr::first(groundspeed))


# Alligier features -------------------------------------------------

alligier <- traj1 %>%
  subset(dist_adep < 2)%>%
  mutate(adep_height = altitude - adep_altitude)%>%
  subset(adep_height > 0)%>%
  group_by(flight_id) %>% arrange(timestamp, .by_group = T)%>%
  mutate(n = row_number())%>%
  subset(n <= 10)%>%
  mutate(track = deg2rad(track))%>%
  mutate(tailwind_speed = sin(track)*u_component_of_wind + cos(track)*v_component_of_wind)%>%
  mutate(timelag = as.numeric(timestamp)- as.numeric(dplyr::lag(timestamp)))%>%
  mutate(wind_distance = tailwind_speed*timelag)%>%
  mutate(tas = (groundspeed*0.514444) - tailwind_speed)%>%
  mutate(specific_energy = (tas^2 + 9.8*(altitude*0.3048)))%>%
  mutate(sqrd_tas = tas^2)%>%
  select(flight_id,tas,sqrd_tas, temperature, adep_height, altitude,specific_energy, vertical_rate,n)%>%
  pivot_wider(names_from = "n",id_cols = "flight_id", values_from = c(2:7))
  
  
#summary(allig$n)
#--------------------------------------------------------------------

#library(plotly)
#fig <- plot_ly(x = x$adep_height, type = "histogram", cumulative = list(enabled = TRUE))
#fig

# kpi17 -----------------------------------------------------------------------

 #traj1 <- traj1
  radiusx <- 200
  vs_limitx <- 300     # 5 ft/s == 500 ft/min
  level_bandx <- 300
  min_level_timex <- 20
  boxx <- 0.99
  min_altx <- 1000
  max_timex <- 0



traj1 <- traj1%>%
  mutate(alt = altitude,
         lat = latitude,
         lon = longitude,
         time = timestamp)%>%
  mutate(alt2 = round(alt, digits = -2))

# exclusion box --------------------------------------------------------------------------------------


exclusion_box1 <- traj1%>%
  subset(dist_adep < radiusx)%>%                   # raio de análise
  mutate(adep_height = altitude - adep_altitude)%>%
  subset(adep_height > 0)%>%
  group_by(flight_id) %>%
  mutate(timex = time)%>%
  mutate(time = as.numeric(time))%>%
  dplyr::arrange(time, .by_group = TRUE)%>% 
  mutate(altmax =  max(alt2, na.rm = T))%>%
  mutate(rn = row_number())

exclusion_box1 <- exclusion_box1%>%
  mutate(altlimit1 = max(alt2, na.rm = T))%>%    
  mutate(altlimit2 = altlimit1*boxx)%>%                            # box
  mutate(position = which.max(alt2))%>%
  slice(which((row_number() < position)))%>%
  subset(alt > altlimit2)

if (nrow(exclusion_box1)>0){
  flight_ids <- exclusion_box1%>%
    mutate(leveltime = ifelse(abs(vertical_rate) < vs_limitx, lead(time)-time,0))%>%       # vertical speed limit
    #mutate(leveltime = ifelse(abs(lead(alt)-alt) < level_bandx, lead(time)-time,0))%>%   # vertical speed limit  # band
    mutate(leveltime = ifelse(is.na(leveltime),0,leveltime))%>%                                                 
    mutate(leveldistance = ifelse(leveltime > 0, distHaversine(cbind(lon, lat), cbind(lead(lon),lead(lat))),0))%>% #level distance
    mutate(leveldistance = ifelse(is.na(leveldistance),0, leveldistance))%>%
    mutate(flown_distance = distHaversine(cbind(lon, lat), cbind(lead(lon),lead(lat))))%>%                         # flown distance
    #filter(any(leveltime > 0))%>%
    mutate(tleveltime = sum(leveltime, na.rm = T))%>%
    mutate(cruise_flag = ifelse(tleveltime >  max_timex, 1, 0))%>%                                   # exclusion box 1
    subset(cruise_flag > 0)%>%
    group_by(flight_id)%>%
    dplyr::summarise(altlimit2 = altlimit2[1])%>%
    mutate(box_flag = 1)
}

#-- KPI 17 ----------------------------------------------------------------------------

vertical_eff <- traj1%>%
  subset(dist_adep < radiusx)%>%                   # raio de análise
  mutate(adep_height = altitude - adep_altitude)%>%
  subset(adep_height > 0)%>%
                       # altura minima de analise
  
  group_by(flight_id) %>%
  mutate(timex = time)%>%
  mutate(time = as.numeric(time))%>%
  dplyr::arrange(time, .by_group = TRUE)%>% 
  
  left_join(flight_ids, by = "flight_id") %>%
  mutate(maxalt = max(alt2, na.rm = T))%>%
  mutate(box_flag = ifelse(is.na(box_flag),0,box_flag))%>%
  mutate(altlimit = ifelse(box_flag == 1, altlimit2, maxalt))%>%
  mutate(position = which.max(alt2))%>%
  slice(which((row_number() < position)))%>%
  subset(alt < altlimit)%>%
  
  mutate(leveltime = ifelse(abs(vertical_rate) < vs_limitx, lead(time)-time,0))%>%       # vertical speed limit
  #mutate(leveltime = ifelse(abs(lead(alt)-alt) < level_bandx, lead(time)-time,0))%>%   # vertical speed limit  # band
  mutate(leveltime = ifelse(is.na(leveltime),0,leveltime))%>%
  
  mutate(leveldistance = ifelse(leveltime > 0, distHaversine(cbind(lon, lat), cbind(lead(lon),lead(lat))),0))%>%
  mutate(leveldistance = ifelse(is.na(leveldistance),0, leveldistance))%>%
  mutate(flown_distance = distHaversine(cbind(lon, lat), cbind(lead(lon),lead(lat))))%>%
  mutate(mean_alt_leveloffs = alt*leveltime)


vertical_eff <- vertical_eff %>%
  summarise(kpi17_distance = sum(leveldistance, na.rm = T),
            flown_distance_kpi17 = sum(flown_distance, na.rm = T),
            kpi17_time = sum(leveltime, na.rm = T),
            transit_time_kpi17 = last(time)-first(time)) 
 
#mutate(across(.cols = everything(), ~ ifelse(is.infinite(.x), 0, .x)))  

vertical_eff[is.na(vertical_eff)] <- 0
vertical_eff[mapply(is.infinite, vertical_eff)] <- 0

#----------------------------------------------------------------------------

traj2 <- traj2 %>%
  left_join(energy_10, by = "flight_id")%>%
  left_join(vlof, by = "flight_id")%>%
  left_join(alligier, by = "flight_id")%>%
  left_join(vertical_eff, by = "flight_id")
            
traj2[sapply(traj2, is.infinite)] <- NA


save(traj2, file = paste0("processed/",files[i,"date"],".rda"))
  
processed_files[[i]] <- traj2

})
}

traj_features <- bind_rows(processed_files)

write_csv(traj_features, "trajectory_features4.csv")

filesrda <- list.files("processed", full.names = T)
filelist <- list()
for (f in 1:length(filesrda)) {
  load(filesrda[f])
  filelist[[f]] <- traj2
}
trajectory_features <- bind_rows(filelist)

trajectory_features <- dataset %>%
  inner_join(trajectory_features, by = "flight_id")

save(trajectory_features, file = "trajectory_features.rda")
write_csv(trajectory_features, file = "trajectory_features.csv")
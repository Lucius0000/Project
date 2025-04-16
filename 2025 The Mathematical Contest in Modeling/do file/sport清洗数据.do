gen sport_clean = proper(sport)
replace sport_clean = subinstr( sport_clean , "-", " ", .)
replace sport = trim(sport)
drop sport
rename sport_clean sport
replace sport = "Baseball and Softball" if sport == "Baseball/Softball"

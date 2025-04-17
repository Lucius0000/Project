use "C:\Users\Lucius\Desktop\do file\sport_medal.dta" 
browse
recast int sport_gold
recast int sport_silver
recast int sport_bronze
recast int sport_total
gsort year noc - sport_total
rename sport_total sport_medal
sum sport_medal if noc =="USA" & year == 2024
save "C:\Users\Lucius\Desktop\do file\sport_medal.dta", replace
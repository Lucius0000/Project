import delimited "C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_athletes.csv"
browse
distinct noc
sort noc
import delimited "C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_athletes.csv", clear 
distinct noc
browse
merge m:1 noc using "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta"
sort _merge noc
drop if _merge == 2
distinct noc if _merge == 1
merge m:1 noc using "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta"
import delimited "C:\Users\Lucius\Desktop\2025_Problem_C_Data\cleaned_athlete_data.csv", clear 
distinct noc
merge m:1 noc using "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta"
sort _merge noc
replace country = "Lebanon" if noc == "LIB"
replace country = "Russia" if noc == "ROC"
replace country = "United Arab Republic" if noc == "UAR"
drop in 923/924
replace country = "South Vietnam" if noc == "VNM"
replace country = "West Indies Federation" if noc == "WIF"
replace country = "North Yemen" if noc == "YAR"
replace country = "South Yemen" if noc == "YMD"
drop _merge
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_athletes.dta", replace
merge m:1 year using "C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_hosts.dta"
sort _merge
drop in 1/2
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_athletes.dta", replace
export delimited using "C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.csv", replace
use "C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.dta" 
browse
drop _merge
drop host_country
merge m:m year using "C:\Users\Lucius\Desktop\data_changed\summerOly_hosts.dta"
drop if _merge == 2
drop _merge
save "C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.dta", replace
export delimited using "athlete", replace
import delimited "C:\Users\Lucius\Desktop\2025_Problem_C_Data\cleaned_data.csv"
browse
merge m:1 noc using "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta"
sort _merge noc
drop if _merge == 2
import delimited "C:\Users\Lucius\Desktop\2025_Problem_C_Data\cleaned_data.csv", clear 
merge m:m noc using "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta"
sort _merge
sort _merge noc
drop if _merge==2
replace A = "AFG" in 1
replace A = "AFG" in 2
replace A = "ROC" in 15
replace A = "TPE" in 18
drop in 18/18
drop in 3/14
drop in 4/8
replace A = "RUS" in 1085
drop _merge
replace noc = "Russia" in 3
rename A country
rename country A
rename noc country
rename A noc
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_medal_counts.dta"
export delimited using "C:\Users\Lucius\Desktop\data_changed\summerOly_medal_counts.csv", replace
use "C:\Users\Lucius\Desktop\data_changed\summerOly_medal_counts.dta" 
merge m:m year using "C:\Users\Lucius\Desktop\data_changed\summerOly_hosts.dta"
browse
sort _merge
drop in 1/3
save "C:\Users\Lucius\Desktop\data_changed\summerOly_medal_counts.dta", replace
export delimited using "medal", replace
import delimited "C:\Users\Lucius\Desktop\2025_MCM-ICM_Problems\2025_Problem_C_Data\summerOly_hosts.csv"
browse
drop if strpos(host,",") == 0
gen host_country = substr(host,strpos(host,",")+3,.)
replace host_country = "Japan" if host_country == "Japan (postponed to 2021 due to the coronavirus pandemic)"
drop host
insobs 1
replace year = 1906 if year == .
replace host_country = "Greece" if year == 1906
sort year
use "C:\Users\Lucius\Desktop\data_changed\summerOly_hosts.dta" 
browse
merge m:1 host_country using "C:\Users\Lucius\Desktop\data_changed\country.dta"
drop if _merge == 2
replace host_country = "Great Britain" if host_country == "United Kingdom"
replace noc = "GBR" if host_country == "Great Britain"
rename noc host_noc
drop _merge
save "C:\Users\Lucius\Desktop\data_changed\summerOly_hosts.dta", replace
export delimited using "host", replace

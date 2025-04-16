import excel "C:\Users\Lucius\Desktop\country.xlsx", sheet("Sheet1")
drop C 
browse
rename A noc
rename B country
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta"
rename noc A
rename country noc
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
rename noc team
rename A noc
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
rename team country
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
drop if noc=="noc"
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
import excel "C:\Users\Lucius\Desktop\country.xlsx", sheet("Sheet1") clear
drop C
rename A noc
rename B country
rename noc A
rename country noc
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
replace noc = "" in 11
replace A = "" in 11
sort A
drop in 1/1
replace noc = "c" in 1
drop in 229/229
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
replace noc = "Afghanistanc" in 1
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
duplicates list noc
drop in 172/172
drop in 123/123
rename noc country
rename A noc
save "C:\Users\Lucius\Desktop\2025_Problem_C_Data\country.dta", replace
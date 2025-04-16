gen sport_event = sport + "_" + event
save "C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.dta", replace

#python代码输出后

use "C:\Users\Lucius\Desktop\do file\output.dta" 
browse
sort year true_gold
sort year - true_gold
merge m:m year country using "C:\Users\Lucius\Desktop\data_changed\summerOly_medal_counts.dta"
/*

    Result                      Number of obs
    -----------------------------------------
    Not matched                           505
        from master                        29  (_merge==1)
        from using                        476  (_merge==2)

    Matched                               941  (_merge==3)
    -----------------------------------------
*/
drop if _merge == 2
sort _merge
drop if year == 1906
count if _merge == 1
//1
drop if _merge == 1
sort year country
count if gold != true_gold
gen diff = true_gold - gold
sum diff
sort diff
gen true_diff = diff if diff != 0
sum true_diff
/*

. sum true_diff

    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
   true_diff |         63    .4603175    1.574242         -3          6

   */
sort true_diff
drop _merge

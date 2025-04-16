use "C:\Users\Lucius\Desktop\do file\re_medal.dta" 
browse
merge m:m year noc using "C:\Users\Lucius\Desktop\data_changed\summerOly_medal_counts.dta"
drop if _merge != 3
drop _merge 
gen diff = sport_total - total
sum diff
gen diff_true = diff if diff != 0
sum diff_true

/*
 sum diff

    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
        diff |      1,411    .0460666    .9179092         -6         17

 sum diff_true

    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
   diff_true |        138    .4710145    2.910319         -6         17
*/
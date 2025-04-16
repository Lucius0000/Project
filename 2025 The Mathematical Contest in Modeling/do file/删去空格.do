* 获取所有字符串变量
ds, has(type string)

* 对所有字符串变量应用 trim()
foreach var of varlist `r(varlist)' {
    replace `var' = trim(`var')
}


replace noc = regexr(noc, "\s+", "")

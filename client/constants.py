sql_statements = [
''''SELECT
   `Detected City`, COUNT(CAST(`Age Bracket` AS FLOAT32) CLAMPED BETWEEN 0 AND 20 )  AS `Num`,  `Gender`, `Age Bracket`
 FROM covid_all_data WHERE `CURRENT STATUS` = "Hospitalized"
 GROUP BY `Detected City`,  `Gender`, `Age Bracket`' ''',
''''SELECT
   `Detected City`, COUNT(CAST(`Age Bracket` AS FLOAT32) CLAMPED BETWEEN 0 AND 20)  AS `Num`,  `Gender`, `Age Bracket`
 FROM covid_all_data WHERE `Detected State` = "Meghalaya" AND `CURRENT STATUS` = "Hospitalized"
 GROUP BY `Detected City`,  `Gender`, `Age Bracket`' ''',
 ''''SELECT 
    MEDIAN(CAST(`Age Bracket` AS FLOAT32) CLAMPED BETWEEN 30 AND 100 )  AS `Average Age`
    FROM covid_all_data GROUP BY `Gender`' 
 ''',
 ''''SELECT 
    AVG(CAST(`Age Bracket` AS FLOAT32)  CLAMPED BETWEEN 10 AND 50)  AS `Average Age`, `GENDER`
    FROM covid_all_data GROUP BY `Gender`' 
 '''
]

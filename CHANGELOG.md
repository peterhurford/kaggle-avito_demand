```
LOG (Comp start 25 Apr, merge deadline 20 June @ 7pm EDT, end 27 June @ 7pm EDT) (26/60 submits used as of 7 May UTC) -- Average Delta 0.0035, Safety Margin 0.002
LGB: no text, geo, date, image, param data, or item_seq_number                   - Dim 51,    5CV 0.23129, Submit 0.2355, Delta -.00421
LGB: +missing data, +OHE params (no text, geo, date, image, or item_seq_number)  - Dim 5057,  5CV 0.22694, Submit 0.2305, Delta -.00356
LGB: +basic NLP (no other text, geo, date, image, or item_seq_number)            - Dim 5078,  5CV 0.22607, Submit 0.2299, Delta -.00383  <a9e424c>
LGB: +date (no other text, geo, image, or item_seq_number)                       - Dim 5086,  5CV 0.22610, Submit ?                      <f6c28f2>
LGB: +OHE city and region (no other text, image, or item_seq_number)             - Dim 6866,  5CV 0.22540, Submit ?                      <531df17>
LGB: +item_seq_numbur (no other text or image)                                   - Dim 6867,  5CV 0.22517, Submit ?                      <624f1a4>
LGB: +more basic NLP (no other text or image)                                    - Dim 6877,  5CV 0.22508, Submit 0.2290, Delta -.00392  <f47d17d>
LGB: +SelectKBest TFIDF description + text (no image)                            - Dim 54877, 5CV 0.22206, Submit 0.2257, Delta -.00364  <7002d68>
LGB: +LGB Encode Cats and Ridge Encode text, -some missing vars, -weekend        - Dim 46,    5CV 0.22120, Submit ?
LGB: +Ridge Encoding title                                                       - Dim 47,    5CV 0.22047, Submit 0.2237, Delta -.00323  <6a183fb>
LGB: -some NLP +some NLP                                                         - Dim 50,    5CV 0.22040, Submit 0.2237, Delta -.00330  <954e3ad>
LGB: +adjusted_seq_num, +user_num_days, +user_days_range                         - Dim 53,    5CV 0.22005, Submit 0.2235, Delta -.00345  <e7ea303>
LGB: +recode city                                                                - Dim 53,    5CV 0.22006, Submit ?                      <2054ce2>
LGB: +normalize desc                                                             - Dim 53,    5CV 0.22002, Submit ?                      <87b52f7>
LGB: +text/title ridge                                                           - Dim 54,    5CV 0.21991, Submit ?                      <abd76a4>
LGB: +SVD(title, 10) +SVD(description, 10) +SVD(titlecat, 10) +SVD(text/title)   - Dim 94,    5CV 0.21967, Submit 0.2230, Delta -.00333  <6e94776>
LGB: +Deep text LGB                                                              - Dim 95,    5CV 0.21862, Submit 0.2217, Delta -.00308  <be831c5>
LGB: +Some tuning                                                                - Dim 95,    5CV 0.21778, Submit ?0.2213?               <627d398>
LGB: +Num unique words +Unique words ratio                                       - Dim 97,    5CV 0.21782, Submit ?0.2214?               <2bcd64e>
LGB: +cat_price_mean +cat_price_diff                                             - Dim 99,    5CV 0.21768, Submit 0.2209, Delta -.00322  <0c9e1e4>
LGB: +lat/lon of cities                                                          - Dim 101,   5CV 0.21766, Submit ?0.2212?               <a3d9005>
LGB: +parent_cat_count, region_X_cat_count                                       - Dim 103,   5CV 0.21763, Submit ?0.2211?
LGB: +city_count                                                                 - Dim 104,   5CV 0.21747, Submit ?0.2210?
LGB: +Region macro +improve title/text Ridge +Text char Ridge                    - Dim 108,   5CV 0.21733, Submit 0.2206, Delta -.00327  <4c18106>
LGB: -title/text Ridge and SVD, +improve Ridges                                  - Dim 97,    5CV 0.21723, Subnit ?0.2207?               <e840c9e>
LGB: +All text Ridge                                                             - Dim 98,    5CV 0.21717, Submit ?0.2207?               <c8e9ada>
LGB: +Add sentence basic NLP                                                     - Dim 101,   5CV 0.21719, Submit ?0.2207?               <528cece>
LGB: +Add parent cat Ridges                                                      - Dim 105,   5CV 0.21679, Submit ?0.2203?
LGB: +Add parent_catXregion Ridges                                               - Dim 109,   5CV ?
LGB: +Add cat_bin                                                                - Dim 110,   5CV ?
LGB: +Add cat_bin Ridges                                                         - Dim 114,   5CV ?
LGB: +More region macro                                                          - Dim ?,     5CV ?
LGB: +City macro                                                                 - Dim ?,     5CV ?
LGB: +More user variables from supplementary data                                - Dim ?,     5CV ?
LGB: +Recalculate category counts with supplementary data                        - Dim ?,     5CV ?
```

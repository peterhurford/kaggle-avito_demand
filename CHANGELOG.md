```
LGB: no text, geo, date, image, param data, or item_seq_number                    - Dim 51,    5CV 0.23129, Submit 0.2355, Delta -.00421
LGB: +missing data, +OHE params (no text, geo, date, image, or item_seq_number)   - Dim 5057,  5CV 0.22694, Submit 0.2305, Delta -.00356
LGB: +basic NLP (no other text, geo, date, image, or item_seq_number)             - Dim 5078,  5CV 0.22607, Submit 0.2299, Delta -.00383  <a9e424c>
LGB: +date (no other text, geo, image, or item_seq_number)                        - Dim 5086,  5CV 0.22610, Submit ?                      <f6c28f2>
LGB: +OHE city and region (no other text, image, or item_seq_number)              - Dim 6866,  5CV 0.22540, Submit ?                      <531df17>
LGB: +item_seq_number (no other text or image)                                    - Dim 6867,  5CV 0.22517, Submit ?                      <624f1a4>
LGB: +more basic NLP (no other text or image)                                     - Dim 6877,  5CV 0.22508, Submit 0.2290, Delta -.00392  <f47d17d>
LGB: +SelectKBest TFIDF description + text (no image)                             - Dim 54877, 5CV 0.22206, Submit 0.2257, Delta -.00364  <7002d68>
LGB: +LGB Encode Cats and Ridge Encode text, -some missing vars, -weekend         - Dim 46,    5CV 0.22120, Submit ?
LGB: +Ridge Encoding title                                                        - Dim 47,    5CV 0.22047, Submit 0.2237, Delta -.00323  <6a183fb>
LGB: -some NLP +some NLP                                                          - Dim 50,    5CV 0.22040, Submit 0.2237, Delta -.00330  <954e3ad>
LGB: +adjusted_seq_num, +user_num_days, +user_days_range                          - Dim 53,    5CV 0.22005, Submit 0.2235, Delta -.00345  <e7ea303>
LGB: +recode city                                                                 - Dim 53,    5CV 0.22006, Submit ?                      <2054ce2>
LGB: +normalize desc                                                              - Dim 53,    5CV 0.22002, Submit ?                      <87b52f7>
LGB: +text/title ridge                                                            - Dim 54,    5CV 0.21991, Submit ?                      <abd76a4>
LGB: +SVD(title, 10) +SVD(description, 10) +SVD(titlecat, 10) +SVD(text/title)    - Dim 94,    5CV 0.21967, Submit 0.2230, Delta -.00333  <6e94776>
LGB: +Deep text LGB                                                               - Dim 95,    5CV 0.21862, Submit 0.2217, Delta -.00308  <be831c5>
LGB: +Some tuning                                                                 - Dim 95,    5CV 0.21778, Submit ?0.2209?               <627d398>
LGB: +Num unique words +Unique words ratio                                        - Dim 97,    5CV 0.21782, Submit ?0.2209?               <2bcd64e>
LGB: +cat_price_mean +cat_price_diff                                              - Dim 99,    5CV 0.21768, Submit 0.2209, Delta -.00322  <0c9e1e4>
LGB: +lat/lon of cities                                                           - Dim 101,   5CV 0.21766, Submit ?0.2209?               <a3d9005>
LGB: +parent_cat_count, region_X_cat_count                                        - Dim 103,   5CV 0.21763, Submit ?0.2209?
LGB: +city_count                                                                  - Dim 104,   5CV 0.21747, Submit ?0.2207?
LGB: +Region macro +improve title/text Ridge +Text char Ridge                     - Dim 108,   5CV 0.21733, Submit 0.2206, Delta -.00327  <4c18106>
LGB: -title/text Ridge and SVD, +improve Ridges                                   - Dim 97,    5CV 0.21723, Subnit ?0.2205?               <e840c9e>
LGB: +All text Ridge                                                              - Dim 98,    5CV 0.21717, Submit ?0.2205?               <c8e9ada>
LGB: +Add sentence basic NLP                                                      - Dim 101,   5CV 0.21719, Submit ?0.2205?               <528cece>
LGB: +Add parent cat Ridges                                                       - Dim 105,   5CV 0.21679, Submit ?0.2201?               <6b70f46>
LGB: +Add parent_catXregion Ridges                                                - Dim 109,   5CV 0.21650, Submit ?0.2198?               <a40f84e>
LGB: +Some tuning                                                                 - Dim 109,   5CV 0.21640, Submit 0.2202, Delta -.00380  <867c0df>
LGB: +Add cat_bin                                                                 - Dim 110,   5CV 0.21636, Submit ?0.2202?               <b038ddd>
LGB: +Add cat_bin Ridges, LR 0.04->0.03                                           - Dim 114,   5CV 0.21611, Submit 0.2200, Delta -.00389  <df0d5e9>
LGB: +Add 10 more SVD dimensions                                                  - Dim 144,   5CV 0.21611, Submit ?0.2200?               <ef3f318>
LGB: +Image stats                                                                 - Dim 171,   5CV 0.21522, Submit 0.2191, Delta -.00388  <4d1d645>
LGB: -image_missing, +tuning                                                      - Dim 170,   5CV 0.21509, Submit ?0.2190?               <ef8b654>
LGB: +avg_times_up_user, +n_user_items, +avg_days_up_user                         - Dim 173,   5CV 0.21407, Submit 0.2184, Delta -.00433  <1998840>
LGB: +user_items_per_day                                                          - Dim 174,   5CV 0.21405, Submit ?0.2184?               <3ca9d36>
LGB: +retrain deep LGB with new features, +complete Ridge                         - Dim 174,   5CV 0.21388, Submit ?.21821?
BLEND: +FM models                                                                 - Dim 85,    5CV 0.21342, Submit 0.2178, Delta -.00438  <8c01031>
BLEND: +Add NIMA to LGBs                                                          - Dim 89,    5CV 0.21332, Submit 0.2177, Delta -.00438
BLEND: +Add Base LGB                                                              - Dim 90,    5CV 0.21324, Submit ?0.2176?
BLEND: +Add CNN FT, +Add TFIDF stats                                              - Dim 100,   5CV 0.21309, Submit ?0.2175?
BLEND: +CNN FT3                                                                   - Dim 101,   5CV 0.21301, Submit ?0.2174?
BLEND: +Improve CNN FT3                                                           - Dim 101,   5CV 0.21296, Submit ?0.2173?
BLEND: +Add SVD of Embedding to Ridge LGB, -Stack LGB                             - Dim 101,   5CV 0.21261, Submit ?0.2170?
BLEND: +Add CNN FT4                                                               - Dim 102,   5CV 0.21253, Submit 0.2173, Delta -.00477
BLEND: +Add Ridge, +SVD of Embedding, -SVD dimensions. -TFIDF Stats               - Dim 50,    5CV 0.21242, Submit ?0.2172?
BLEND: +Add Embedding to Base LGB                                                 - Dim 50,    5CV 0.21210, Submit ?0.2169?
BLEND: +Add TE LGB                                                                - Dim 51,    5CV 0.21168, Submit ?0.2165?
BLEND: +Add Ryan's LGB, +Ryan's Ridge, +Ryan's SGD                                - Dim 54,    5CV 0.21147, Submit 0.2166, Delta -.00513
BLEND: -FT3                                                                       - Dim 55,    5CV 0.21149, Submit ?0.2166?
Blend: +Deep LGB2                                                                 - Dim 54,    5CV 0.21146, Submit ?0.2166?
Blend: +retrain Base LGB (with Ryan features)                                     - Dim 54,    5CV 0.21141, Submit ?0.2165?
Blend: +Liu Jilong's NN                                                           - Dim 55,    5CV 0.21095, Submit ?0.2161?
Blend: +Liu Jilong's LGB                                                          - Dim 56,    5CV 0.21092, Submit ?0.2161?
Blend: +Tuning                                                                    - Dim 56,    5CV 0.21088, Submit 0.2158, Delta -.00492
```


module mul8_15 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n216, n217,
         n218, n219, n220, n221, n222, n223, n224, n225, n226, n227, n228,
         n229, n230, n231, n232, n233, n234, n235, n236, n237, n238, n239,
         n240, n241, n242, n243, n244, n245, n246, n247, n248, n249, n250,
         n251, n252, n253, n254, n255, n256, n257, n258, n259, n260, n261,
         n262, n263, n264, n265, n266, n267, n268, n269, n270, n271, n272,
         n273, n274, n275, n276, n277, n278, n279, n280, n281, n282, n283,
         n284, n285, n286, n287, n288, n289, n290, n291, n292, n293, n294,
         n295, n296, n297, n298, n299, n300, n301, n302, n303, n304, n305,
         n306, n307, n308, n309, n310, n311, n312, n313, n314, n315, n316,
         n317, n318, n319, n320, n321, n322, n323, n324, n325, n326, n327,
         n328, n329, n330, n331, n332, n333, n334, n335, n336, n337, n338,
         n339, n340, n341, n342, n343, n344, n345, n346, n347, n348, n349,
         n350, n351, n352, n353, n354, n355, n356, n357, n358, n359, n360,
         n361, n362, n363, n364, n365, n366, n367, n368, n369, n370, n371,
         n372, n373, n374, n375, n376, n377, n378, n379, n380, n381, n382,
         n383, n384, n385, n386, n387, n388, n389, n390, n391, n392, n393,
         n394, n395, n396, n397, n398, n399, n400, n401, n402, n403, n404,
         n405, n406, n407, n408, n409, n410, n411, n412;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U206 ( .A1(n223), .B1(n256), .ZN(n243) );
  VHSR_NOR2_1 U207 ( .A1(n277), .A2(n313), .ZN(n290) );
  VHSR_NOR2_1 U208 ( .A1(n299), .A2(n303), .ZN(n298) );
  VHSR_NOR2_1 U209 ( .A1(n296), .A2(n297), .ZN(n352) );
  VHSR_IOA21_2 U210 ( .A1(n402), .A2(n401), .B(n400), .ZN(n404) );
  VHSR_NOR2_1 U211 ( .A1(n318), .A2(n313), .ZN(n380) );
  VHSR_IN_2 U212 ( .I(n363), .ZN(product[13]) );
  VHSR_CLKN_1 U213 ( .I(n368), .ZN(n369) );
  VHSR_INAND3_1 U214 ( .A1(n395), .B1(n398), .B2(n397), .ZN(n368) );
  VHSR_INOR3_1 U215 ( .A1(n240), .B1(n323), .B2(n282), .ZN(n301) );
  VHSR_INOR2_1 U216 ( .A1(n367), .B1(n366), .ZN(n398) );
  VHSR_NOR2_2 U217 ( .A1(n250), .A2(n246), .ZN(n240) );
  VHSR_NOR2_2 U218 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_INOR2_1 U219 ( .A1(n353), .B1(n352), .ZN(n365) );
  VHSR_NOR2_2 U220 ( .A1(n298), .A2(n294), .ZN(n296) );
  VHSR_INAND2_1 U221 ( .A1(n358), .B1(n356), .ZN(n359) );
  VHSR_INOR3_1 U222 ( .A1(n290), .B1(n282), .B2(n316), .ZN(n360) );
  VHSR_INOR2_1 U223 ( .A1(n380), .B1(n288), .ZN(n292) );
  VHSR_AD1_1 U224 ( .A(n386), .B(n385), .CI(n403), .CO(n382), .S(product[5])
         );
  VHSR_AD1_1 U225 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U226 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(product[10])
         );
  VHSR_AD1_1 U227 ( .A(n388), .B(n410), .CI(n387), .CO(n350), .S(product[3])
         );
  VHSR_AD1_1 U228 ( .A(n384), .B(n383), .CI(n382), .CO(n389), .S(product[6])
         );
  VHSR_AD1_1 U229 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(product[9])
         );
  VHSR_AD1_1 U230 ( .A(n372), .B(n371), .CI(n370), .CO(n392), .S(product[11])
         );
  VHSR_IN_2 U231 ( .I(b[7]), .ZN(n284) );
  VHSR_IN_2 U232 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U233 ( .I(b[6]), .ZN(n285) );
  VHSR_IN_2 U234 ( .I(a[2]), .ZN(n325) );
  VHSR_OAI22_2 U235 ( .A1(n285), .A2(n324), .B1(n284), .B2(n325), .ZN(n245) );
  VHSR_NOR2_1 U236 ( .A1(n284), .A2(n325), .ZN(n217) );
  VHSR_IN_2 U237 ( .I(a[1]), .ZN(n406) );
  VHSR_NOR2_1 U238 ( .A1(n285), .A2(n406), .ZN(n216) );
  VHSR_IN_2 U239 ( .I(b[5]), .ZN(n316) );
  VHSR_AOI211_2 U240 ( .A1(b[4]), .A2(a[2]), .B(n316), .C(n324), .ZN(n222) );
  VHSR_OAI22_2 U241 ( .A1(n285), .A2(n325), .B1(n284), .B2(n406), .ZN(n221) );
  VHSR_AOI22_2 U242 ( .A1(n217), .A2(n216), .B1(n222), .B2(n221), .ZN(n223) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[4]), .A2(a[2]), .ZN(n268) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[4]), .A2(a[0]), .ZN(n401) );
  VHSR_NAND3_2 U245 ( .A1(a[1]), .A2(b[5]), .A3(n401), .ZN(n267) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[6]), .A2(a[0]), .ZN(n266) );
  VHSR_MAOI222_2 U247 ( .A(n268), .B(n267), .C(n266), .ZN(n265) );
  VHSR_NOR3_2 U248 ( .A1(n316), .A2(n406), .A3(n401), .ZN(n273) );
  VHSR_NAND4_2 U249 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n242) );
  VHSR_IN_2 U250 ( .I(b[4]), .ZN(n313) );
  VHSR_OAI22_2 U251 ( .A1(n313), .A2(n324), .B1(n316), .B2(n325), .ZN(n218) );
  VHSR_AND2_2 U252 ( .A1(n242), .A2(n218), .Z(n220) );
  VHSR_IN_2 U253 ( .I(a[0]), .ZN(n408) );
  VHSR_OAI22_2 U254 ( .A1(n285), .A2(n406), .B1(n284), .B2(n408), .ZN(n219) );
  VHSR_AND2_2 U255 ( .A1(n265), .A2(n261), .Z(n260) );
  VHSR_AD1_1 U256 ( .A(n273), .B(n220), .CI(n219), .CO(n255), .S(n261) );
  VHSR_NOR2_1 U257 ( .A1(n260), .A2(n255), .ZN(n258) );
  VHSR_OAI21_2 U258 ( .A1(n222), .A2(n221), .B(n223), .ZN(n259) );
  VHSR_NOR2_1 U259 ( .A1(n258), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U260 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U261 ( .A1(n245), .A2(n241), .ZN(n238) );
  VHSR_NOR3_2 U262 ( .A1(n284), .A2(n324), .A3(n238), .ZN(n302) );
  VHSR_AOI22_2 U263 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n250) );
  VHSR_IN_2 U264 ( .I(b[3]), .ZN(n323) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[2]), .A2(a[4]), .ZN(n272) );
  VHSR_IN_2 U266 ( .I(a[5]), .ZN(n314) );
  VHSR_NOR3_2 U267 ( .A1(n323), .A2(n272), .A3(n314), .ZN(n248) );
  VHSR_IN_2 U268 ( .I(a[7]), .ZN(n282) );
  VHSR_IN_2 U269 ( .I(a[6]), .ZN(n277) );
  VHSR_IN_2 U270 ( .I(b[1]), .ZN(n409) );
  VHSR_NOR2_1 U271 ( .A1(n277), .A2(n409), .ZN(n231) );
  VHSR_IN_2 U272 ( .I(n231), .ZN(n234) );
  VHSR_NOR2_1 U273 ( .A1(n282), .A2(n234), .ZN(n224) );
  VHSR_AOI211_2 U274 ( .A1(a[4]), .A2(b[2]), .B(n323), .C(n314), .ZN(n225) );
  VHSR_IN_2 U275 ( .I(b[2]), .ZN(n326) );
  VHSR_NOR2_1 U276 ( .A1(n277), .A2(n326), .ZN(n226) );
  VHSR_MAOI222_2 U277 ( .A(n224), .B(n225), .C(n226), .ZN(n237) );
  VHSR_AOI31_2 U278 ( .A1(a[7]), .A2(a[6]), .A3(b[1]), .B(n225), .ZN(n229) );
  VHSR_IN_2 U279 ( .I(n226), .ZN(n228) );
  VHSR_IN_2 U280 ( .I(n237), .ZN(n227) );
  VHSR_AOI21_2 U281 ( .A1(n229), .A2(n228), .B(n227), .ZN(n253) );
  VHSR_IN_2 U282 ( .I(a[4]), .ZN(n318) );
  VHSR_IN_2 U283 ( .I(b[0]), .ZN(n407) );
  VHSR_NOR4_2 U284 ( .A1(n318), .A2(n314), .A3(n409), .A4(n407), .ZN(n276) );
  VHSR_AOI22_2 U285 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n230) );
  VHSR_NOR2_1 U286 ( .A1(n248), .A2(n230), .ZN(n232) );
  VHSR_MAOI222_2 U287 ( .A(n231), .B(n276), .C(n232), .ZN(n236) );
  VHSR_OAI211_2 U288 ( .A1(n318), .A2(n407), .B(a[5]), .C(b[1]), .ZN(n271) );
  VHSR_CLKNAND2_2 U289 ( .A1(a[6]), .A2(b[0]), .ZN(n270) );
  VHSR_MAOI222_2 U290 ( .A(n272), .B(n271), .C(n270), .ZN(n269) );
  VHSR_NOR2_1 U291 ( .A1(n276), .A2(n232), .ZN(n235) );
  VHSR_IN_2 U292 ( .I(n236), .ZN(n233) );
  VHSR_AOI21_2 U293 ( .A1(n235), .A2(n234), .B(n233), .ZN(n263) );
  VHSR_CLKNAND2_2 U294 ( .A1(n269), .A2(n263), .ZN(n262) );
  VHSR_CLKNAND2_2 U295 ( .A1(n236), .A2(n262), .ZN(n252) );
  VHSR_CLKNAND2_2 U296 ( .A1(n253), .A2(n252), .ZN(n251) );
  VHSR_CLKNAND2_2 U297 ( .A1(n237), .A2(n251), .ZN(n247) );
  VHSR_OAI32_2 U298 ( .A1(n302), .A2(n324), .A3(n284), .B1(n238), .B2(n302), 
        .ZN(n309) );
  VHSR_NOR2_1 U299 ( .A1(n323), .A2(n282), .ZN(n239) );
  VHSR_IAO21_2 U300 ( .A1(n240), .A2(n239), .B(n301), .ZN(n308) );
  VHSR_OAI21_2 U301 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U302 ( .A1(n245), .A2(n244), .ZN(n312) );
  VHSR_AOI21_2 U303 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U304 ( .A1(n250), .A2(n249), .ZN(n311) );
  VHSR_OAI21_2 U305 ( .A1(n253), .A2(n252), .B(n251), .ZN(n254) );
  VHSR_IN_2 U306 ( .I(n254), .ZN(n321) );
  VHSR_CLKNAND2_2 U307 ( .A1(n260), .A2(n255), .ZN(n257) );
  VHSR_AOI22_2 U308 ( .A1(n259), .A2(n258), .B1(n257), .B2(n256), .ZN(n320) );
  VHSR_IAO21_2 U309 ( .A1(n265), .A2(n261), .B(n260), .ZN(n336) );
  VHSR_OAI21_2 U310 ( .A1(n269), .A2(n263), .B(n262), .ZN(n264) );
  VHSR_IN_2 U311 ( .I(n264), .ZN(n335) );
  VHSR_AOI31_2 U312 ( .A1(n268), .A2(n267), .A3(n266), .B(n265), .ZN(n343) );
  VHSR_AOI31_2 U313 ( .A1(n272), .A2(n271), .A3(n270), .B(n269), .ZN(n342) );
  VHSR_AOI22_2 U314 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n274) );
  VHSR_NOR2_1 U315 ( .A1(n274), .A2(n273), .ZN(n345) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[4]), .A2(b[1]), .ZN(n275) );
  VHSR_OAI32_2 U317 ( .A1(n276), .A2(n314), .A3(n407), .B1(n275), .B2(n276), 
        .ZN(n344) );
  VHSR_NOR2_1 U318 ( .A1(n407), .A2(n408), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U319 ( .A1(n380), .A2(product[0]), .ZN(n400) );
  VHSR_IN_2 U320 ( .I(n400), .ZN(n351) );
  VHSR_NOR2_1 U321 ( .A1(n277), .A2(n285), .ZN(n395) );
  VHSR_NOR2_1 U322 ( .A1(n318), .A2(n285), .ZN(n289) );
  VHSR_CLKNAND2_2 U323 ( .A1(a[5]), .A2(b[7]), .ZN(n279) );
  VHSR_CLKNAND2_2 U324 ( .A1(a[7]), .A2(b[5]), .ZN(n278) );
  VHSR_OAI22_2 U325 ( .A1(n289), .A2(n279), .B1(n290), .B2(n278), .ZN(n281) );
  VHSR_OR2_2 U326 ( .A1(n289), .A2(n290), .Z(n304) );
  VHSR_CLKNAND2_2 U327 ( .A1(a[5]), .A2(b[5]), .ZN(n288) );
  VHSR_CLKNAND2_2 U328 ( .A1(a[7]), .A2(b[7]), .ZN(n396) );
  VHSR_NOR3_2 U329 ( .A1(n304), .A2(n288), .A3(n396), .ZN(n280) );
  VHSR_AOI31_2 U330 ( .A1(b[6]), .A2(a[6]), .A3(n281), .B(n280), .ZN(n353) );
  VHSR_OAI21_2 U331 ( .A1(n395), .A2(n281), .B(n353), .ZN(n297) );
  VHSR_AOI22_2 U332 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n283) );
  VHSR_NOR2_1 U333 ( .A1(n360), .A2(n283), .ZN(n293) );
  VHSR_NOR4_2 U334 ( .A1(n318), .A2(n314), .A3(n285), .A4(n284), .ZN(n358) );
  VHSR_AOI22_2 U335 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n286) );
  VHSR_NOR2_1 U336 ( .A1(n358), .A2(n286), .ZN(n291) );
  VHSR_IN_2 U337 ( .I(n287), .ZN(n299) );
  VHSR_NOR2_1 U338 ( .A1(n380), .A2(n288), .ZN(n305) );
  VHSR_AOI22_2 U339 ( .A1(n290), .A2(n289), .B1(n305), .B2(n304), .ZN(n303) );
  VHSR_AD1_1 U340 ( .A(n293), .B(n292), .CI(n291), .CO(n294), .S(n287) );
  VHSR_CLKNAND2_2 U341 ( .A1(n298), .A2(n294), .ZN(n295) );
  VHSR_AOI22_2 U342 ( .A1(n297), .A2(n296), .B1(n295), .B2(n352), .ZN(n393) );
  VHSR_AOI21_2 U343 ( .A1(n303), .A2(n299), .B(n298), .ZN(n372) );
  VHSR_AD1_1 U344 ( .A(n302), .B(n301), .CI(n300), .CO(n394), .S(n371) );
  VHSR_OAI21_2 U345 ( .A1(n305), .A2(n304), .B(n303), .ZN(n306) );
  VHSR_IN_2 U346 ( .I(n306), .ZN(n375) );
  VHSR_AD1_1 U347 ( .A(n309), .B(n308), .CI(n307), .CO(n300), .S(n374) );
  VHSR_AD1_1 U348 ( .A(n312), .B(n311), .CI(n310), .CO(n307), .S(n378) );
  VHSR_NOR2_1 U349 ( .A1(n314), .A2(n313), .ZN(n317) );
  VHSR_OAI21_2 U350 ( .A1(n318), .A2(n316), .B(n317), .ZN(n315) );
  VHSR_OAI31_2 U351 ( .A1(n318), .A2(n317), .A3(n316), .B(n315), .ZN(n377) );
  VHSR_AD1_1 U352 ( .A(n321), .B(n320), .CI(n319), .CO(n310), .S(n381) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[2]), .A2(a[2]), .ZN(n333) );
  VHSR_IN_2 U354 ( .I(n333), .ZN(n340) );
  VHSR_NOR4_2 U355 ( .A1(n323), .A2(n326), .A3(n406), .A4(n408), .ZN(n348) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[2]), .A2(a[1]), .ZN(n322) );
  VHSR_OAI32_2 U357 ( .A1(n348), .A2(n408), .A3(n323), .B1(n322), .B2(n348), 
        .ZN(n388) );
  VHSR_AOI22_2 U358 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n412) );
  VHSR_NOR3_2 U359 ( .A1(n412), .A2(n408), .A3(n326), .ZN(n410) );
  VHSR_OAI22_2 U360 ( .A1(n409), .A2(n325), .B1(n407), .B2(n324), .ZN(n387) );
  VHSR_OAI211_2 U361 ( .A1(n326), .A2(n408), .B(b[3]), .C(a[1]), .ZN(n327) );
  VHSR_AND2_2 U362 ( .A1(n327), .A2(n333), .Z(n329) );
  VHSR_CLKNAND2_2 U363 ( .A1(b[1]), .A2(a[3]), .ZN(n328) );
  VHSR_MAOI222_2 U364 ( .A(n328), .B(n333), .C(n327), .ZN(n330) );
  VHSR_AOI21_2 U365 ( .A1(n329), .A2(n328), .B(n330), .ZN(n349) );
  VHSR_AOI21_2 U366 ( .A1(n350), .A2(n349), .B(n330), .ZN(n331) );
  VHSR_IN_2 U367 ( .I(n331), .ZN(n347) );
  VHSR_CLKNAND2_2 U368 ( .A1(b[3]), .A2(a[3]), .ZN(n338) );
  VHSR_AOI22_2 U369 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n332) );
  VHSR_IAO21_2 U370 ( .A1(n338), .A2(n333), .B(n332), .ZN(n346) );
  VHSR_IAO21_2 U371 ( .A1(n340), .A2(n339), .B(n338), .ZN(n391) );
  VHSR_AD1_1 U372 ( .A(n336), .B(n335), .CI(n334), .CO(n319), .S(n390) );
  VHSR_OAI21_2 U373 ( .A1(n340), .A2(n338), .B(n339), .ZN(n337) );
  VHSR_OAI31_2 U374 ( .A1(n340), .A2(n339), .A3(n338), .B(n337), .ZN(n384) );
  VHSR_AD1_1 U375 ( .A(n343), .B(n342), .CI(n341), .CO(n334), .S(n383) );
  VHSR_AD1_1 U376 ( .A(n345), .B(n344), .CI(n351), .CO(n341), .S(n386) );
  VHSR_AD1_1 U377 ( .A(n348), .B(n347), .CI(n346), .CO(n339), .S(n385) );
  VHSR_CLKNAND2_2 U378 ( .A1(a[4]), .A2(b[0]), .ZN(n402) );
  VHSR_XNOR2_2 U379 ( .A1(n350), .A2(n349), .ZN(n405) );
  VHSR_AOI211_2 U380 ( .A1(n402), .A2(n401), .B(n351), .C(n405), .ZN(n403) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[7]), .A2(b[6]), .ZN(n355) );
  VHSR_AOI21_2 U382 ( .A1(a[6]), .A2(b[7]), .B(n355), .ZN(n354) );
  VHSR_AOI31_2 U383 ( .A1(a[6]), .A2(n355), .A3(b[7]), .B(n354), .ZN(n356) );
  VHSR_IN_2 U384 ( .I(n356), .ZN(n357) );
  VHSR_MAOI222_2 U385 ( .A(n360), .B(n358), .C(n357), .ZN(n367) );
  VHSR_OAI21_2 U386 ( .A1(n360), .A2(n359), .B(n367), .ZN(n364) );
  VHSR_CLKXOR2_2 U387 ( .A1(n365), .A2(n364), .Z(n361) );
  VHSR_CLKNAND2_2 U388 ( .A1(n362), .A2(n361), .ZN(n397) );
  VHSR_OAI21_2 U389 ( .A1(n362), .A2(n361), .B(n397), .ZN(n363) );
  VHSR_NOR2_1 U390 ( .A1(n365), .A2(n364), .ZN(n366) );
  VHSR_NOR2_1 U391 ( .A1(n396), .A2(n369), .ZN(product[15]) );
  VHSR_AD1_1 U392 ( .A(n391), .B(n390), .CI(n389), .CO(n379), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U393 ( .A(n394), .B(n393), .CI(n392), .CO(n362), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U394 ( .A1(n396), .A2(n395), .ZN(n399) );
  VHSR_XOR3_2 U395 ( .A1(n399), .A2(n398), .A3(n397), .Z(product[14]) );
  VHSR_AOI21_2 U396 ( .A1(n405), .A2(n404), .B(n403), .ZN(product[4]) );
  VHSR_OAI22_2 U397 ( .A1(n409), .A2(n408), .B1(n407), .B2(n406), .ZN(
        product[1]) );
  VHSR_CLKNAND2_2 U398 ( .A1(b[2]), .A2(a[0]), .ZN(n411) );
  VHSR_AOI21_2 U399 ( .A1(n412), .A2(n411), .B(n410), .ZN(product[2]) );
endmodule


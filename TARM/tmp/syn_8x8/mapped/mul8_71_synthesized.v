
module mul8_71 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236, n237, n238,
         n239, n240, n241, n242, n243, n244, n245, n246, n247, n248, n249,
         n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, n260,
         n261, n262, n263, n264, n265, n266, n267, n268, n269, n270, n271,
         n272, n273, n274, n275, n276, n277, n278, n279, n280, n281, n282,
         n283, n284, n285, n286, n287, n288, n289, n290, n291, n292, n293,
         n294, n295, n296, n297, n298, n299, n300, n301, n302, n303, n304,
         n305, n306, n307, n308, n309, n310, n311, n312, n313, n314, n315,
         n316, n317, n318, n319, n320, n321, n322, n323, n324, n325, n326,
         n327, n328, n329, n330, n331, n332, n333, n334, n335, n336, n337,
         n338, n339, n340, n341, n342, n343, n344, n345, n346, n347, n348,
         n349, n350, n351, n352, n353, n354, n355, n356, n357, n358, n359,
         n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, n370,
         n371, n372, n373, n374, n375, n376, n377, n378, n379, n380, n381,
         n382, n383, n384, n385, n386, n387, n388, n389, n390, n391, n392,
         n393, n394, n395, n396, n397, n398, n399, n400, n401, n402, n403,
         n404, n405, n406, n407, n408, n409;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U209 ( .A1(n273), .B1(a[5]), .B2(b[3]), .ZN(n230) );
  VHSR_INOR2_2 U210 ( .A1(n239), .B1(n258), .ZN(n248) );
  VHSR_INOR2_2 U211 ( .A1(n237), .B1(n261), .ZN(n260) );
  VHSR_NOR2_1 U212 ( .A1(n288), .A2(n323), .ZN(n245) );
  VHSR_NOR2_1 U213 ( .A1(n322), .A2(n323), .ZN(n337) );
  VHSR_NOR2_1 U214 ( .A1(n241), .A2(n240), .ZN(n300) );
  VHSR_IOA21_2 U215 ( .A1(n399), .A2(n398), .B(n397), .ZN(n401) );
  VHSR_INOR2_2 U216 ( .A1(n364), .B1(n363), .ZN(n395) );
  VHSR_IN_2 U217 ( .I(n360), .ZN(product[13]) );
  VHSR_NOR2_2 U218 ( .A1(n350), .A2(n349), .ZN(n362) );
  VHSR_AD1_1 U219 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U220 ( .A(n379), .B(n407), .CI(n378), .CO(n342), .S(product[3])
         );
  VHSR_AD1_1 U221 ( .A(n400), .B(n377), .CI(n376), .CO(n380), .S(product[5])
         );
  VHSR_AD1_1 U222 ( .A(n372), .B(n371), .CI(n370), .CO(n386), .S(product[9])
         );
  VHSR_AD1_1 U223 ( .A(n369), .B(n368), .CI(n367), .CO(n389), .S(
        \intadd_0/SUM[6] ) );
  VHSR_IN_2 U224 ( .I(b[6]), .ZN(n288) );
  VHSR_IN_2 U225 ( .I(a[2]), .ZN(n323) );
  VHSR_IN_2 U226 ( .I(b[7]), .ZN(n287) );
  VHSR_NOR2_1 U227 ( .A1(n287), .A2(n323), .ZN(n221) );
  VHSR_IN_2 U228 ( .I(a[3]), .ZN(n324) );
  VHSR_OAI21_2 U229 ( .A1(n288), .A2(n324), .B(n221), .ZN(n220) );
  VHSR_OAI31_2 U230 ( .A1(n288), .A2(n221), .A3(n324), .B(n220), .ZN(n253) );
  VHSR_IN_2 U231 ( .I(b[5]), .ZN(n286) );
  VHSR_CLKNAND2_2 U232 ( .A1(b[4]), .A2(a[2]), .ZN(n269) );
  VHSR_NOR3_2 U233 ( .A1(n286), .A2(n269), .A3(n324), .ZN(n251) );
  VHSR_AOI22_2 U234 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n222) );
  VHSR_NOR2_1 U235 ( .A1(n251), .A2(n222), .ZN(n225) );
  VHSR_IN_2 U236 ( .I(a[1]), .ZN(n403) );
  VHSR_IN_2 U237 ( .I(a[0]), .ZN(n405) );
  VHSR_OAI22_2 U238 ( .A1(n288), .A2(n403), .B1(n287), .B2(n405), .ZN(n224) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[4]), .A2(a[0]), .ZN(n398) );
  VHSR_NOR3_2 U240 ( .A1(n286), .A2(n403), .A3(n398), .ZN(n276) );
  VHSR_IN_2 U241 ( .I(n255), .ZN(n229) );
  VHSR_NOR2_1 U242 ( .A1(n287), .A2(n403), .ZN(n223) );
  VHSR_AOI211_2 U243 ( .A1(b[4]), .A2(a[2]), .B(n286), .C(n324), .ZN(n226) );
  VHSR_MAOI222_2 U244 ( .A(n223), .B(n226), .C(n245), .ZN(n228) );
  VHSR_NAND3_2 U245 ( .A1(a[1]), .A2(b[5]), .A3(n398), .ZN(n268) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[6]), .A2(a[0]), .ZN(n267) );
  VHSR_MAOI222_2 U247 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_AD1_1 U248 ( .A(n225), .B(n224), .CI(n276), .CO(n255), .S(n264) );
  VHSR_CLKNAND2_2 U249 ( .A1(n266), .A2(n264), .ZN(n263) );
  VHSR_OR2_2 U250 ( .A1(n226), .A2(n245), .Z(n227) );
  VHSR_AOI32_2 U251 ( .A1(a[1]), .A2(n228), .A3(b[7]), .B1(n227), .B2(n228), 
        .ZN(n254) );
  VHSR_AOI32_2 U252 ( .A1(n229), .A2(n228), .A3(n263), .B1(n254), .B2(n228), 
        .ZN(n252) );
  VHSR_CLKNAND2_2 U253 ( .A1(b[7]), .A2(a[3]), .ZN(n243) );
  VHSR_IAO21_2 U254 ( .A1(n245), .A2(n244), .B(n243), .ZN(n301) );
  VHSR_CLKNAND2_2 U255 ( .A1(b[3]), .A2(a[7]), .ZN(n241) );
  VHSR_IN_2 U256 ( .I(b[3]), .ZN(n325) );
  VHSR_IN_2 U257 ( .I(a[6]), .ZN(n279) );
  VHSR_IN_2 U258 ( .I(a[7]), .ZN(n284) );
  VHSR_IN_2 U259 ( .I(b[2]), .ZN(n322) );
  VHSR_OAI22_2 U260 ( .A1(n325), .A2(n279), .B1(n284), .B2(n322), .ZN(n250) );
  VHSR_IN_2 U261 ( .I(b[1]), .ZN(n406) );
  VHSR_IN_2 U262 ( .I(a[4]), .ZN(n289) );
  VHSR_NOR2_1 U263 ( .A1(n322), .A2(n289), .ZN(n273) );
  VHSR_OAI21_2 U264 ( .A1(n406), .A2(n284), .B(n230), .ZN(n238) );
  VHSR_IN_2 U265 ( .I(a[5]), .ZN(n290) );
  VHSR_NOR4_2 U266 ( .A1(n273), .A2(n290), .A3(n241), .A4(n406), .ZN(n231) );
  VHSR_AOI31_2 U267 ( .A1(b[2]), .A2(a[6]), .A3(n238), .B(n231), .ZN(n239) );
  VHSR_IN_2 U268 ( .I(b[0]), .ZN(n404) );
  VHSR_NOR4_2 U269 ( .A1(n290), .A2(n289), .A3(n406), .A4(n404), .ZN(n278) );
  VHSR_NAND4_2 U270 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n247) );
  VHSR_NOR2_1 U271 ( .A1(n322), .A2(n290), .ZN(n232) );
  VHSR_AOI32_2 U272 ( .A1(b[3]), .A2(n247), .A3(a[4]), .B1(n232), .B2(n247), 
        .ZN(n233) );
  VHSR_IN_2 U273 ( .I(n233), .ZN(n234) );
  VHSR_OAI22_2 U274 ( .A1(n284), .A2(n404), .B1(n279), .B2(n406), .ZN(n235) );
  VHSR_MAOI222_2 U275 ( .A(n278), .B(n234), .C(n235), .ZN(n237) );
  VHSR_NOR2_1 U276 ( .A1(n279), .A2(n404), .ZN(n272) );
  VHSR_AOI211_2 U277 ( .A1(a[4]), .A2(b[0]), .B(n290), .C(n406), .ZN(n271) );
  VHSR_MAOI222_2 U278 ( .A(n273), .B(n272), .C(n271), .ZN(n270) );
  VHSR_OR2_2 U279 ( .A1(n278), .A2(n234), .Z(n236) );
  VHSR_OAI21_2 U280 ( .A1(n236), .A2(n235), .B(n237), .ZN(n262) );
  VHSR_NOR2_1 U281 ( .A1(n270), .A2(n262), .ZN(n261) );
  VHSR_AOI32_2 U282 ( .A1(b[2]), .A2(n239), .A3(a[6]), .B1(n238), .B2(n239), 
        .ZN(n259) );
  VHSR_NOR2_1 U283 ( .A1(n260), .A2(n259), .ZN(n258) );
  VHSR_CLKNAND2_2 U284 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_CLKNAND2_2 U285 ( .A1(n250), .A2(n246), .ZN(n240) );
  VHSR_AOI21_2 U286 ( .A1(n241), .A2(n240), .B(n300), .ZN(n306) );
  VHSR_OAI21_2 U287 ( .A1(n245), .A2(n243), .B(n244), .ZN(n242) );
  VHSR_OAI31_2 U288 ( .A1(n245), .A2(n244), .A3(n243), .B(n242), .ZN(n305) );
  VHSR_OAI21_2 U289 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U290 ( .A1(n250), .A2(n249), .ZN(n313) );
  VHSR_AD1_1 U291 ( .A(n253), .B(n252), .CI(n251), .CO(n244), .S(n312) );
  VHSR_NOR2_1 U292 ( .A1(n255), .A2(n254), .ZN(n257) );
  VHSR_AOI22_2 U293 ( .A1(n255), .A2(n254), .B1(n263), .B2(n257), .ZN(n256) );
  VHSR_OAI21_2 U294 ( .A1(n263), .A2(n257), .B(n256), .ZN(n318) );
  VHSR_AOI21_2 U295 ( .A1(n260), .A2(n259), .B(n258), .ZN(n317) );
  VHSR_AOI21_2 U296 ( .A1(n270), .A2(n262), .B(n261), .ZN(n333) );
  VHSR_OAI21_2 U297 ( .A1(n266), .A2(n264), .B(n263), .ZN(n265) );
  VHSR_IN_2 U298 ( .I(n265), .ZN(n332) );
  VHSR_AOI31_2 U299 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n340) );
  VHSR_OAI31_2 U300 ( .A1(n273), .A2(n272), .A3(n271), .B(n270), .ZN(n274) );
  VHSR_IN_2 U301 ( .I(n274), .ZN(n339) );
  VHSR_AOI22_2 U302 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n275) );
  VHSR_NOR2_1 U303 ( .A1(n276), .A2(n275), .ZN(n345) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[4]), .ZN(n292) );
  VHSR_IN_2 U305 ( .I(n292), .ZN(n374) );
  VHSR_NOR2_1 U306 ( .A1(n404), .A2(n405), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U307 ( .A1(n374), .A2(product[0]), .ZN(n397) );
  VHSR_IN_2 U308 ( .I(n397), .ZN(n344) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[4]), .A2(b[1]), .ZN(n277) );
  VHSR_OAI32_2 U310 ( .A1(n278), .A2(n404), .A3(n290), .B1(n277), .B2(n278), 
        .ZN(n343) );
  VHSR_NOR2_1 U311 ( .A1(n279), .A2(n288), .ZN(n392) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n310) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n310), .ZN(n281) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[6]), .ZN(n309) );
  VHSR_NAND3_2 U315 ( .A1(b[7]), .A2(a[5]), .A3(n309), .ZN(n280) );
  VHSR_CLKNAND2_2 U316 ( .A1(n281), .A2(n280), .ZN(n283) );
  VHSR_IN_2 U317 ( .I(n392), .ZN(n365) );
  VHSR_MAOI222_2 U318 ( .A(n365), .B(n281), .C(n280), .ZN(n349) );
  VHSR_IN_2 U319 ( .I(n349), .ZN(n282) );
  VHSR_OAI21_2 U320 ( .A1(n392), .A2(n283), .B(n282), .ZN(n298) );
  VHSR_NOR3_2 U321 ( .A1(n284), .A2(n310), .A3(n286), .ZN(n357) );
  VHSR_AOI22_2 U322 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n285) );
  VHSR_NOR2_1 U323 ( .A1(n357), .A2(n285), .ZN(n294) );
  VHSR_NOR3_2 U324 ( .A1(n290), .A2(n286), .A3(n292), .ZN(n315) );
  VHSR_NOR4_2 U325 ( .A1(n290), .A2(n289), .A3(n288), .A4(n287), .ZN(n355) );
  VHSR_AOI22_2 U326 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n291) );
  VHSR_NOR2_1 U327 ( .A1(n355), .A2(n291), .ZN(n293) );
  VHSR_NAND3_2 U328 ( .A1(b[5]), .A2(a[5]), .A3(n292), .ZN(n308) );
  VHSR_MAOI222_2 U329 ( .A(n310), .B(n309), .C(n308), .ZN(n307) );
  VHSR_AND2_2 U330 ( .A1(n303), .A2(n307), .Z(n302) );
  VHSR_AD1_1 U331 ( .A(n294), .B(n315), .CI(n293), .CO(n295), .S(n303) );
  VHSR_NOR2_1 U332 ( .A1(n302), .A2(n295), .ZN(n297) );
  VHSR_CLKNAND2_2 U333 ( .A1(n302), .A2(n295), .ZN(n296) );
  VHSR_NOR2_1 U334 ( .A1(n297), .A2(n298), .ZN(n350) );
  VHSR_AOI22_2 U335 ( .A1(n298), .A2(n297), .B1(n296), .B2(n350), .ZN(n390) );
  VHSR_AD1_1 U336 ( .A(n301), .B(n300), .CI(n299), .CO(n391), .S(n369) );
  VHSR_IAO21_2 U337 ( .A1(n303), .A2(n307), .B(n302), .ZN(n368) );
  VHSR_AD1_1 U338 ( .A(n306), .B(n305), .CI(n304), .CO(n299), .S(n388) );
  VHSR_AOI31_2 U339 ( .A1(n310), .A2(n309), .A3(n308), .B(n307), .ZN(n387) );
  VHSR_AD1_1 U340 ( .A(n313), .B(n312), .CI(n311), .CO(n304), .S(n372) );
  VHSR_AOI22_2 U341 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n314) );
  VHSR_NOR2_1 U342 ( .A1(n315), .A2(n314), .ZN(n371) );
  VHSR_AD1_1 U343 ( .A(n318), .B(n317), .CI(n316), .CO(n311), .S(n375) );
  VHSR_NOR4_2 U344 ( .A1(n325), .A2(n322), .A3(n403), .A4(n405), .ZN(n348) );
  VHSR_NOR2_1 U345 ( .A1(n322), .A2(n324), .ZN(n320) );
  VHSR_OAI21_2 U346 ( .A1(n325), .A2(n323), .B(n320), .ZN(n319) );
  VHSR_OAI31_2 U347 ( .A1(n325), .A2(n320), .A3(n323), .B(n319), .ZN(n347) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[2]), .A2(a[1]), .ZN(n321) );
  VHSR_OAI32_2 U349 ( .A1(n348), .A2(n405), .A3(n325), .B1(n321), .B2(n348), 
        .ZN(n379) );
  VHSR_AOI22_2 U350 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n409) );
  VHSR_NOR3_2 U351 ( .A1(n409), .A2(n405), .A3(n322), .ZN(n407) );
  VHSR_OAI22_2 U352 ( .A1(n406), .A2(n323), .B1(n404), .B2(n324), .ZN(n378) );
  VHSR_IN_2 U353 ( .I(n342), .ZN(n330) );
  VHSR_NOR2_1 U354 ( .A1(n406), .A2(n324), .ZN(n326) );
  VHSR_AOI211_2 U355 ( .A1(b[2]), .A2(a[0]), .B(n325), .C(n403), .ZN(n327) );
  VHSR_MAOI222_2 U356 ( .A(n326), .B(n337), .C(n327), .ZN(n329) );
  VHSR_OR2_2 U357 ( .A1(n337), .A2(n327), .Z(n328) );
  VHSR_AOI32_2 U358 ( .A1(a[3]), .A2(n329), .A3(b[1]), .B1(n328), .B2(n329), 
        .ZN(n341) );
  VHSR_OAI21_2 U359 ( .A1(n330), .A2(n341), .B(n329), .ZN(n346) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[3]), .A2(a[3]), .ZN(n335) );
  VHSR_IAO21_2 U361 ( .A1(n337), .A2(n336), .B(n335), .ZN(n385) );
  VHSR_AD1_1 U362 ( .A(n333), .B(n332), .CI(n331), .CO(n316), .S(n384) );
  VHSR_OAI21_2 U363 ( .A1(n337), .A2(n335), .B(n336), .ZN(n334) );
  VHSR_OAI31_2 U364 ( .A1(n337), .A2(n336), .A3(n335), .B(n334), .ZN(n382) );
  VHSR_AD1_1 U365 ( .A(n340), .B(n339), .CI(n338), .CO(n331), .S(n381) );
  VHSR_CLKNAND2_2 U366 ( .A1(a[4]), .A2(b[0]), .ZN(n399) );
  VHSR_CLKXOR2_2 U367 ( .A1(n342), .A2(n341), .Z(n402) );
  VHSR_AOI211_2 U368 ( .A1(n399), .A2(n398), .B(n344), .C(n402), .ZN(n400) );
  VHSR_AD1_1 U369 ( .A(n345), .B(n344), .CI(n343), .CO(n338), .S(n377) );
  VHSR_AD1_1 U370 ( .A(n348), .B(n347), .CI(n346), .CO(n336), .S(n376) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[6]), .A2(b[7]), .ZN(n352) );
  VHSR_AOI21_2 U372 ( .A1(a[7]), .A2(b[6]), .B(n352), .ZN(n351) );
  VHSR_AOI31_2 U373 ( .A1(a[7]), .A2(n352), .A3(b[6]), .B(n351), .ZN(n353) );
  VHSR_IN_2 U374 ( .I(n353), .ZN(n354) );
  VHSR_OR2_2 U375 ( .A1(n355), .A2(n354), .Z(n356) );
  VHSR_MAOI222_2 U376 ( .A(n357), .B(n355), .C(n354), .ZN(n364) );
  VHSR_OAI21_2 U377 ( .A1(n357), .A2(n356), .B(n364), .ZN(n361) );
  VHSR_CLKXOR2_2 U378 ( .A1(n362), .A2(n361), .Z(n358) );
  VHSR_CLKNAND2_2 U379 ( .A1(n359), .A2(n358), .ZN(n394) );
  VHSR_OAI21_2 U380 ( .A1(n359), .A2(n358), .B(n394), .ZN(n360) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[7]), .A2(b[7]), .ZN(n393) );
  VHSR_NOR2_1 U382 ( .A1(n362), .A2(n361), .ZN(n363) );
  VHSR_AND3_2 U383 ( .A1(n395), .A2(n365), .A3(n394), .Z(n366) );
  VHSR_NOR2_1 U384 ( .A1(n393), .A2(n366), .ZN(product[15]) );
  VHSR_AD1_1 U385 ( .A(n382), .B(n381), .CI(n380), .CO(n383), .S(product[6])
         );
  VHSR_AD1_1 U386 ( .A(n385), .B(n384), .CI(n383), .CO(n373), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U387 ( .A(n388), .B(n387), .CI(n386), .CO(n367), .S(product[10])
         );
  VHSR_AD1_1 U388 ( .A(n391), .B(n390), .CI(n389), .CO(n359), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U389 ( .A1(n393), .A2(n392), .ZN(n396) );
  VHSR_XOR3_2 U390 ( .A1(n396), .A2(n395), .A3(n394), .Z(product[14]) );
  VHSR_AOI21_2 U391 ( .A1(n402), .A2(n401), .B(n400), .ZN(product[4]) );
  VHSR_OAI22_2 U392 ( .A1(n406), .A2(n405), .B1(n404), .B2(n403), .ZN(
        product[1]) );
  VHSR_CLKNAND2_2 U393 ( .A1(b[2]), .A2(a[0]), .ZN(n408) );
  VHSR_AOI21_2 U394 ( .A1(n409), .A2(n408), .B(n407), .ZN(product[2]) );
endmodule



module mul8_110 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
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
         n404, n405, n406, n407;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U205 ( .A1(n324), .B1(n329), .ZN(n325) );
  VHSR_NOR2_1 U206 ( .A1(n348), .A2(n347), .ZN(n360) );
  VHSR_NOR2_1 U207 ( .A1(n397), .A2(n396), .ZN(n395) );
  VHSR_INAND3_2 U208 ( .A1(n375), .B1(b[5]), .B2(a[5]), .ZN(n306) );
  VHSR_NOR2_1 U209 ( .A1(n286), .A2(n280), .ZN(n375) );
  VHSR_IN_2 U210 ( .I(n358), .ZN(product[13]) );
  VHSR_INOR2_1 U211 ( .A1(n362), .B1(n361), .ZN(n393) );
  VHSR_NOR2_2 U212 ( .A1(n298), .A2(n297), .ZN(n296) );
  VHSR_NOR2_2 U213 ( .A1(n317), .A2(n319), .ZN(n336) );
  VHSR_AD1_1 U214 ( .A(n376), .B(n375), .CI(n374), .CO(n371), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(product[10])
         );
  VHSR_AD1_1 U216 ( .A(n380), .B(n405), .CI(n379), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n399), .B(n378), .CI(n377), .CO(n381), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n373), .B(n372), .CI(n371), .CO(n368), .S(product[9])
         );
  VHSR_AD1_1 U219 ( .A(n367), .B(n366), .CI(n365), .CO(n387), .S(product[11])
         );
  VHSR_IN_2 U220 ( .I(b[0]), .ZN(n403) );
  VHSR_IN_2 U221 ( .I(a[1]), .ZN(n321) );
  VHSR_NOR2_1 U222 ( .A1(n403), .A2(n321), .ZN(product[1]) );
  VHSR_AOI22_2 U223 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n250) );
  VHSR_IN_2 U224 ( .I(b[3]), .ZN(n322) );
  VHSR_IN_2 U225 ( .I(b[2]), .ZN(n317) );
  VHSR_IN_2 U226 ( .I(a[5]), .ZN(n287) );
  VHSR_IN_2 U227 ( .I(a[4]), .ZN(n286) );
  VHSR_NOR4_2 U228 ( .A1(n322), .A2(n317), .A3(n287), .A4(n286), .ZN(n248) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n282) );
  VHSR_IN_2 U230 ( .I(b[1]), .ZN(n404) );
  VHSR_NOR2_1 U231 ( .A1(n282), .A2(n404), .ZN(n216) );
  VHSR_AOI211_2 U232 ( .A1(b[2]), .A2(a[4]), .B(n322), .C(n287), .ZN(n217) );
  VHSR_CLKNAND2_2 U233 ( .A1(a[6]), .A2(b[2]), .ZN(n219) );
  VHSR_IN_2 U234 ( .I(n219), .ZN(n215) );
  VHSR_MAOI222_2 U235 ( .A(n216), .B(n217), .C(n215), .ZN(n229) );
  VHSR_AOI21_2 U236 ( .A1(b[1]), .A2(a[7]), .B(n217), .ZN(n220) );
  VHSR_IN_2 U237 ( .I(n229), .ZN(n218) );
  VHSR_AOI21_2 U238 ( .A1(n220), .A2(n219), .B(n218), .ZN(n257) );
  VHSR_CLKNAND2_2 U239 ( .A1(a[6]), .A2(b[1]), .ZN(n226) );
  VHSR_IN_2 U240 ( .I(n226), .ZN(n223) );
  VHSR_NOR4_2 U241 ( .A1(n287), .A2(n286), .A3(n404), .A4(n403), .ZN(n275) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[2]), .A2(a[5]), .ZN(n222) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[3]), .A2(a[4]), .ZN(n221) );
  VHSR_AOI21_2 U244 ( .A1(n222), .A2(n221), .B(n248), .ZN(n224) );
  VHSR_MAOI222_2 U245 ( .A(n223), .B(n275), .C(n224), .ZN(n228) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[2]), .A2(a[4]), .ZN(n271) );
  VHSR_OAI21_2 U247 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n270) );
  VHSR_CLKNAND2_2 U248 ( .A1(a[4]), .A2(b[0]), .ZN(n397) );
  VHSR_NAND3_2 U249 ( .A1(b[1]), .A2(a[5]), .A3(n397), .ZN(n269) );
  VHSR_MAOI222_2 U250 ( .A(n271), .B(n270), .C(n269), .ZN(n268) );
  VHSR_NOR2_1 U251 ( .A1(n275), .A2(n224), .ZN(n227) );
  VHSR_IN_2 U252 ( .I(n228), .ZN(n225) );
  VHSR_AOI21_2 U253 ( .A1(n227), .A2(n226), .B(n225), .ZN(n260) );
  VHSR_CLKNAND2_2 U254 ( .A1(n268), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U255 ( .A1(n228), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U256 ( .A1(n257), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U257 ( .A1(n229), .A2(n255), .ZN(n247) );
  VHSR_NOR2_1 U258 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_NOR2_1 U259 ( .A1(n250), .A2(n246), .ZN(n239) );
  VHSR_AND3_2 U260 ( .A1(n239), .A2(b[3]), .A3(a[7]), .Z(n301) );
  VHSR_IN_2 U261 ( .I(b[7]), .ZN(n284) );
  VHSR_IN_2 U262 ( .I(a[3]), .ZN(n320) );
  VHSR_IN_2 U263 ( .I(b[6]), .ZN(n285) );
  VHSR_IN_2 U264 ( .I(a[2]), .ZN(n319) );
  VHSR_OAI22_2 U265 ( .A1(n285), .A2(n320), .B1(n284), .B2(n319), .ZN(n245) );
  VHSR_AOI22_2 U266 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n236) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[4]), .A2(a[2]), .ZN(n267) );
  VHSR_NAND3_2 U268 ( .A1(a[3]), .A2(b[5]), .A3(n267), .ZN(n235) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[7]), .A2(a[2]), .ZN(n230) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[1]), .ZN(n232) );
  VHSR_OAI22_2 U271 ( .A1(n236), .A2(n235), .B1(n230), .B2(n232), .ZN(n237) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[4]), .A2(a[0]), .ZN(n396) );
  VHSR_NAND3_2 U273 ( .A1(a[1]), .A2(b[5]), .A3(n396), .ZN(n266) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[6]), .A2(a[0]), .ZN(n265) );
  VHSR_MAOI222_2 U275 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NAND4_2 U276 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n242) );
  VHSR_IN_2 U277 ( .I(b[4]), .ZN(n280) );
  VHSR_IN_2 U278 ( .I(b[5]), .ZN(n281) );
  VHSR_OAI22_2 U279 ( .A1(n280), .A2(n320), .B1(n281), .B2(n319), .ZN(n231) );
  VHSR_AND2_2 U280 ( .A1(n242), .A2(n231), .Z(n234) );
  VHSR_IN_2 U281 ( .I(a[0]), .ZN(n402) );
  VHSR_OAI21_2 U282 ( .A1(n284), .A2(n402), .B(n232), .ZN(n233) );
  VHSR_NOR4_2 U283 ( .A1(n280), .A2(n281), .A3(n321), .A4(n402), .ZN(n273) );
  VHSR_AND2_2 U284 ( .A1(n264), .A2(n263), .Z(n262) );
  VHSR_AD1_1 U285 ( .A(n234), .B(n233), .CI(n273), .CO(n251), .S(n263) );
  VHSR_AOI21_2 U286 ( .A1(n236), .A2(n235), .B(n237), .ZN(n254) );
  VHSR_OAI32_2 U287 ( .A1(n237), .A2(n262), .A3(n251), .B1(n254), .B2(n237), 
        .ZN(n243) );
  VHSR_CLKNAND2_2 U288 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U289 ( .A1(n245), .A2(n241), .ZN(n240) );
  VHSR_NOR3_2 U290 ( .A1(n284), .A2(n320), .A3(n240), .ZN(n300) );
  VHSR_NOR2_1 U291 ( .A1(n322), .A2(n282), .ZN(n238) );
  VHSR_IAO21_2 U292 ( .A1(n239), .A2(n238), .B(n301), .ZN(n304) );
  VHSR_OAI32_2 U293 ( .A1(n300), .A2(n320), .A3(n284), .B1(n240), .B2(n300), 
        .ZN(n303) );
  VHSR_OAI21_2 U294 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U295 ( .A1(n245), .A2(n244), .ZN(n311) );
  VHSR_AOI21_2 U296 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U297 ( .A1(n250), .A2(n249), .ZN(n310) );
  VHSR_NOR2_1 U298 ( .A1(n262), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U299 ( .A1(n262), .A2(n251), .B1(n254), .B2(n253), .ZN(n252) );
  VHSR_OAI21_2 U300 ( .A1(n254), .A2(n253), .B(n252), .ZN(n316) );
  VHSR_OAI21_2 U301 ( .A1(n257), .A2(n256), .B(n255), .ZN(n258) );
  VHSR_IN_2 U302 ( .I(n258), .ZN(n315) );
  VHSR_OAI21_2 U303 ( .A1(n268), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U304 ( .I(n261), .ZN(n332) );
  VHSR_IAO21_2 U305 ( .A1(n264), .A2(n263), .B(n262), .ZN(n331) );
  VHSR_AOI31_2 U306 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n339) );
  VHSR_AOI31_2 U307 ( .A1(n271), .A2(n270), .A3(n269), .B(n268), .ZN(n338) );
  VHSR_CLKNAND2_2 U308 ( .A1(b[5]), .A2(a[0]), .ZN(n272) );
  VHSR_OAI32_2 U309 ( .A1(n273), .A2(n321), .A3(n280), .B1(n272), .B2(n273), 
        .ZN(n343) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[5]), .A2(b[0]), .ZN(n274) );
  VHSR_OAI32_2 U311 ( .A1(n275), .A2(n404), .A3(n286), .B1(n274), .B2(n275), 
        .ZN(n342) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[6]), .ZN(n363) );
  VHSR_IN_2 U313 ( .I(n363), .ZN(n390) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[6]), .A2(b[4]), .ZN(n308) );
  VHSR_NAND3_2 U315 ( .A1(a[7]), .A2(b[5]), .A3(n308), .ZN(n277) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[4]), .A2(b[6]), .ZN(n307) );
  VHSR_NAND3_2 U317 ( .A1(b[7]), .A2(a[5]), .A3(n307), .ZN(n276) );
  VHSR_CLKNAND2_2 U318 ( .A1(n277), .A2(n276), .ZN(n279) );
  VHSR_MAOI222_2 U319 ( .A(n363), .B(n277), .C(n276), .ZN(n347) );
  VHSR_IN_2 U320 ( .I(n347), .ZN(n278) );
  VHSR_OAI21_2 U321 ( .A1(n390), .A2(n279), .B(n278), .ZN(n295) );
  VHSR_AND3_2 U322 ( .A1(n375), .A2(a[5]), .A3(b[5]), .Z(n312) );
  VHSR_NOR3_2 U323 ( .A1(n282), .A2(n308), .A3(n281), .ZN(n355) );
  VHSR_AOI22_2 U324 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n283) );
  VHSR_NOR2_1 U325 ( .A1(n355), .A2(n283), .ZN(n291) );
  VHSR_NOR4_2 U326 ( .A1(n287), .A2(n286), .A3(n285), .A4(n284), .ZN(n353) );
  VHSR_AOI22_2 U327 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n288) );
  VHSR_NOR2_1 U328 ( .A1(n353), .A2(n288), .ZN(n290) );
  VHSR_IN_2 U329 ( .I(n289), .ZN(n298) );
  VHSR_MAOI222_2 U330 ( .A(n308), .B(n307), .C(n306), .ZN(n305) );
  VHSR_IN_2 U331 ( .I(n305), .ZN(n297) );
  VHSR_AD1_1 U332 ( .A(n312), .B(n291), .CI(n290), .CO(n292), .S(n289) );
  VHSR_NOR2_1 U333 ( .A1(n296), .A2(n292), .ZN(n294) );
  VHSR_CLKNAND2_2 U334 ( .A1(n296), .A2(n292), .ZN(n293) );
  VHSR_NOR2_1 U335 ( .A1(n294), .A2(n295), .ZN(n348) );
  VHSR_AOI22_2 U336 ( .A1(n295), .A2(n294), .B1(n293), .B2(n348), .ZN(n388) );
  VHSR_AOI21_2 U337 ( .A1(n298), .A2(n297), .B(n296), .ZN(n367) );
  VHSR_AD1_1 U338 ( .A(n301), .B(n300), .CI(n299), .CO(n389), .S(n366) );
  VHSR_AD1_1 U339 ( .A(n304), .B(n303), .CI(n302), .CO(n299), .S(n370) );
  VHSR_AOI31_2 U340 ( .A1(n308), .A2(n307), .A3(n306), .B(n305), .ZN(n369) );
  VHSR_AD1_1 U341 ( .A(n311), .B(n310), .CI(n309), .CO(n302), .S(n373) );
  VHSR_AOI22_2 U342 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n313) );
  VHSR_NOR2_1 U343 ( .A1(n313), .A2(n312), .ZN(n372) );
  VHSR_AD1_1 U344 ( .A(n316), .B(n315), .CI(n314), .CO(n309), .S(n376) );
  VHSR_NOR4_2 U345 ( .A1(n322), .A2(n317), .A3(n321), .A4(n402), .ZN(n346) );
  VHSR_CLKNAND2_2 U346 ( .A1(b[2]), .A2(a[1]), .ZN(n318) );
  VHSR_OAI32_2 U347 ( .A1(n346), .A2(n402), .A3(n322), .B1(n318), .B2(n346), 
        .ZN(n380) );
  VHSR_AOI22_2 U348 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n407) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[2]), .A2(a[0]), .ZN(n406) );
  VHSR_NOR2_1 U350 ( .A1(n407), .A2(n406), .ZN(n405) );
  VHSR_OAI22_2 U351 ( .A1(n404), .A2(n319), .B1(n403), .B2(n320), .ZN(n379) );
  VHSR_IN_2 U352 ( .I(n341), .ZN(n327) );
  VHSR_NOR2_1 U353 ( .A1(n404), .A2(n320), .ZN(n323) );
  VHSR_AOI211_2 U354 ( .A1(b[2]), .A2(a[0]), .B(n322), .C(n321), .ZN(n324) );
  VHSR_MAOI222_2 U355 ( .A(n323), .B(n324), .C(n336), .ZN(n326) );
  VHSR_IN_2 U356 ( .I(n336), .ZN(n329) );
  VHSR_AOI32_2 U357 ( .A1(a[3]), .A2(n326), .A3(b[1]), .B1(n325), .B2(n326), 
        .ZN(n340) );
  VHSR_OAI21_2 U358 ( .A1(n327), .A2(n340), .B(n326), .ZN(n345) );
  VHSR_CLKNAND2_2 U359 ( .A1(b[3]), .A2(a[3]), .ZN(n334) );
  VHSR_AOI22_2 U360 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n328) );
  VHSR_IAO21_2 U361 ( .A1(n334), .A2(n329), .B(n328), .ZN(n344) );
  VHSR_IAO21_2 U362 ( .A1(n336), .A2(n335), .B(n334), .ZN(n386) );
  VHSR_AD1_1 U363 ( .A(n332), .B(n331), .CI(n330), .CO(n314), .S(n385) );
  VHSR_OAI21_2 U364 ( .A1(n336), .A2(n334), .B(n335), .ZN(n333) );
  VHSR_OAI31_2 U365 ( .A1(n336), .A2(n335), .A3(n334), .B(n333), .ZN(n383) );
  VHSR_AD1_1 U366 ( .A(n339), .B(n338), .CI(n337), .CO(n330), .S(n382) );
  VHSR_CLKXOR2_2 U367 ( .A1(n341), .A2(n340), .Z(n401) );
  VHSR_AOI211_2 U368 ( .A1(n397), .A2(n396), .B(n395), .C(n401), .ZN(n399) );
  VHSR_AD1_1 U369 ( .A(n343), .B(n395), .CI(n342), .CO(n337), .S(n378) );
  VHSR_AD1_1 U370 ( .A(n346), .B(n345), .CI(n344), .CO(n335), .S(n377) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[7]), .A2(b[6]), .ZN(n350) );
  VHSR_AOI21_2 U372 ( .A1(a[6]), .A2(b[7]), .B(n350), .ZN(n349) );
  VHSR_AOI31_2 U373 ( .A1(a[6]), .A2(n350), .A3(b[7]), .B(n349), .ZN(n351) );
  VHSR_IN_2 U374 ( .I(n351), .ZN(n352) );
  VHSR_OR2_2 U375 ( .A1(n353), .A2(n352), .Z(n354) );
  VHSR_MAOI222_2 U376 ( .A(n355), .B(n353), .C(n352), .ZN(n362) );
  VHSR_OAI21_2 U377 ( .A1(n355), .A2(n354), .B(n362), .ZN(n359) );
  VHSR_CLKXOR2_2 U378 ( .A1(n360), .A2(n359), .Z(n356) );
  VHSR_CLKNAND2_2 U379 ( .A1(n357), .A2(n356), .ZN(n392) );
  VHSR_OAI21_2 U380 ( .A1(n357), .A2(n356), .B(n392), .ZN(n358) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[7]), .A2(b[7]), .ZN(n391) );
  VHSR_NOR2_1 U382 ( .A1(n360), .A2(n359), .ZN(n361) );
  VHSR_AND3_2 U383 ( .A1(n393), .A2(n363), .A3(n392), .Z(n364) );
  VHSR_NOR2_1 U384 ( .A1(n391), .A2(n364), .ZN(product[15]) );
  VHSR_AD1_1 U385 ( .A(n383), .B(n382), .CI(n381), .CO(n384), .S(product[6])
         );
  VHSR_AD1_1 U386 ( .A(n386), .B(n385), .CI(n384), .CO(n374), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U387 ( .A(n389), .B(n388), .CI(n387), .CO(n357), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U388 ( .A1(n391), .A2(n390), .ZN(n394) );
  VHSR_XOR3_2 U389 ( .A1(n394), .A2(n393), .A3(n392), .Z(product[14]) );
  VHSR_AOI21_2 U390 ( .A1(n397), .A2(n396), .B(n395), .ZN(n398) );
  VHSR_IN_2 U391 ( .I(n398), .ZN(n400) );
  VHSR_AOI21_2 U392 ( .A1(n401), .A2(n400), .B(n399), .ZN(product[4]) );
  VHSR_AOI21_2 U393 ( .A1(n404), .A2(n403), .B(n402), .ZN(product[0]) );
  VHSR_AOI21_2 U394 ( .A1(n407), .A2(n406), .B(n405), .ZN(product[2]) );
endmodule


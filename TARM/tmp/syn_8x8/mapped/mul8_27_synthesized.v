
module mul8_27 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n243, n244,
         n245, n246, n247, n248, n249, n250, n251, n252, n253, n254, n255,
         n256, n257, n258, n259, n260, n261, n262, n263, n264, n265, n266,
         n267, n268, n269, n270, n271, n272, n273, n274, n275, n276, n277,
         n278, n279, n280, n281, n282, n283, n284, n285, n286, n287, n288,
         n289, n290, n291, n292, n293, n294, n295, n296, n297, n298, n299,
         n300, n301, n302, n303, n304, n305, n306, n307, n308, n309, n310,
         n311, n312, n313, n314, n315, n316, n317, n318, n319, n320, n321,
         n322, n323, n324, n325, n326, n327, n328, n329, n330, n331, n332,
         n333, n334, n335, n336, n337, n338, n339, n340, n341, n342, n343,
         n344, n345, n346, n347, n348, n349, n350, n351, n352, n353, n354,
         n355, n356, n357, n358, n359, n360, n361, n362, n363, n364, n365,
         n366, n367, n368, n369, n370, n371, n372, n373, n374, n375, n376,
         n377, n378, n379, n380, n381, n382, n383, n384, n385, n386, n387,
         n388, n389, n390, n391, n392, n393, n394, n395, n396, n397, n398,
         n399, n400, n401, n402, n403, n404, n405, n406, n407, n408, n409,
         n410, n411, n412, n413, n414, n415, n416, n417, n418, n419, n420,
         n421, n422, n423, n424, n425, n426, n427, n428, n429, n430, n431,
         n432, n433, n434, n435, n436, n437, n438, n439, n440, n441, n442,
         n443, n444, n445, n446, n447;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U234 ( .A1(n255), .B1(n243), .ZN(n246) );
  VHSR_INOR2_2 U235 ( .A1(n363), .B1(n377), .ZN(n366) );
  VHSR_INOR2_2 U236 ( .A1(n394), .B1(n325), .ZN(n326) );
  VHSR_INAND3_2 U237 ( .A1(n408), .B1(a[5]), .B2(b[5]), .ZN(n341) );
  VHSR_INOR2_2 U238 ( .A1(n402), .B1(n401), .ZN(n404) );
  VHSR_NOR2_1 U239 ( .A1(n439), .A2(n438), .ZN(n437) );
  VHSR_NOR2_1 U240 ( .A1(n345), .A2(n380), .ZN(n408) );
  VHSR_IN_2 U241 ( .I(n400), .ZN(product[13]) );
  VHSR_AD1_1 U242 ( .A(n416), .B(n415), .CI(n444), .CO(n379), .S(product[3])
         );
  VHSR_AD1_1 U243 ( .A(n437), .B(n414), .CI(n413), .CO(n417), .S(product[5])
         );
  VHSR_AD1_1 U244 ( .A(n412), .B(n411), .CI(n410), .CO(n407), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U245 ( .A(n409), .B(n408), .CI(n407), .CO(n420), .S(
        \intadd_0/SUM[3] ) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[6]), .A2(a[2]), .ZN(n281) );
  VHSR_CLKNAND2_2 U247 ( .A1(b[6]), .A2(a[0]), .ZN(n309) );
  VHSR_NAND3_2 U248 ( .A1(b[7]), .A2(a[1]), .A3(n309), .ZN(n247) );
  VHSR_CLKNAND2_2 U249 ( .A1(b[4]), .A2(a[2]), .ZN(n308) );
  VHSR_NAND3_2 U250 ( .A1(a[3]), .A2(b[5]), .A3(n308), .ZN(n249) );
  VHSR_MAOI222_2 U251 ( .A(n281), .B(n247), .C(n249), .ZN(n251) );
  VHSR_IN_2 U252 ( .I(b[4]), .ZN(n345) );
  VHSR_IN_2 U253 ( .I(a[0]), .ZN(n443) );
  VHSR_OAI211_2 U254 ( .A1(n345), .A2(n443), .B(b[5]), .C(a[1]), .ZN(n307) );
  VHSR_MAOI222_2 U255 ( .A(n309), .B(n308), .C(n307), .ZN(n306) );
  VHSR_NAND4_2 U256 ( .A1(b[4]), .A2(b[5]), .A3(a[3]), .A4(a[2]), .ZN(n255) );
  VHSR_AOI22_2 U257 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n243) );
  VHSR_IN_2 U258 ( .I(b[5]), .ZN(n324) );
  VHSR_IN_2 U259 ( .I(a[1]), .ZN(n441) );
  VHSR_NOR4_2 U260 ( .A1(n345), .A2(n324), .A3(n443), .A4(n441), .ZN(n317) );
  VHSR_IN_2 U261 ( .I(b[7]), .ZN(n277) );
  VHSR_NOR3_2 U262 ( .A1(n277), .A2(n309), .A3(n441), .ZN(n259) );
  VHSR_AOI22_2 U263 ( .A1(b[6]), .A2(a[1]), .B1(b[7]), .B2(a[0]), .ZN(n244) );
  VHSR_NOR2_1 U264 ( .A1(n259), .A2(n244), .ZN(n245) );
  VHSR_AND2_2 U265 ( .A1(n306), .A2(n302), .Z(n301) );
  VHSR_AD1_1 U266 ( .A(n246), .B(n317), .CI(n245), .CO(n296), .S(n302) );
  VHSR_NOR2_1 U267 ( .A1(n301), .A2(n296), .ZN(n299) );
  VHSR_AND2_2 U268 ( .A1(n281), .A2(n247), .Z(n248) );
  VHSR_AOI21_2 U269 ( .A1(n249), .A2(n248), .B(n251), .ZN(n250) );
  VHSR_IN_2 U270 ( .I(n250), .ZN(n300) );
  VHSR_NOR2_1 U271 ( .A1(n299), .A2(n300), .ZN(n297) );
  VHSR_NOR2_1 U272 ( .A1(n251), .A2(n297), .ZN(n288) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[7]), .A2(a[2]), .ZN(n253) );
  VHSR_AOI21_2 U274 ( .A1(b[6]), .A2(a[3]), .B(n253), .ZN(n252) );
  VHSR_AOI31_2 U275 ( .A1(b[6]), .A2(n253), .A3(a[3]), .B(n252), .ZN(n254) );
  VHSR_CLKNAND2_2 U276 ( .A1(n255), .A2(n254), .ZN(n258) );
  VHSR_IN_2 U277 ( .I(n259), .ZN(n256) );
  VHSR_MAOI222_2 U278 ( .A(n256), .B(n255), .C(n254), .ZN(n260) );
  VHSR_IN_2 U279 ( .I(n260), .ZN(n257) );
  VHSR_OAI21_2 U280 ( .A1(n259), .A2(n258), .B(n257), .ZN(n287) );
  VHSR_NOR2_1 U281 ( .A1(n288), .A2(n287), .ZN(n286) );
  VHSR_NOR2_1 U282 ( .A1(n286), .A2(n260), .ZN(n276) );
  VHSR_IN_2 U283 ( .I(a[3]), .ZN(n370) );
  VHSR_AOI211_2 U284 ( .A1(n276), .A2(n281), .B(n370), .C(n277), .ZN(n336) );
  VHSR_CLKNAND2_2 U285 ( .A1(a[6]), .A2(b[2]), .ZN(n275) );
  VHSR_IN_2 U286 ( .I(n275), .ZN(n285) );
  VHSR_IN_2 U287 ( .I(a[4]), .ZN(n380) );
  VHSR_IN_2 U288 ( .I(a[5]), .ZN(n346) );
  VHSR_IN_2 U289 ( .I(b[3]), .ZN(n359) );
  VHSR_IN_2 U290 ( .I(b[2]), .ZN(n358) );
  VHSR_NOR4_2 U291 ( .A1(n380), .A2(n346), .A3(n359), .A4(n358), .ZN(n291) );
  VHSR_AOI211_2 U292 ( .A1(a[4]), .A2(b[2]), .B(n346), .C(n359), .ZN(n262) );
  VHSR_IN_2 U293 ( .I(a[7]), .ZN(n322) );
  VHSR_IN_2 U294 ( .I(b[1]), .ZN(n442) );
  VHSR_NOR2_1 U295 ( .A1(n322), .A2(n442), .ZN(n261) );
  VHSR_MAOI222_2 U296 ( .A(n262), .B(n261), .C(n285), .ZN(n273) );
  VHSR_AOI21_2 U297 ( .A1(a[7]), .A2(b[1]), .B(n262), .ZN(n264) );
  VHSR_IN_2 U298 ( .I(n273), .ZN(n263) );
  VHSR_AOI21_2 U299 ( .A1(n264), .A2(n275), .B(n263), .ZN(n294) );
  VHSR_IN_2 U300 ( .I(b[0]), .ZN(n440) );
  VHSR_NOR4_2 U301 ( .A1(n380), .A2(n346), .A3(n442), .A4(n440), .ZN(n315) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[5]), .A2(b[2]), .ZN(n266) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[3]), .ZN(n265) );
  VHSR_AOI21_2 U304 ( .A1(n266), .A2(n265), .B(n291), .ZN(n268) );
  VHSR_AOI22_2 U305 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n270) );
  VHSR_IN_2 U306 ( .I(n270), .ZN(n267) );
  VHSR_MAOI222_2 U307 ( .A(n315), .B(n268), .C(n267), .ZN(n272) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[2]), .ZN(n313) );
  VHSR_OAI211_2 U309 ( .A1(n380), .A2(n440), .B(a[5]), .C(b[1]), .ZN(n312) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[0]), .ZN(n311) );
  VHSR_MAOI222_2 U311 ( .A(n313), .B(n312), .C(n311), .ZN(n310) );
  VHSR_NOR2_1 U312 ( .A1(n315), .A2(n268), .ZN(n271) );
  VHSR_IN_2 U313 ( .I(n272), .ZN(n269) );
  VHSR_AOI21_2 U314 ( .A1(n271), .A2(n270), .B(n269), .ZN(n304) );
  VHSR_CLKNAND2_2 U315 ( .A1(n310), .A2(n304), .ZN(n303) );
  VHSR_CLKNAND2_2 U316 ( .A1(n272), .A2(n303), .ZN(n293) );
  VHSR_CLKNAND2_2 U317 ( .A1(n294), .A2(n293), .ZN(n292) );
  VHSR_CLKNAND2_2 U318 ( .A1(n273), .A2(n292), .ZN(n290) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[7]), .A2(b[3]), .ZN(n283) );
  VHSR_AOI22_2 U320 ( .A1(a[6]), .A2(b[3]), .B1(a[7]), .B2(b[2]), .ZN(n274) );
  VHSR_IAO21_2 U321 ( .A1(n283), .A2(n275), .B(n274), .ZN(n289) );
  VHSR_IAO21_2 U322 ( .A1(n285), .A2(n284), .B(n283), .ZN(n335) );
  VHSR_IN_2 U323 ( .I(n276), .ZN(n280) );
  VHSR_NOR2_1 U324 ( .A1(n277), .A2(n370), .ZN(n279) );
  VHSR_AOI21_2 U325 ( .A1(n281), .A2(n279), .B(n280), .ZN(n278) );
  VHSR_AOI31_2 U326 ( .A1(n281), .A2(n280), .A3(n279), .B(n278), .ZN(n339) );
  VHSR_OAI21_2 U327 ( .A1(n285), .A2(n283), .B(n284), .ZN(n282) );
  VHSR_OAI31_2 U328 ( .A1(n285), .A2(n284), .A3(n283), .B(n282), .ZN(n338) );
  VHSR_AOI21_2 U329 ( .A1(n288), .A2(n287), .B(n286), .ZN(n350) );
  VHSR_AD1_1 U330 ( .A(n291), .B(n290), .CI(n289), .CO(n284), .S(n349) );
  VHSR_OAI21_2 U331 ( .A1(n294), .A2(n293), .B(n292), .ZN(n295) );
  VHSR_IN_2 U332 ( .I(n295), .ZN(n353) );
  VHSR_CLKNAND2_2 U333 ( .A1(n301), .A2(n296), .ZN(n298) );
  VHSR_AOI22_2 U334 ( .A1(n300), .A2(n299), .B1(n298), .B2(n297), .ZN(n352) );
  VHSR_IAO21_2 U335 ( .A1(n306), .A2(n302), .B(n301), .ZN(n356) );
  VHSR_OAI21_2 U336 ( .A1(n310), .A2(n304), .B(n303), .ZN(n305) );
  VHSR_IN_2 U337 ( .I(n305), .ZN(n355) );
  VHSR_AOI31_2 U338 ( .A1(n309), .A2(n308), .A3(n307), .B(n306), .ZN(n373) );
  VHSR_AOI31_2 U339 ( .A1(n313), .A2(n312), .A3(n311), .B(n310), .ZN(n372) );
  VHSR_NOR2_1 U340 ( .A1(n443), .A2(n440), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U341 ( .A1(n408), .A2(product[0]), .ZN(n382) );
  VHSR_IN_2 U342 ( .I(n382), .ZN(n388) );
  VHSR_CLKNAND2_2 U343 ( .A1(a[5]), .A2(b[0]), .ZN(n314) );
  VHSR_OAI32_2 U344 ( .A1(n315), .A2(n442), .A3(n380), .B1(n314), .B2(n315), 
        .ZN(n387) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[5]), .A2(a[0]), .ZN(n316) );
  VHSR_OAI32_2 U346 ( .A1(n317), .A2(n441), .A3(n345), .B1(n316), .B2(n317), 
        .ZN(n386) );
  VHSR_CLKNAND2_2 U347 ( .A1(a[6]), .A2(b[6]), .ZN(n405) );
  VHSR_IN_2 U348 ( .I(n405), .ZN(n432) );
  VHSR_CLKNAND2_2 U349 ( .A1(a[6]), .A2(b[4]), .ZN(n343) );
  VHSR_NAND3_2 U350 ( .A1(a[7]), .A2(b[5]), .A3(n343), .ZN(n319) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[6]), .A2(a[4]), .ZN(n342) );
  VHSR_NAND3_2 U352 ( .A1(b[7]), .A2(a[5]), .A3(n342), .ZN(n318) );
  VHSR_CLKNAND2_2 U353 ( .A1(n319), .A2(n318), .ZN(n321) );
  VHSR_MAOI222_2 U354 ( .A(n405), .B(n319), .C(n318), .ZN(n389) );
  VHSR_IN_2 U355 ( .I(n389), .ZN(n320) );
  VHSR_OAI21_2 U356 ( .A1(n432), .A2(n321), .B(n320), .ZN(n331) );
  VHSR_NOR3_2 U357 ( .A1(n322), .A2(n343), .A3(n324), .ZN(n397) );
  VHSR_AOI22_2 U358 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n323) );
  VHSR_NOR2_1 U359 ( .A1(n397), .A2(n323), .ZN(n327) );
  VHSR_NOR4_2 U360 ( .A1(n345), .A2(n324), .A3(n380), .A4(n346), .ZN(n347) );
  VHSR_NAND4_2 U361 ( .A1(b[6]), .A2(b[7]), .A3(a[4]), .A4(a[5]), .ZN(n394) );
  VHSR_AOI22_2 U362 ( .A1(b[6]), .A2(a[5]), .B1(b[7]), .B2(a[4]), .ZN(n325) );
  VHSR_MAOI222_2 U363 ( .A(n343), .B(n342), .C(n341), .ZN(n340) );
  VHSR_AND2_2 U364 ( .A1(n333), .A2(n340), .Z(n332) );
  VHSR_AD1_1 U365 ( .A(n327), .B(n347), .CI(n326), .CO(n328), .S(n333) );
  VHSR_NOR2_1 U366 ( .A1(n332), .A2(n328), .ZN(n330) );
  VHSR_CLKNAND2_2 U367 ( .A1(n332), .A2(n328), .ZN(n329) );
  VHSR_NOR2_1 U368 ( .A1(n330), .A2(n331), .ZN(n390) );
  VHSR_AOI22_2 U369 ( .A1(n331), .A2(n330), .B1(n329), .B2(n390), .ZN(n430) );
  VHSR_IAO21_2 U370 ( .A1(n333), .A2(n340), .B(n332), .ZN(n428) );
  VHSR_AD1_1 U371 ( .A(n336), .B(n335), .CI(n334), .CO(n431), .S(n427) );
  VHSR_AD1_1 U372 ( .A(n339), .B(n338), .CI(n337), .CO(n334), .S(n425) );
  VHSR_AOI31_2 U373 ( .A1(n343), .A2(n342), .A3(n341), .B(n340), .ZN(n424) );
  VHSR_CLKNAND2_2 U374 ( .A1(b[5]), .A2(a[4]), .ZN(n344) );
  VHSR_OAI32_2 U375 ( .A1(n347), .A2(n346), .A3(n345), .B1(n344), .B2(n347), 
        .ZN(n422) );
  VHSR_AD1_1 U376 ( .A(n350), .B(n349), .CI(n348), .CO(n337), .S(n421) );
  VHSR_AD1_1 U377 ( .A(n353), .B(n352), .CI(n351), .CO(n348), .S(n409) );
  VHSR_AD1_1 U378 ( .A(n356), .B(n355), .CI(n354), .CO(n351), .S(n412) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[2]), .A2(b[2]), .ZN(n364) );
  VHSR_IN_2 U380 ( .I(n364), .ZN(n377) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[3]), .A2(b[3]), .ZN(n375) );
  VHSR_AOI22_2 U382 ( .A1(a[3]), .A2(b[2]), .B1(a[2]), .B2(b[3]), .ZN(n357) );
  VHSR_IAO21_2 U383 ( .A1(n375), .A2(n364), .B(n357), .ZN(n385) );
  VHSR_CLKNAND2_2 U384 ( .A1(a[0]), .A2(b[2]), .ZN(n446) );
  VHSR_OR3_2 U385 ( .A1(n446), .A2(n441), .A3(n359), .Z(n369) );
  VHSR_OAI22_2 U386 ( .A1(n443), .A2(n359), .B1(n441), .B2(n358), .ZN(n360) );
  VHSR_AND2_2 U387 ( .A1(n369), .A2(n360), .Z(n416) );
  VHSR_CLKNAND2_2 U388 ( .A1(a[2]), .A2(b[0]), .ZN(n447) );
  VHSR_NOR3_2 U389 ( .A1(n370), .A2(n442), .A3(n447), .ZN(n362) );
  VHSR_AOI22_2 U390 ( .A1(a[3]), .A2(b[0]), .B1(a[2]), .B2(b[1]), .ZN(n361) );
  VHSR_NOR2_1 U391 ( .A1(n362), .A2(n361), .ZN(n415) );
  VHSR_CLKNAND2_2 U392 ( .A1(a[1]), .A2(b[1]), .ZN(n445) );
  VHSR_MAOI222_2 U393 ( .A(n447), .B(n446), .C(n445), .ZN(n444) );
  VHSR_NAND3_2 U394 ( .A1(n446), .A2(b[3]), .A3(a[1]), .ZN(n363) );
  VHSR_NAND3_2 U395 ( .A1(b[1]), .A2(a[3]), .A3(n447), .ZN(n365) );
  VHSR_MAOI222_2 U396 ( .A(n365), .B(n364), .C(n363), .ZN(n367) );
  VHSR_AOI21_2 U397 ( .A1(n366), .A2(n365), .B(n367), .ZN(n378) );
  VHSR_AOI21_2 U398 ( .A1(n379), .A2(n378), .B(n367), .ZN(n368) );
  VHSR_IN_2 U399 ( .I(n368), .ZN(n384) );
  VHSR_OAI31_2 U400 ( .A1(n442), .A2(n370), .A3(n447), .B(n369), .ZN(n383) );
  VHSR_IAO21_2 U401 ( .A1(n377), .A2(n376), .B(n375), .ZN(n411) );
  VHSR_AD1_1 U402 ( .A(n373), .B(n372), .CI(n371), .CO(n354), .S(n419) );
  VHSR_OAI21_2 U403 ( .A1(n377), .A2(n375), .B(n376), .ZN(n374) );
  VHSR_OAI31_2 U404 ( .A1(n377), .A2(n376), .A3(n375), .B(n374), .ZN(n418) );
  VHSR_XNOR2_2 U405 ( .A1(n379), .A2(n378), .ZN(n439) );
  VHSR_NOR2_1 U406 ( .A1(n380), .A2(n440), .ZN(n381) );
  VHSR_AOI32_2 U407 ( .A1(a[0]), .A2(n382), .A3(b[4]), .B1(n381), .B2(n382), 
        .ZN(n438) );
  VHSR_AD1_1 U408 ( .A(n385), .B(n384), .CI(n383), .CO(n376), .S(n414) );
  VHSR_AD1_1 U409 ( .A(n388), .B(n387), .CI(n386), .CO(n371), .S(n413) );
  VHSR_NOR2_1 U410 ( .A1(n390), .A2(n389), .ZN(n401) );
  VHSR_CLKNAND2_2 U411 ( .A1(b[6]), .A2(a[7]), .ZN(n392) );
  VHSR_AOI21_2 U412 ( .A1(a[6]), .A2(b[7]), .B(n392), .ZN(n391) );
  VHSR_AOI31_2 U413 ( .A1(a[6]), .A2(n392), .A3(b[7]), .B(n391), .ZN(n393) );
  VHSR_CLKNAND2_2 U414 ( .A1(n394), .A2(n393), .ZN(n396) );
  VHSR_IN_2 U415 ( .I(n397), .ZN(n395) );
  VHSR_MAOI222_2 U416 ( .A(n395), .B(n394), .C(n393), .ZN(n403) );
  VHSR_IAO21_2 U417 ( .A1(n397), .A2(n396), .B(n403), .ZN(n402) );
  VHSR_XNOR2_2 U418 ( .A1(n401), .A2(n402), .ZN(n398) );
  VHSR_CLKNAND2_2 U419 ( .A1(n399), .A2(n398), .ZN(n434) );
  VHSR_OAI21_2 U420 ( .A1(n399), .A2(n398), .B(n434), .ZN(n400) );
  VHSR_CLKNAND2_2 U421 ( .A1(a[7]), .A2(b[7]), .ZN(n433) );
  VHSR_NOR2_1 U422 ( .A1(n404), .A2(n403), .ZN(n435) );
  VHSR_AND3_2 U423 ( .A1(n435), .A2(n405), .A3(n434), .Z(n406) );
  VHSR_NOR2_1 U424 ( .A1(n433), .A2(n406), .ZN(product[15]) );
  VHSR_AD1_1 U425 ( .A(n419), .B(n418), .CI(n417), .CO(n410), .S(product[6])
         );
  VHSR_AD1_1 U426 ( .A(n422), .B(n421), .CI(n420), .CO(n423), .S(product[9])
         );
  VHSR_AD1_1 U427 ( .A(n425), .B(n424), .CI(n423), .CO(n426), .S(product[10])
         );
  VHSR_AD1_1 U428 ( .A(n428), .B(n427), .CI(n426), .CO(n429), .S(product[11])
         );
  VHSR_AD1_1 U429 ( .A(n431), .B(n430), .CI(n429), .CO(n399), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U430 ( .A1(n433), .A2(n432), .ZN(n436) );
  VHSR_XOR3_2 U431 ( .A1(n436), .A2(n435), .A3(n434), .Z(product[14]) );
  VHSR_AOI21_2 U432 ( .A1(n439), .A2(n438), .B(n437), .ZN(product[4]) );
  VHSR_OAI22_2 U433 ( .A1(n443), .A2(n442), .B1(n441), .B2(n440), .ZN(
        product[1]) );
  VHSR_AOI31_2 U434 ( .A1(n447), .A2(n446), .A3(n445), .B(n444), .ZN(
        product[2]) );
endmodule


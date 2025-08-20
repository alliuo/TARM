
module mul8_134 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n256, n257,
         n258, n259, n260, n261, n262, n263, n264, n265, n266, n267, n268,
         n269, n270, n271, n272, n273, n274, n275, n276, n277, n278, n279,
         n280, n281, n282, n283, n284, n285, n286, n287, n288, n289, n290,
         n291, n292, n293, n294, n295, n296, n297, n298, n299, n300, n301,
         n302, n303, n304, n305, n306, n307, n308, n309, n310, n311, n312,
         n313, n314, n315, n316, n317, n318, n319, n320, n321, n322, n323,
         n324, n325, n326, n327, n328, n329, n330, n331, n332, n333, n334,
         n335, n336, n337, n338, n339, n340, n341, n342, n343, n344, n345,
         n346, n347, n348, n349, n350, n351, n352, n353, n354, n355, n356,
         n357, n358, n359, n360, n361, n362, n363, n364, n365, n366, n367,
         n368, n369, n370, n371, n372, n373, n374, n375, n376, n377, n378,
         n379, n380, n381, n382, n383, n384, n385, n386, n387, n388, n389,
         n390, n391, n392, n393, n394, n395, n396, n397, n398, n399, n400,
         n401, n402, n403, n404, n405, n406, n407, n408, n409, n410, n411,
         n412, n413, n414, n415, n416, n417, n418, n419, n420, n421, n422,
         n423, n424, n425, n426, n427, n428, n429, n430, n431, n432, n433,
         n434, n435, n436, n437, n438, n439, n440, n441, n442, n443, n444,
         n445, n446, n447, n448, n449, n450, n451, n452, n453, n454, n455,
         n456, n457, n458, n459, n460, n461, n462, n463, n464, n465, n466,
         n467, n468, n469, n470, n471, n472, n473, n474, n475, n476, n477,
         n478, n479;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR3_2 U250 ( .A1(n352), .B1(n332), .B2(n466), .ZN(n328) );
  VHSR_NOR2_1 U251 ( .A1(n304), .A2(n305), .ZN(n302) );
  VHSR_NOR2_1 U252 ( .A1(n264), .A2(n302), .ZN(n297) );
  VHSR_IN_2 U253 ( .I(n293), .ZN(n273) );
  VHSR_NOR2_1 U254 ( .A1(n410), .A2(n409), .ZN(n413) );
  VHSR_NOR2_1 U255 ( .A1(n379), .A2(n462), .ZN(n472) );
  VHSR_INAND3_2 U256 ( .A1(n441), .B1(a[5]), .B2(b[5]), .ZN(n351) );
  VHSR_NOR2_1 U257 ( .A1(n343), .A2(n344), .ZN(n415) );
  VHSR_INAND3_2 U258 ( .A1(product[0]), .B1(a[1]), .B2(b[1]), .ZN(n477) );
  VHSR_NOR2_1 U259 ( .A1(n364), .A2(n359), .ZN(n441) );
  VHSR_CLKN_1 U260 ( .I(n426), .ZN(product[13]) );
  VHSR_NAND2_2 U261 ( .A1(n425), .A2(n424), .ZN(n467) );
  VHSR_AD1_2 U262 ( .A(n455), .B(n454), .CI(n453), .CO(n425), .S(
        \intadd_0/SUM[7] ) );
  VHSR_AD1_2 U263 ( .A(n452), .B(n451), .CI(n450), .CO(n453), .S(product[11])
         );
  VHSR_AD1_2 U264 ( .A(n349), .B(n348), .CI(n347), .CO(n455), .S(n451) );
  VHSR_AD1_2 U265 ( .A(n355), .B(n354), .CI(n353), .CO(n347), .S(n434) );
  VHSR_NOR2_2 U266 ( .A1(n406), .A2(n394), .ZN(n401) );
  VHSR_NOR2_2 U267 ( .A1(n408), .A2(n407), .ZN(n406) );
  VHSR_INOR2_1 U268 ( .A1(n430), .B1(n429), .ZN(n468) );
  VHSR_NOR2_2 U269 ( .A1(n472), .A2(n471), .ZN(n470) );
  VHSR_INAND2_1 U270 ( .A1(n295), .B1(n272), .ZN(n293) );
  VHSR_NOR2_2 U271 ( .A1(n297), .A2(n296), .ZN(n295) );
  VHSR_INOR2_1 U272 ( .A1(n416), .B1(n415), .ZN(n428) );
  VHSR_NOR2_2 U273 ( .A1(n464), .A2(n463), .ZN(n462) );
  VHSR_NOR2_2 U274 ( .A1(n345), .A2(n341), .ZN(n343) );
  VHSR_NOR2_2 U275 ( .A1(n308), .A2(n301), .ZN(n304) );
  VHSR_MOAI22_1 U276 ( .A1(n352), .A2(n351), .B1(n337), .B2(n336), .ZN(n350)
         );
  VHSR_INOR3_1 U277 ( .A1(n337), .B1(n360), .B2(n330), .ZN(n423) );
  VHSR_INOR2_1 U278 ( .A1(n441), .B1(n332), .ZN(n339) );
  VHSR_NOR2_2 U279 ( .A1(n325), .A2(n334), .ZN(n465) );
  VHSR_NOR2_2 U280 ( .A1(n364), .A2(n325), .ZN(n337) );
  VHSR_AD1_1 U281 ( .A(n446), .B(n473), .CI(n445), .CO(n442), .S(product[5])
         );
  VHSR_AD1_1 U282 ( .A(n438), .B(n437), .CI(n436), .CO(n433), .S(product[9])
         );
  VHSR_AD1_1 U283 ( .A(n444), .B(n443), .CI(n442), .CO(n447), .S(product[6])
         );
  VHSR_AD1_1 U284 ( .A(n441), .B(n440), .CI(n439), .CO(n436), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U285 ( .A(n435), .B(n434), .CI(n433), .CO(n450), .S(product[10])
         );
  VHSR_CLKNAND2_2 U286 ( .A1(a[2]), .A2(b[6]), .ZN(n294) );
  VHSR_CLKNAND2_2 U287 ( .A1(a[0]), .A2(b[6]), .ZN(n316) );
  VHSR_NAND3_2 U288 ( .A1(b[7]), .A2(a[1]), .A3(n316), .ZN(n262) );
  VHSR_CLKNAND2_2 U289 ( .A1(a[2]), .A2(b[4]), .ZN(n315) );
  VHSR_NAND3_2 U290 ( .A1(b[5]), .A2(a[3]), .A3(n315), .ZN(n260) );
  VHSR_MAOI222_2 U291 ( .A(n294), .B(n262), .C(n260), .ZN(n264) );
  VHSR_CLKNAND2_2 U292 ( .A1(a[0]), .A2(b[4]), .ZN(n410) );
  VHSR_NAND3_2 U293 ( .A1(b[5]), .A2(a[1]), .A3(n410), .ZN(n314) );
  VHSR_MAOI222_2 U294 ( .A(n316), .B(n315), .C(n314), .ZN(n313) );
  VHSR_IN_2 U295 ( .I(a[1]), .ZN(n456) );
  VHSR_IN_2 U296 ( .I(b[5]), .ZN(n360) );
  VHSR_NOR3_2 U297 ( .A1(n456), .A2(n360), .A3(n410), .ZN(n321) );
  VHSR_IN_2 U298 ( .I(a[3]), .ZN(n395) );
  VHSR_NOR3_2 U299 ( .A1(n395), .A2(n360), .A3(n315), .ZN(n269) );
  VHSR_AOI22_2 U300 ( .A1(a[3]), .A2(b[4]), .B1(a[2]), .B2(b[5]), .ZN(n256) );
  VHSR_NOR2_1 U301 ( .A1(n269), .A2(n256), .ZN(n259) );
  VHSR_IN_2 U302 ( .I(b[7]), .ZN(n333) );
  VHSR_NOR3_2 U303 ( .A1(n456), .A2(n333), .A3(n316), .ZN(n271) );
  VHSR_AOI22_2 U304 ( .A1(a[0]), .A2(b[7]), .B1(a[1]), .B2(b[6]), .ZN(n257) );
  VHSR_NOR2_1 U305 ( .A1(n271), .A2(n257), .ZN(n258) );
  VHSR_AND2_2 U306 ( .A1(n313), .A2(n309), .Z(n308) );
  VHSR_AD1_1 U307 ( .A(n321), .B(n259), .CI(n258), .CO(n301), .S(n309) );
  VHSR_AND2_2 U308 ( .A1(n294), .A2(n260), .Z(n261) );
  VHSR_AOI21_2 U309 ( .A1(n262), .A2(n261), .B(n264), .ZN(n263) );
  VHSR_IN_2 U310 ( .I(n263), .ZN(n305) );
  VHSR_IN_2 U311 ( .I(b[6]), .ZN(n334) );
  VHSR_CLKNAND2_2 U312 ( .A1(b[7]), .A2(a[2]), .ZN(n266) );
  VHSR_OAI21_2 U313 ( .A1(n334), .A2(n395), .B(n266), .ZN(n265) );
  VHSR_OAI31_2 U314 ( .A1(n334), .A2(n266), .A3(n395), .B(n265), .ZN(n267) );
  VHSR_IN_2 U315 ( .I(n267), .ZN(n268) );
  VHSR_OR2_2 U316 ( .A1(n269), .A2(n268), .Z(n270) );
  VHSR_MAOI222_2 U317 ( .A(n271), .B(n269), .C(n268), .ZN(n272) );
  VHSR_OAI21_2 U318 ( .A1(n271), .A2(n270), .B(n272), .ZN(n296) );
  VHSR_AOI211_2 U319 ( .A1(n273), .A2(n294), .B(n333), .C(n395), .ZN(n349) );
  VHSR_CLKNAND2_2 U320 ( .A1(b[2]), .A2(a[6]), .ZN(n276) );
  VHSR_IN_2 U321 ( .I(n276), .ZN(n290) );
  VHSR_IN_2 U322 ( .I(b[3]), .ZN(n396) );
  VHSR_IN_2 U323 ( .I(a[5]), .ZN(n362) );
  VHSR_CLKNAND2_2 U324 ( .A1(b[2]), .A2(a[4]), .ZN(n320) );
  VHSR_NOR3_2 U325 ( .A1(n396), .A2(n362), .A3(n320), .ZN(n300) );
  VHSR_CLKNAND2_2 U326 ( .A1(b[3]), .A2(a[7]), .ZN(n288) );
  VHSR_AOI22_2 U327 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n274) );
  VHSR_IAO21_2 U328 ( .A1(n288), .A2(n276), .B(n274), .ZN(n299) );
  VHSR_CLKNAND2_2 U329 ( .A1(b[1]), .A2(a[7]), .ZN(n277) );
  VHSR_NAND3_2 U330 ( .A1(a[5]), .A2(b[3]), .A3(n320), .ZN(n275) );
  VHSR_MAOI222_2 U331 ( .A(n277), .B(n276), .C(n275), .ZN(n285) );
  VHSR_IN_2 U332 ( .I(a[7]), .ZN(n330) );
  VHSR_IN_2 U333 ( .I(b[1]), .ZN(n461) );
  VHSR_AOI31_2 U334 ( .A1(a[5]), .A2(b[3]), .A3(n320), .B(n290), .ZN(n278) );
  VHSR_OAI32_2 U335 ( .A1(n285), .A2(n330), .A3(n461), .B1(n278), .B2(n285), 
        .ZN(n307) );
  VHSR_CLKNAND2_2 U336 ( .A1(b[0]), .A2(a[4]), .ZN(n409) );
  VHSR_NOR3_2 U337 ( .A1(n461), .A2(n362), .A3(n409), .ZN(n324) );
  VHSR_AOI22_2 U338 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n279) );
  VHSR_NOR2_1 U339 ( .A1(n279), .A2(n300), .ZN(n280) );
  VHSR_IN_2 U340 ( .I(a[6]), .ZN(n325) );
  VHSR_IN_2 U341 ( .I(b[0]), .ZN(n457) );
  VHSR_OAI22_2 U342 ( .A1(n461), .A2(n325), .B1(n457), .B2(n330), .ZN(n281) );
  VHSR_MAOI222_2 U343 ( .A(n324), .B(n280), .C(n281), .ZN(n284) );
  VHSR_NAND3_2 U344 ( .A1(a[5]), .A2(b[1]), .A3(n409), .ZN(n319) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[0]), .A2(a[6]), .ZN(n318) );
  VHSR_MAOI222_2 U346 ( .A(n320), .B(n319), .C(n318), .ZN(n317) );
  VHSR_OR2_2 U347 ( .A1(n324), .A2(n280), .Z(n282) );
  VHSR_OAI21_2 U348 ( .A1(n282), .A2(n281), .B(n284), .ZN(n283) );
  VHSR_IN_2 U349 ( .I(n283), .ZN(n311) );
  VHSR_CLKNAND2_2 U350 ( .A1(n317), .A2(n311), .ZN(n310) );
  VHSR_CLKNAND2_2 U351 ( .A1(n284), .A2(n310), .ZN(n306) );
  VHSR_AOI21_2 U352 ( .A1(n307), .A2(n306), .B(n285), .ZN(n286) );
  VHSR_IN_2 U353 ( .I(n286), .ZN(n298) );
  VHSR_IAO21_2 U354 ( .A1(n290), .A2(n289), .B(n288), .ZN(n348) );
  VHSR_OAI21_2 U355 ( .A1(n290), .A2(n288), .B(n289), .ZN(n287) );
  VHSR_OAI31_2 U356 ( .A1(n290), .A2(n289), .A3(n288), .B(n287), .ZN(n355) );
  VHSR_NOR2_1 U357 ( .A1(n395), .A2(n333), .ZN(n292) );
  VHSR_AOI21_2 U358 ( .A1(n294), .A2(n292), .B(n293), .ZN(n291) );
  VHSR_AOI31_2 U359 ( .A1(n294), .A2(n293), .A3(n292), .B(n291), .ZN(n354) );
  VHSR_AOI21_2 U360 ( .A1(n297), .A2(n296), .B(n295), .ZN(n358) );
  VHSR_AD1_1 U361 ( .A(n300), .B(n299), .CI(n298), .CO(n289), .S(n357) );
  VHSR_CLKNAND2_2 U362 ( .A1(n308), .A2(n301), .ZN(n303) );
  VHSR_AOI22_2 U363 ( .A1(n305), .A2(n304), .B1(n303), .B2(n302), .ZN(n367) );
  VHSR_CLKXOR2_2 U364 ( .A1(n307), .A2(n306), .Z(n366) );
  VHSR_IAO21_2 U365 ( .A1(n313), .A2(n309), .B(n308), .ZN(n370) );
  VHSR_OAI21_2 U366 ( .A1(n317), .A2(n311), .B(n310), .ZN(n312) );
  VHSR_IN_2 U367 ( .I(n312), .ZN(n369) );
  VHSR_AOI31_2 U368 ( .A1(n316), .A2(n315), .A3(n314), .B(n313), .ZN(n400) );
  VHSR_AOI31_2 U369 ( .A1(n320), .A2(n319), .A3(n318), .B(n317), .ZN(n399) );
  VHSR_AOI22_2 U370 ( .A1(a[0]), .A2(b[5]), .B1(a[1]), .B2(b[4]), .ZN(n322) );
  VHSR_NOR2_1 U371 ( .A1(n322), .A2(n321), .ZN(n414) );
  VHSR_AOI22_2 U372 ( .A1(b[1]), .A2(a[4]), .B1(b[0]), .B2(a[5]), .ZN(n323) );
  VHSR_NOR2_1 U373 ( .A1(n324), .A2(n323), .ZN(n412) );
  VHSR_IN_2 U374 ( .I(a[4]), .ZN(n359) );
  VHSR_NOR2_1 U375 ( .A1(n359), .A2(n334), .ZN(n336) );
  VHSR_CLKNAND2_2 U376 ( .A1(a[5]), .A2(b[7]), .ZN(n327) );
  VHSR_IN_2 U377 ( .I(b[4]), .ZN(n364) );
  VHSR_CLKNAND2_2 U378 ( .A1(b[5]), .A2(a[7]), .ZN(n326) );
  VHSR_OAI22_2 U379 ( .A1(n336), .A2(n327), .B1(n337), .B2(n326), .ZN(n329) );
  VHSR_AOI22_2 U380 ( .A1(b[4]), .A2(a[6]), .B1(a[4]), .B2(b[6]), .ZN(n352) );
  VHSR_CLKNAND2_2 U381 ( .A1(b[5]), .A2(a[5]), .ZN(n332) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[7]), .A2(b[7]), .ZN(n466) );
  VHSR_AOI31_2 U383 ( .A1(b[6]), .A2(a[6]), .A3(n329), .B(n328), .ZN(n416) );
  VHSR_OAI21_2 U384 ( .A1(n465), .A2(n329), .B(n416), .ZN(n344) );
  VHSR_AOI22_2 U385 ( .A1(b[4]), .A2(a[7]), .B1(a[6]), .B2(b[5]), .ZN(n331) );
  VHSR_NOR2_1 U386 ( .A1(n423), .A2(n331), .ZN(n340) );
  VHSR_NOR4_2 U387 ( .A1(n359), .A2(n334), .A3(n362), .A4(n333), .ZN(n421) );
  VHSR_AOI22_2 U388 ( .A1(a[4]), .A2(b[7]), .B1(b[6]), .B2(a[5]), .ZN(n335) );
  VHSR_NOR2_1 U389 ( .A1(n421), .A2(n335), .ZN(n338) );
  VHSR_AND2_2 U390 ( .A1(n346), .A2(n350), .Z(n345) );
  VHSR_AD1_1 U391 ( .A(n340), .B(n339), .CI(n338), .CO(n341), .S(n346) );
  VHSR_CLKNAND2_2 U392 ( .A1(n345), .A2(n341), .ZN(n342) );
  VHSR_AOI22_2 U393 ( .A1(n344), .A2(n343), .B1(n342), .B2(n415), .ZN(n454) );
  VHSR_IAO21_2 U394 ( .A1(n346), .A2(n350), .B(n345), .ZN(n452) );
  VHSR_AOI21_2 U395 ( .A1(n352), .A2(n351), .B(n350), .ZN(n435) );
  VHSR_AD1_1 U396 ( .A(n358), .B(n357), .CI(n356), .CO(n353), .S(n438) );
  VHSR_NOR2_1 U397 ( .A1(n360), .A2(n359), .ZN(n363) );
  VHSR_OAI21_2 U398 ( .A1(n364), .A2(n362), .B(n363), .ZN(n361) );
  VHSR_OAI31_2 U399 ( .A1(n364), .A2(n363), .A3(n362), .B(n361), .ZN(n437) );
  VHSR_AD1_1 U400 ( .A(n367), .B(n366), .CI(n365), .CO(n356), .S(n440) );
  VHSR_AD1_1 U401 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(n449) );
  VHSR_NAND4_2 U402 ( .A1(a[3]), .A2(a[2]), .A3(b[1]), .A4(b[0]), .ZN(n392) );
  VHSR_AND2_2 U403 ( .A1(a[2]), .A2(b[1]), .Z(n371) );
  VHSR_AOI32_2 U404 ( .A1(a[3]), .A2(n392), .A3(b[0]), .B1(n371), .B2(n392), 
        .ZN(n375) );
  VHSR_IN_2 U405 ( .I(a[0]), .ZN(n459) );
  VHSR_NOR2_1 U406 ( .A1(n457), .A2(n459), .ZN(product[0]) );
  VHSR_NAND3_2 U407 ( .A1(b[1]), .A2(a[1]), .A3(product[0]), .ZN(n374) );
  VHSR_AOI22_2 U408 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n372) );
  VHSR_NAND4_2 U409 ( .A1(b[3]), .A2(b[2]), .A3(a[0]), .A4(a[1]), .ZN(n391) );
  VHSR_IN_2 U410 ( .I(n391), .ZN(n387) );
  VHSR_NOR2_1 U411 ( .A1(n372), .A2(n387), .ZN(n378) );
  VHSR_IN_2 U412 ( .I(n378), .ZN(n373) );
  VHSR_MAOI222_2 U413 ( .A(n375), .B(n374), .C(n373), .ZN(n379) );
  VHSR_CLKNAND2_2 U414 ( .A1(b[2]), .A2(a[0]), .ZN(n479) );
  VHSR_CLKNAND2_2 U415 ( .A1(a[2]), .A2(b[0]), .ZN(n478) );
  VHSR_MAOI222_2 U416 ( .A(n479), .B(n478), .C(n477), .ZN(n476) );
  VHSR_IN_2 U417 ( .I(n476), .ZN(n464) );
  VHSR_CLKNAND2_2 U418 ( .A1(n375), .A2(n374), .ZN(n377) );
  VHSR_IN_2 U419 ( .I(n379), .ZN(n376) );
  VHSR_OAI21_2 U420 ( .A1(n378), .A2(n377), .B(n376), .ZN(n463) );
  VHSR_CLKNAND2_2 U421 ( .A1(a[2]), .A2(b[2]), .ZN(n397) );
  VHSR_IN_2 U422 ( .I(n397), .ZN(n404) );
  VHSR_NAND3_2 U423 ( .A1(b[3]), .A2(a[1]), .A3(n479), .ZN(n381) );
  VHSR_NAND3_2 U424 ( .A1(b[1]), .A2(a[3]), .A3(n478), .ZN(n380) );
  VHSR_CLKNAND2_2 U425 ( .A1(n381), .A2(n380), .ZN(n383) );
  VHSR_MAOI222_2 U426 ( .A(n397), .B(n381), .C(n380), .ZN(n384) );
  VHSR_IN_2 U427 ( .I(n384), .ZN(n382) );
  VHSR_OAI21_2 U428 ( .A1(n404), .A2(n383), .B(n382), .ZN(n471) );
  VHSR_NOR2_1 U429 ( .A1(n470), .A2(n384), .ZN(n408) );
  VHSR_CLKNAND2_2 U430 ( .A1(b[3]), .A2(a[2]), .ZN(n386) );
  VHSR_AOI21_2 U431 ( .A1(a[3]), .A2(b[2]), .B(n386), .ZN(n385) );
  VHSR_AOI31_2 U432 ( .A1(a[3]), .A2(n386), .A3(b[2]), .B(n385), .ZN(n393) );
  VHSR_IN_2 U433 ( .I(n392), .ZN(n388) );
  VHSR_NOR2_1 U434 ( .A1(n388), .A2(n387), .ZN(n390) );
  VHSR_AOI22_2 U435 ( .A1(n388), .A2(n387), .B1(n393), .B2(n390), .ZN(n389) );
  VHSR_OAI21_2 U436 ( .A1(n393), .A2(n390), .B(n389), .ZN(n407) );
  VHSR_MAOI222_2 U437 ( .A(n393), .B(n392), .C(n391), .ZN(n394) );
  VHSR_AOI211_2 U438 ( .A1(n401), .A2(n397), .B(n396), .C(n395), .ZN(n448) );
  VHSR_AD1_1 U439 ( .A(n400), .B(n399), .CI(n398), .CO(n368), .S(n444) );
  VHSR_CLKNAND2_2 U440 ( .A1(a[3]), .A2(b[3]), .ZN(n405) );
  VHSR_IN_2 U441 ( .I(n401), .ZN(n403) );
  VHSR_OAI21_2 U442 ( .A1(n405), .A2(n404), .B(n403), .ZN(n402) );
  VHSR_OAI31_2 U443 ( .A1(n405), .A2(n404), .A3(n403), .B(n402), .ZN(n443) );
  VHSR_AOI21_2 U444 ( .A1(n408), .A2(n407), .B(n406), .ZN(n446) );
  VHSR_AOI21_2 U445 ( .A1(n410), .A2(n409), .B(n413), .ZN(n474) );
  VHSR_IN_2 U446 ( .I(n474), .ZN(n411) );
  VHSR_AOI211_2 U447 ( .A1(n472), .A2(n471), .B(n470), .C(n411), .ZN(n473) );
  VHSR_AD1_1 U448 ( .A(n414), .B(n413), .CI(n412), .CO(n398), .S(n445) );
  VHSR_CLKNAND2_2 U449 ( .A1(a[7]), .A2(b[6]), .ZN(n418) );
  VHSR_AOI21_2 U450 ( .A1(a[6]), .A2(b[7]), .B(n418), .ZN(n417) );
  VHSR_AOI31_2 U451 ( .A1(a[6]), .A2(n418), .A3(b[7]), .B(n417), .ZN(n419) );
  VHSR_IN_2 U452 ( .I(n419), .ZN(n420) );
  VHSR_OR2_2 U453 ( .A1(n421), .A2(n420), .Z(n422) );
  VHSR_MAOI222_2 U454 ( .A(n423), .B(n421), .C(n420), .ZN(n430) );
  VHSR_OAI21_2 U455 ( .A1(n423), .A2(n422), .B(n430), .ZN(n427) );
  VHSR_CLKXOR2_2 U456 ( .A1(n428), .A2(n427), .Z(n424) );
  VHSR_OAI21_2 U457 ( .A1(n425), .A2(n424), .B(n467), .ZN(n426) );
  VHSR_IN_2 U458 ( .I(n465), .ZN(n431) );
  VHSR_NOR2_1 U459 ( .A1(n428), .A2(n427), .ZN(n429) );
  VHSR_AND3_2 U460 ( .A1(n431), .A2(n468), .A3(n467), .Z(n432) );
  VHSR_NOR2_1 U461 ( .A1(n466), .A2(n432), .ZN(product[15]) );
  VHSR_AD1_1 U462 ( .A(n449), .B(n448), .CI(n447), .CO(n439), .S(
        \intadd_0/SUM[2] ) );
  VHSR_NOR2_1 U463 ( .A1(n457), .A2(n456), .ZN(n460) );
  VHSR_OAI21_2 U464 ( .A1(n461), .A2(n459), .B(n460), .ZN(n458) );
  VHSR_OAI31_2 U465 ( .A1(n461), .A2(n460), .A3(n459), .B(n458), .ZN(
        product[1]) );
  VHSR_AOI21_2 U466 ( .A1(n464), .A2(n463), .B(n462), .ZN(product[3]) );
  VHSR_NOR2_1 U467 ( .A1(n466), .A2(n465), .ZN(n469) );
  VHSR_XOR3_2 U468 ( .A1(n469), .A2(n468), .A3(n467), .Z(product[14]) );
  VHSR_AOI21_2 U469 ( .A1(n472), .A2(n471), .B(n470), .ZN(n475) );
  VHSR_IAO21_2 U470 ( .A1(n475), .A2(n474), .B(n473), .ZN(product[4]) );
  VHSR_AOI31_2 U471 ( .A1(n479), .A2(n478), .A3(n477), .B(n476), .ZN(
        product[2]) );
endmodule


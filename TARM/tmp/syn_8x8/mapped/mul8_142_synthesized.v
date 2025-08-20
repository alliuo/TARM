
module mul8_142 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n258, n259,
         n260, n261, n262, n263, n264, n265, n266, n267, n268, n269, n270,
         n271, n272, n273, n274, n275, n276, n277, n278, n279, n280, n281,
         n282, n283, n284, n285, n286, n287, n288, n289, n290, n291, n292,
         n293, n294, n295, n296, n297, n298, n299, n300, n301, n302, n303,
         n304, n305, n306, n307, n308, n309, n310, n311, n312, n313, n314,
         n315, n316, n317, n318, n319, n320, n321, n322, n323, n324, n325,
         n326, n327, n328, n329, n330, n331, n332, n333, n334, n335, n336,
         n337, n338, n339, n340, n341, n342, n343, n344, n345, n346, n347,
         n348, n349, n350, n351, n352, n353, n354, n355, n356, n357, n358,
         n359, n360, n361, n362, n363, n364, n365, n366, n367, n368, n369,
         n370, n371, n372, n373, n374, n375, n376, n377, n378, n379, n380,
         n381, n382, n383, n384, n385, n386, n387, n388, n389, n390, n391,
         n392, n393, n394, n395, n396, n397, n398, n399, n400, n401, n402,
         n403, n404, n405, n406, n407, n408, n409, n410, n411, n412, n413,
         n414, n415, n416, n417, n418, n419, n420, n421, n422, n423, n424,
         n425, n426, n427, n428, n429, n430, n431, n432, n433, n434, n435,
         n436, n437, n438, n439, n440, n441, n442, n443, n444, n445, n446,
         n447, n448, n449, n450, n451, n452, n453, n454, n455, n456, n457,
         n458, n459, n460, n461, n462, n463, n464, n465, n466, n467, n468,
         n469, n470, n471, n472, n473, n474, n475, n476, n477, n478, n479,
         n480;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U250 ( .A1(n284), .B1(n283), .ZN(n285) );
  VHSR_INOR2_2 U251 ( .A1(n270), .B1(n258), .ZN(n261) );
  VHSR_INOR3_2 U252 ( .A1(n357), .B1(n339), .B2(n466), .ZN(n335) );
  VHSR_NOR2_1 U253 ( .A1(n307), .A2(n306), .ZN(n305) );
  VHSR_INOR2_2 U254 ( .A1(n421), .B1(n340), .ZN(n343) );
  VHSR_NOR2_1 U255 ( .A1(n424), .A2(n338), .ZN(n345) );
  VHSR_NOR2_1 U256 ( .A1(n410), .A2(n409), .ZN(n408) );
  VHSR_INOR2_2 U257 ( .A1(n385), .B1(n412), .ZN(n410) );
  VHSR_INOR2_2 U258 ( .A1(n391), .B1(n408), .ZN(n398) );
  VHSR_NOR2_1 U259 ( .A1(n348), .A2(n349), .ZN(n416) );
  VHSR_INAND3_2 U260 ( .A1(product[0]), .B1(b[1]), .B2(a[1]), .ZN(n471) );
  VHSR_NOR2_1 U261 ( .A1(n369), .A2(n364), .ZN(n439) );
  VHSR_CLKN_1 U262 ( .I(n427), .ZN(product[13]) );
  VHSR_AD1_2 U263 ( .A(n456), .B(n455), .CI(n454), .CO(n426), .S(
        \intadd_0/SUM[7] ) );
  VHSR_AD1_2 U264 ( .A(n354), .B(n353), .CI(n352), .CO(n456), .S(n452) );
  VHSR_NOR2_2 U265 ( .A1(n431), .A2(n430), .ZN(n468) );
  VHSR_INOR2_1 U266 ( .A1(n429), .B1(n428), .ZN(n431) );
  VHSR_NOR2_2 U267 ( .A1(n305), .A2(n275), .ZN(n296) );
  VHSR_INOR2_1 U268 ( .A1(n417), .B1(n416), .ZN(n428) );
  VHSR_MOAI22_1 U269 ( .A1(n415), .A2(n414), .B1(n413), .B2(n412), .ZN(n479)
         );
  VHSR_NOR2_2 U270 ( .A1(n350), .A2(n346), .ZN(n348) );
  VHSR_INAND3_1 U271 ( .A1(n439), .B1(a[5]), .B2(b[5]), .ZN(n356) );
  VHSR_INOR2_1 U272 ( .A1(n439), .B1(n339), .ZN(n344) );
  VHSR_INAND3_1 U273 ( .A1(n326), .B1(a[7]), .B2(b[1]), .ZN(n284) );
  VHSR_CLKN_1 U274 ( .I(n327), .ZN(n282) );
  VHSR_NOR2_2 U275 ( .A1(n476), .A2(n475), .ZN(n474) );
  VHSR_AD1_1 U276 ( .A(n445), .B(n444), .CI(n443), .CO(n440), .S(product[6])
         );
  VHSR_AD1_1 U277 ( .A(n447), .B(n446), .CI(n478), .CO(n443), .S(product[5])
         );
  VHSR_AD1_1 U278 ( .A(n442), .B(n441), .CI(n440), .CO(n437), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U279 ( .A(n439), .B(n438), .CI(n437), .CO(n448), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U280 ( .A(n436), .B(n435), .CI(n434), .CO(n451), .S(product[10])
         );
  VHSR_CLKNAND2_2 U281 ( .A1(b[6]), .A2(a[2]), .ZN(n301) );
  VHSR_CLKNAND2_2 U282 ( .A1(b[4]), .A2(a[2]), .ZN(n322) );
  VHSR_NAND3_2 U283 ( .A1(a[3]), .A2(b[5]), .A3(n322), .ZN(n262) );
  VHSR_CLKNAND2_2 U284 ( .A1(b[6]), .A2(a[0]), .ZN(n323) );
  VHSR_NAND3_2 U285 ( .A1(b[7]), .A2(a[1]), .A3(n323), .ZN(n264) );
  VHSR_MAOI222_2 U286 ( .A(n301), .B(n262), .C(n264), .ZN(n266) );
  VHSR_CLKNAND2_2 U287 ( .A1(b[4]), .A2(a[0]), .ZN(n476) );
  VHSR_NAND3_2 U288 ( .A1(a[1]), .A2(b[5]), .A3(n476), .ZN(n321) );
  VHSR_MAOI222_2 U289 ( .A(n323), .B(n322), .C(n321), .ZN(n320) );
  VHSR_IN_2 U290 ( .I(b[5]), .ZN(n365) );
  VHSR_IN_2 U291 ( .I(a[1]), .ZN(n458) );
  VHSR_NOR3_2 U292 ( .A1(n365), .A2(n458), .A3(n476), .ZN(n330) );
  VHSR_NAND4_2 U293 ( .A1(b[4]), .A2(b[5]), .A3(a[3]), .A4(a[2]), .ZN(n270) );
  VHSR_AOI22_2 U294 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n258) );
  VHSR_IN_2 U295 ( .I(b[7]), .ZN(n297) );
  VHSR_NOR3_2 U296 ( .A1(n297), .A2(n323), .A3(n458), .ZN(n274) );
  VHSR_AOI22_2 U297 ( .A1(b[6]), .A2(a[1]), .B1(b[7]), .B2(a[0]), .ZN(n259) );
  VHSR_NOR2_1 U298 ( .A1(n274), .A2(n259), .ZN(n260) );
  VHSR_AND2_2 U299 ( .A1(n320), .A2(n316), .Z(n315) );
  VHSR_AD1_1 U300 ( .A(n330), .B(n261), .CI(n260), .CO(n310), .S(n316) );
  VHSR_NOR2_1 U301 ( .A1(n315), .A2(n310), .ZN(n313) );
  VHSR_AND2_2 U302 ( .A1(n301), .A2(n262), .Z(n263) );
  VHSR_AOI21_2 U303 ( .A1(n264), .A2(n263), .B(n266), .ZN(n265) );
  VHSR_IN_2 U304 ( .I(n265), .ZN(n314) );
  VHSR_NOR2_1 U305 ( .A1(n313), .A2(n314), .ZN(n311) );
  VHSR_NOR2_1 U306 ( .A1(n266), .A2(n311), .ZN(n307) );
  VHSR_CLKNAND2_2 U307 ( .A1(b[7]), .A2(a[2]), .ZN(n268) );
  VHSR_AOI21_2 U308 ( .A1(b[6]), .A2(a[3]), .B(n268), .ZN(n267) );
  VHSR_AOI31_2 U309 ( .A1(b[6]), .A2(n268), .A3(a[3]), .B(n267), .ZN(n269) );
  VHSR_CLKNAND2_2 U310 ( .A1(n270), .A2(n269), .ZN(n273) );
  VHSR_IN_2 U311 ( .I(n274), .ZN(n271) );
  VHSR_MAOI222_2 U312 ( .A(n271), .B(n270), .C(n269), .ZN(n275) );
  VHSR_IN_2 U313 ( .I(n275), .ZN(n272) );
  VHSR_OAI21_2 U314 ( .A1(n274), .A2(n273), .B(n272), .ZN(n306) );
  VHSR_IN_2 U315 ( .I(a[3]), .ZN(n392) );
  VHSR_AOI211_2 U316 ( .A1(n296), .A2(n301), .B(n392), .C(n297), .ZN(n354) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[6]), .A2(b[2]), .ZN(n279) );
  VHSR_IN_2 U318 ( .I(n279), .ZN(n295) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[7]), .A2(b[3]), .ZN(n293) );
  VHSR_AOI22_2 U320 ( .A1(a[6]), .A2(b[3]), .B1(a[7]), .B2(b[2]), .ZN(n276) );
  VHSR_IAO21_2 U321 ( .A1(n293), .A2(n279), .B(n276), .ZN(n304) );
  VHSR_IN_2 U322 ( .I(b[3]), .ZN(n393) );
  VHSR_IN_2 U323 ( .I(a[5]), .ZN(n367) );
  VHSR_CLKNAND2_2 U324 ( .A1(a[4]), .A2(b[2]), .ZN(n327) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[6]), .A2(b[0]), .ZN(n326) );
  VHSR_OAI31_2 U326 ( .A1(n393), .A2(n367), .A3(n327), .B(n284), .ZN(n303) );
  VHSR_NAND3_2 U327 ( .A1(n326), .A2(b[1]), .A3(a[7]), .ZN(n278) );
  VHSR_OAI31_2 U328 ( .A1(n393), .A2(n367), .A3(n282), .B(n278), .ZN(n280) );
  VHSR_NAND3_2 U329 ( .A1(b[3]), .A2(a[5]), .A3(n327), .ZN(n277) );
  VHSR_MAOI222_2 U330 ( .A(n279), .B(n278), .C(n277), .ZN(n290) );
  VHSR_IAO21_2 U331 ( .A1(n280), .A2(n295), .B(n290), .ZN(n309) );
  VHSR_IN_2 U332 ( .I(b[1]), .ZN(n460) );
  VHSR_CLKNAND2_2 U333 ( .A1(a[4]), .A2(b[0]), .ZN(n475) );
  VHSR_NOR3_2 U334 ( .A1(n367), .A2(n460), .A3(n475), .ZN(n329) );
  VHSR_AOI22_2 U335 ( .A1(a[4]), .A2(b[3]), .B1(a[5]), .B2(b[2]), .ZN(n281) );
  VHSR_AOI31_2 U336 ( .A1(b[3]), .A2(a[5]), .A3(n282), .B(n281), .ZN(n286) );
  VHSR_AOI22_2 U337 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n283) );
  VHSR_MAOI222_2 U338 ( .A(n329), .B(n286), .C(n285), .ZN(n289) );
  VHSR_NAND3_2 U339 ( .A1(b[1]), .A2(a[5]), .A3(n475), .ZN(n325) );
  VHSR_MAOI222_2 U340 ( .A(n327), .B(n326), .C(n325), .ZN(n324) );
  VHSR_OR2_2 U341 ( .A1(n286), .A2(n285), .Z(n287) );
  VHSR_OAI21_2 U342 ( .A1(n287), .A2(n329), .B(n289), .ZN(n288) );
  VHSR_IN_2 U343 ( .I(n288), .ZN(n318) );
  VHSR_CLKNAND2_2 U344 ( .A1(n324), .A2(n318), .ZN(n317) );
  VHSR_CLKNAND2_2 U345 ( .A1(n289), .A2(n317), .ZN(n308) );
  VHSR_AOI21_2 U346 ( .A1(n309), .A2(n308), .B(n290), .ZN(n291) );
  VHSR_IN_2 U347 ( .I(n291), .ZN(n302) );
  VHSR_IAO21_2 U348 ( .A1(n295), .A2(n294), .B(n293), .ZN(n353) );
  VHSR_OAI21_2 U349 ( .A1(n295), .A2(n293), .B(n294), .ZN(n292) );
  VHSR_OAI31_2 U350 ( .A1(n295), .A2(n294), .A3(n293), .B(n292), .ZN(n360) );
  VHSR_IN_2 U351 ( .I(n296), .ZN(n300) );
  VHSR_NOR2_1 U352 ( .A1(n297), .A2(n392), .ZN(n299) );
  VHSR_AOI21_2 U353 ( .A1(n301), .A2(n299), .B(n300), .ZN(n298) );
  VHSR_AOI31_2 U354 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n359) );
  VHSR_AD1_1 U355 ( .A(n304), .B(n303), .CI(n302), .CO(n294), .S(n363) );
  VHSR_AOI21_2 U356 ( .A1(n307), .A2(n306), .B(n305), .ZN(n362) );
  VHSR_CLKXOR2_2 U357 ( .A1(n309), .A2(n308), .Z(n372) );
  VHSR_CLKNAND2_2 U358 ( .A1(n315), .A2(n310), .ZN(n312) );
  VHSR_AOI22_2 U359 ( .A1(n314), .A2(n313), .B1(n312), .B2(n311), .ZN(n371) );
  VHSR_IAO21_2 U360 ( .A1(n320), .A2(n316), .B(n315), .ZN(n397) );
  VHSR_OAI21_2 U361 ( .A1(n324), .A2(n318), .B(n317), .ZN(n319) );
  VHSR_IN_2 U362 ( .I(n319), .ZN(n396) );
  VHSR_AOI31_2 U363 ( .A1(n323), .A2(n322), .A3(n321), .B(n320), .ZN(n405) );
  VHSR_AOI31_2 U364 ( .A1(n327), .A2(n326), .A3(n325), .B(n324), .ZN(n404) );
  VHSR_AOI22_2 U365 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n328) );
  VHSR_NOR2_1 U366 ( .A1(n329), .A2(n328), .ZN(n407) );
  VHSR_AOI22_2 U367 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n331) );
  VHSR_NOR2_1 U368 ( .A1(n331), .A2(n330), .ZN(n406) );
  VHSR_CLKNAND2_2 U369 ( .A1(a[6]), .A2(b[6]), .ZN(n432) );
  VHSR_IN_2 U370 ( .I(n432), .ZN(n465) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[4]), .A2(b[6]), .ZN(n341) );
  VHSR_IN_2 U372 ( .I(n341), .ZN(n334) );
  VHSR_CLKNAND2_2 U373 ( .A1(a[5]), .A2(b[7]), .ZN(n333) );
  VHSR_CLKNAND2_2 U374 ( .A1(b[4]), .A2(a[6]), .ZN(n342) );
  VHSR_IN_2 U375 ( .I(n342), .ZN(n337) );
  VHSR_CLKNAND2_2 U376 ( .A1(b[5]), .A2(a[7]), .ZN(n332) );
  VHSR_OAI22_2 U377 ( .A1(n334), .A2(n333), .B1(n337), .B2(n332), .ZN(n336) );
  VHSR_AOI22_2 U378 ( .A1(b[4]), .A2(a[6]), .B1(a[4]), .B2(b[6]), .ZN(n357) );
  VHSR_CLKNAND2_2 U379 ( .A1(b[5]), .A2(a[5]), .ZN(n339) );
  VHSR_CLKNAND2_2 U380 ( .A1(a[7]), .A2(b[7]), .ZN(n466) );
  VHSR_AOI31_2 U381 ( .A1(b[6]), .A2(a[6]), .A3(n336), .B(n335), .ZN(n417) );
  VHSR_OAI21_2 U382 ( .A1(n465), .A2(n336), .B(n417), .ZN(n349) );
  VHSR_NAND3_2 U383 ( .A1(n337), .A2(b[5]), .A3(a[7]), .ZN(n422) );
  VHSR_IN_2 U384 ( .I(n422), .ZN(n424) );
  VHSR_AOI22_2 U385 ( .A1(b[4]), .A2(a[7]), .B1(a[6]), .B2(b[5]), .ZN(n338) );
  VHSR_IN_2 U386 ( .I(b[4]), .ZN(n369) );
  VHSR_IN_2 U387 ( .I(a[4]), .ZN(n364) );
  VHSR_NAND4_2 U388 ( .A1(a[4]), .A2(b[6]), .A3(a[5]), .A4(b[7]), .ZN(n421) );
  VHSR_AOI22_2 U389 ( .A1(a[4]), .A2(b[7]), .B1(b[6]), .B2(a[5]), .ZN(n340) );
  VHSR_OAI22_2 U390 ( .A1(n357), .A2(n356), .B1(n342), .B2(n341), .ZN(n355) );
  VHSR_AND2_2 U391 ( .A1(n351), .A2(n355), .Z(n350) );
  VHSR_AD1_1 U392 ( .A(n345), .B(n344), .CI(n343), .CO(n346), .S(n351) );
  VHSR_CLKNAND2_2 U393 ( .A1(n350), .A2(n346), .ZN(n347) );
  VHSR_AOI22_2 U394 ( .A1(n349), .A2(n348), .B1(n347), .B2(n416), .ZN(n455) );
  VHSR_IAO21_2 U395 ( .A1(n351), .A2(n355), .B(n350), .ZN(n453) );
  VHSR_AOI21_2 U396 ( .A1(n357), .A2(n356), .B(n355), .ZN(n436) );
  VHSR_AD1_1 U397 ( .A(n360), .B(n359), .CI(n358), .CO(n352), .S(n435) );
  VHSR_AD1_1 U398 ( .A(n363), .B(n362), .CI(n361), .CO(n358), .S(n450) );
  VHSR_NOR2_1 U399 ( .A1(n365), .A2(n364), .ZN(n368) );
  VHSR_OAI21_2 U400 ( .A1(n369), .A2(n367), .B(n368), .ZN(n366) );
  VHSR_OAI31_2 U401 ( .A1(n369), .A2(n368), .A3(n367), .B(n366), .ZN(n449) );
  VHSR_AD1_1 U402 ( .A(n372), .B(n371), .CI(n370), .CO(n361), .S(n438) );
  VHSR_CLKNAND2_2 U403 ( .A1(a[2]), .A2(b[3]), .ZN(n374) );
  VHSR_AOI21_2 U404 ( .A1(a[3]), .A2(b[2]), .B(n374), .ZN(n373) );
  VHSR_AOI31_2 U405 ( .A1(a[3]), .A2(n374), .A3(b[2]), .B(n373), .ZN(n390) );
  VHSR_IN_2 U406 ( .I(n390), .ZN(n375) );
  VHSR_CLKNAND2_2 U407 ( .A1(a[2]), .A2(b[0]), .ZN(n473) );
  VHSR_NOR3_2 U408 ( .A1(n392), .A2(n473), .A3(n460), .ZN(n387) );
  VHSR_CLKNAND2_2 U409 ( .A1(a[0]), .A2(b[2]), .ZN(n472) );
  VHSR_NOR3_2 U410 ( .A1(n458), .A2(n393), .A3(n472), .ZN(n386) );
  VHSR_MAOI222_2 U411 ( .A(n375), .B(n387), .C(n386), .ZN(n391) );
  VHSR_CLKNAND2_2 U412 ( .A1(a[2]), .A2(b[2]), .ZN(n394) );
  VHSR_IN_2 U413 ( .I(n394), .ZN(n401) );
  VHSR_NAND3_2 U414 ( .A1(b[3]), .A2(a[1]), .A3(n472), .ZN(n382) );
  VHSR_IN_2 U415 ( .I(n382), .ZN(n376) );
  VHSR_AOI211_2 U416 ( .A1(b[0]), .A2(a[2]), .B(n392), .C(n460), .ZN(n384) );
  VHSR_MAOI222_2 U417 ( .A(n401), .B(n376), .C(n384), .ZN(n385) );
  VHSR_AOI22_2 U418 ( .A1(a[3]), .A2(b[0]), .B1(a[2]), .B2(b[1]), .ZN(n377) );
  VHSR_NOR2_1 U419 ( .A1(n387), .A2(n377), .ZN(n381) );
  VHSR_IN_2 U420 ( .I(a[0]), .ZN(n462) );
  VHSR_IN_2 U421 ( .I(b[0]), .ZN(n457) );
  VHSR_NOR2_1 U422 ( .A1(n462), .A2(n457), .ZN(product[0]) );
  VHSR_AND3_2 U423 ( .A1(product[0]), .A2(a[1]), .A3(b[1]), .Z(n380) );
  VHSR_AOI22_2 U424 ( .A1(a[0]), .A2(b[3]), .B1(a[1]), .B2(b[2]), .ZN(n378) );
  VHSR_NOR2_1 U425 ( .A1(n386), .A2(n378), .ZN(n379) );
  VHSR_AD1_1 U426 ( .A(n381), .B(n380), .CI(n379), .CO(n415), .S(n464) );
  VHSR_MAOI222_2 U427 ( .A(n473), .B(n472), .C(n471), .ZN(n470) );
  VHSR_CLKNAND2_2 U428 ( .A1(n464), .A2(n470), .ZN(n413) );
  VHSR_IN_2 U429 ( .I(n413), .ZN(n463) );
  VHSR_CLKNAND2_2 U430 ( .A1(n394), .A2(n382), .ZN(n383) );
  VHSR_OAI21_2 U431 ( .A1(n384), .A2(n383), .B(n385), .ZN(n411) );
  VHSR_IAO21_2 U432 ( .A1(n415), .A2(n463), .B(n411), .ZN(n412) );
  VHSR_NOR2_1 U433 ( .A1(n387), .A2(n386), .ZN(n389) );
  VHSR_AOI22_2 U434 ( .A1(n387), .A2(n386), .B1(n390), .B2(n389), .ZN(n388) );
  VHSR_OAI21_2 U435 ( .A1(n390), .A2(n389), .B(n388), .ZN(n409) );
  VHSR_AOI211_2 U436 ( .A1(n398), .A2(n394), .B(n393), .C(n392), .ZN(n442) );
  VHSR_AD1_1 U437 ( .A(n397), .B(n396), .CI(n395), .CO(n370), .S(n441) );
  VHSR_CLKNAND2_2 U438 ( .A1(a[3]), .A2(b[3]), .ZN(n402) );
  VHSR_IN_2 U439 ( .I(n398), .ZN(n400) );
  VHSR_OAI21_2 U440 ( .A1(n402), .A2(n401), .B(n400), .ZN(n399) );
  VHSR_OAI31_2 U441 ( .A1(n402), .A2(n401), .A3(n400), .B(n399), .ZN(n445) );
  VHSR_AD1_1 U442 ( .A(n405), .B(n404), .CI(n403), .CO(n395), .S(n444) );
  VHSR_AD1_1 U443 ( .A(n407), .B(n474), .CI(n406), .CO(n403), .S(n447) );
  VHSR_AOI21_2 U444 ( .A1(n410), .A2(n409), .B(n408), .ZN(n446) );
  VHSR_AOI21_2 U445 ( .A1(n413), .A2(n411), .B(n412), .ZN(n414) );
  VHSR_AOI211_2 U446 ( .A1(n476), .A2(n475), .B(n474), .C(n479), .ZN(n478) );
  VHSR_CLKNAND2_2 U447 ( .A1(a[7]), .A2(b[6]), .ZN(n419) );
  VHSR_AOI21_2 U448 ( .A1(a[6]), .A2(b[7]), .B(n419), .ZN(n418) );
  VHSR_AOI31_2 U449 ( .A1(a[6]), .A2(n419), .A3(b[7]), .B(n418), .ZN(n420) );
  VHSR_CLKNAND2_2 U450 ( .A1(n421), .A2(n420), .ZN(n423) );
  VHSR_MAOI222_2 U451 ( .A(n422), .B(n421), .C(n420), .ZN(n430) );
  VHSR_IAO21_2 U452 ( .A1(n424), .A2(n423), .B(n430), .ZN(n429) );
  VHSR_XNOR2_2 U453 ( .A1(n428), .A2(n429), .ZN(n425) );
  VHSR_CLKNAND2_2 U454 ( .A1(n426), .A2(n425), .ZN(n467) );
  VHSR_OAI21_2 U455 ( .A1(n426), .A2(n425), .B(n467), .ZN(n427) );
  VHSR_AND3_2 U456 ( .A1(n468), .A2(n432), .A3(n467), .Z(n433) );
  VHSR_NOR2_1 U457 ( .A1(n466), .A2(n433), .ZN(product[15]) );
  VHSR_AD1_1 U458 ( .A(n450), .B(n449), .CI(n448), .CO(n434), .S(product[9])
         );
  VHSR_AD1_1 U459 ( .A(n453), .B(n452), .CI(n451), .CO(n454), .S(product[11])
         );
  VHSR_NOR2_1 U460 ( .A1(n458), .A2(n457), .ZN(n461) );
  VHSR_OAI21_2 U461 ( .A1(n462), .A2(n460), .B(n461), .ZN(n459) );
  VHSR_OAI31_2 U462 ( .A1(n462), .A2(n461), .A3(n460), .B(n459), .ZN(
        product[1]) );
  VHSR_IAO21_2 U463 ( .A1(n464), .A2(n470), .B(n463), .ZN(product[3]) );
  VHSR_NOR2_1 U464 ( .A1(n466), .A2(n465), .ZN(n469) );
  VHSR_XOR3_2 U465 ( .A1(n469), .A2(n468), .A3(n467), .Z(product[14]) );
  VHSR_AOI31_2 U466 ( .A1(n473), .A2(n472), .A3(n471), .B(n470), .ZN(
        product[2]) );
  VHSR_AOI21_2 U467 ( .A1(n476), .A2(n475), .B(n474), .ZN(n477) );
  VHSR_IN_2 U468 ( .I(n477), .ZN(n480) );
  VHSR_AOI21_2 U469 ( .A1(n480), .A2(n479), .B(n478), .ZN(product[4]) );
endmodule


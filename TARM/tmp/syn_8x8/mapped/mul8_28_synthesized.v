
module mul8_28 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , \intadd_0/SUM[0] , n250, n251, n252, n253, n254,
         n255, n256, n257, n258, n259, n260, n261, n262, n263, n264, n265,
         n266, n267, n268, n269, n270, n271, n272, n273, n274, n275, n276,
         n277, n278, n279, n280, n281, n282, n283, n284, n285, n286, n287,
         n288, n289, n290, n291, n292, n293, n294, n295, n296, n297, n298,
         n299, n300, n301, n302, n303, n304, n305, n306, n307, n308, n309,
         n310, n311, n312, n313, n314, n315, n316, n317, n318, n319, n320,
         n321, n322, n323, n324, n325, n326, n327, n328, n329, n330, n331,
         n332, n333, n334, n335, n336, n337, n338, n339, n340, n341, n342,
         n343, n344, n345, n346, n347, n348, n349, n350, n351, n352, n353,
         n354, n355, n356, n357, n358, n359, n360, n361, n362, n363, n364,
         n365, n366, n367, n368, n369, n370, n371, n372, n373, n374, n375,
         n376, n377, n378, n379, n380, n381, n382, n383, n384, n385, n386,
         n387, n388, n389, n390, n391, n392, n393, n394, n395, n396, n397,
         n398, n399, n400, n401, n402, n403, n404, n405, n406, n407, n408,
         n409, n410, n411, n412, n413, n414, n415, n416, n417, n418, n419,
         n420, n421, n422, n423, n424, n425, n426, n427, n428, n429, n430,
         n431, n432, n433, n434, n435, n436, n437, n438, n439, n440, n441,
         n442, n443, n444, n445, n446, n447, n448, n449, n450, n451, n452,
         n453, n454, n455, n456, n457, n458, n459, n460, n461, n462, n463,
         n464, n465, n466, n467, n468, n469, n470, n471, n472, n473, n474,
         n475, n476;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;
  assign product[5] = \intadd_0/SUM[0] ;

  VHSR_INOR2_2 U240 ( .A1(n420), .B1(n333), .ZN(n338) );
  VHSR_INAND2_2 U241 ( .A1(n297), .B1(n282), .ZN(n287) );
  VHSR_IN_2 U242 ( .I(n287), .ZN(n283) );
  VHSR_NOR2_1 U243 ( .A1(n457), .A2(n371), .ZN(n379) );
  VHSR_NOR2_1 U244 ( .A1(n408), .A2(n406), .ZN(n409) );
  VHSR_NOR2_1 U245 ( .A1(n346), .A2(n350), .ZN(n345) );
  VHSR_INOR2_2 U246 ( .A1(n428), .B1(n427), .ZN(n430) );
  VHSR_NOR2_1 U247 ( .A1(n463), .A2(n464), .ZN(n462) );
  VHSR_NOR2_1 U248 ( .A1(n365), .A2(n360), .ZN(n444) );
  VHSR_IN_2 U249 ( .I(n426), .ZN(product[13]) );
  VHSR_INOR2_1 U250 ( .A1(n416), .B1(n415), .ZN(n427) );
  VHSR_INOR2_1 U251 ( .A1(n388), .B1(n414), .ZN(n407) );
  VHSR_INOR2_1 U252 ( .A1(n444), .B1(n335), .ZN(n339) );
  VHSR_NOR2_2 U253 ( .A1(n472), .A2(n471), .ZN(n470) );
  VHSR_NOR2_2 U254 ( .A1(n369), .A2(n371), .ZN(n399) );
  VHSR_AD1_1 U255 ( .A(n441), .B(n440), .CI(n439), .CO(n436), .S(product[9])
         );
  VHSR_AD1_1 U256 ( .A(n438), .B(n437), .CI(n436), .CO(n433), .S(product[10])
         );
  VHSR_AD1_1 U257 ( .A(n449), .B(n448), .CI(n474), .CO(n450), .S(
        \intadd_0/SUM[0] ) );
  VHSR_AD1_1 U258 ( .A(n447), .B(n446), .CI(n445), .CO(n442), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U259 ( .A(n444), .B(n443), .CI(n442), .CO(n439), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U260 ( .A(n435), .B(n434), .CI(n433), .CO(n453), .S(
        \intadd_0/SUM[6] ) );
  VHSR_IN_2 U261 ( .I(b[0]), .ZN(n457) );
  VHSR_IN_2 U262 ( .I(a[2]), .ZN(n371) );
  VHSR_IN_2 U263 ( .I(b[2]), .ZN(n369) );
  VHSR_IN_2 U264 ( .I(a[0]), .ZN(n459) );
  VHSR_NOR2_1 U265 ( .A1(n369), .A2(n459), .ZN(n380) );
  VHSR_NOR2_1 U266 ( .A1(n457), .A2(n459), .ZN(product[0]) );
  VHSR_IN_2 U267 ( .I(a[1]), .ZN(n456) );
  VHSR_IN_2 U268 ( .I(b[1]), .ZN(n461) );
  VHSR_NOR3_2 U269 ( .A1(product[0]), .A2(n456), .A3(n461), .ZN(n250) );
  VHSR_MAOI222_2 U270 ( .A(n379), .B(n380), .C(n250), .ZN(n464) );
  VHSR_OAI31_2 U271 ( .A1(n379), .A2(n380), .A3(n250), .B(n464), .ZN(n251) );
  VHSR_IN_2 U272 ( .I(n251), .ZN(product[2]) );
  VHSR_AOI22_2 U273 ( .A1(a[7]), .A2(b[2]), .B1(b[3]), .B2(a[6]), .ZN(n296) );
  VHSR_IN_2 U274 ( .I(b[3]), .ZN(n390) );
  VHSR_CLKNAND2_2 U275 ( .A1(b[2]), .A2(a[4]), .ZN(n321) );
  VHSR_IN_2 U276 ( .I(a[5]), .ZN(n361) );
  VHSR_NOR3_2 U277 ( .A1(n390), .A2(n321), .A3(n361), .ZN(n294) );
  VHSR_AOI211_2 U278 ( .A1(a[4]), .A2(b[2]), .B(n390), .C(n361), .ZN(n254) );
  VHSR_IN_2 U279 ( .I(a[7]), .ZN(n289) );
  VHSR_NOR2_1 U280 ( .A1(n289), .A2(n461), .ZN(n253) );
  VHSR_CLKNAND2_2 U281 ( .A1(a[6]), .A2(b[2]), .ZN(n256) );
  VHSR_IN_2 U282 ( .I(n256), .ZN(n252) );
  VHSR_MAOI222_2 U283 ( .A(n254), .B(n253), .C(n252), .ZN(n265) );
  VHSR_AOI21_2 U284 ( .A1(a[7]), .A2(b[1]), .B(n254), .ZN(n257) );
  VHSR_IN_2 U285 ( .I(n265), .ZN(n255) );
  VHSR_AOI21_2 U286 ( .A1(n257), .A2(n256), .B(n255), .ZN(n302) );
  VHSR_CLKNAND2_2 U287 ( .A1(a[4]), .A2(b[0]), .ZN(n472) );
  VHSR_NOR3_2 U288 ( .A1(n361), .A2(n461), .A3(n472), .ZN(n323) );
  VHSR_AOI22_2 U289 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n258) );
  VHSR_NOR2_1 U290 ( .A1(n294), .A2(n258), .ZN(n260) );
  VHSR_AOI22_2 U291 ( .A1(a[7]), .A2(b[0]), .B1(a[6]), .B2(b[1]), .ZN(n262) );
  VHSR_IN_2 U292 ( .I(n262), .ZN(n259) );
  VHSR_MAOI222_2 U293 ( .A(n323), .B(n260), .C(n259), .ZN(n264) );
  VHSR_NAND3_2 U294 ( .A1(b[1]), .A2(a[5]), .A3(n472), .ZN(n320) );
  VHSR_CLKNAND2_2 U295 ( .A1(a[6]), .A2(b[0]), .ZN(n319) );
  VHSR_MAOI222_2 U296 ( .A(n321), .B(n320), .C(n319), .ZN(n318) );
  VHSR_NOR2_1 U297 ( .A1(n323), .A2(n260), .ZN(n263) );
  VHSR_IN_2 U298 ( .I(n264), .ZN(n261) );
  VHSR_AOI21_2 U299 ( .A1(n263), .A2(n262), .B(n261), .ZN(n312) );
  VHSR_CLKNAND2_2 U300 ( .A1(n318), .A2(n312), .ZN(n311) );
  VHSR_CLKNAND2_2 U301 ( .A1(n264), .A2(n311), .ZN(n301) );
  VHSR_CLKNAND2_2 U302 ( .A1(n302), .A2(n301), .ZN(n300) );
  VHSR_CLKNAND2_2 U303 ( .A1(n265), .A2(n300), .ZN(n293) );
  VHSR_NOR2_1 U304 ( .A1(n294), .A2(n293), .ZN(n292) );
  VHSR_NOR2_1 U305 ( .A1(n296), .A2(n292), .ZN(n290) );
  VHSR_AND3_2 U306 ( .A1(n290), .A2(a[7]), .A3(b[3]), .Z(n349) );
  VHSR_CLKNAND2_2 U307 ( .A1(b[6]), .A2(a[2]), .ZN(n288) );
  VHSR_CLKNAND2_2 U308 ( .A1(b[4]), .A2(a[2]), .ZN(n316) );
  VHSR_NAND3_2 U309 ( .A1(a[3]), .A2(b[5]), .A3(n316), .ZN(n270) );
  VHSR_CLKNAND2_2 U310 ( .A1(b[6]), .A2(a[0]), .ZN(n317) );
  VHSR_NAND3_2 U311 ( .A1(b[7]), .A2(a[1]), .A3(n317), .ZN(n272) );
  VHSR_MAOI222_2 U312 ( .A(n288), .B(n270), .C(n272), .ZN(n274) );
  VHSR_CLKNAND2_2 U313 ( .A1(b[4]), .A2(a[0]), .ZN(n471) );
  VHSR_NAND3_2 U314 ( .A1(a[1]), .A2(b[5]), .A3(n471), .ZN(n315) );
  VHSR_MAOI222_2 U315 ( .A(n317), .B(n316), .C(n315), .ZN(n314) );
  VHSR_IN_2 U316 ( .I(b[5]), .ZN(n363) );
  VHSR_NOR3_2 U317 ( .A1(n363), .A2(n456), .A3(n471), .ZN(n324) );
  VHSR_IN_2 U318 ( .I(b[4]), .ZN(n360) );
  VHSR_IN_2 U319 ( .I(a[3]), .ZN(n391) );
  VHSR_NOR4_2 U320 ( .A1(n360), .A2(n363), .A3(n391), .A4(n371), .ZN(n279) );
  VHSR_AOI22_2 U321 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n266) );
  VHSR_NOR2_1 U322 ( .A1(n279), .A2(n266), .ZN(n269) );
  VHSR_IN_2 U323 ( .I(b[7]), .ZN(n284) );
  VHSR_NOR3_2 U324 ( .A1(n284), .A2(n317), .A3(n456), .ZN(n281) );
  VHSR_AOI22_2 U325 ( .A1(b[6]), .A2(a[1]), .B1(b[7]), .B2(a[0]), .ZN(n267) );
  VHSR_NOR2_1 U326 ( .A1(n281), .A2(n267), .ZN(n268) );
  VHSR_AND2_2 U327 ( .A1(n314), .A2(n310), .Z(n309) );
  VHSR_AD1_1 U328 ( .A(n324), .B(n269), .CI(n268), .CO(n304), .S(n310) );
  VHSR_NOR2_1 U329 ( .A1(n309), .A2(n304), .ZN(n307) );
  VHSR_AND2_2 U330 ( .A1(n288), .A2(n270), .Z(n271) );
  VHSR_AOI21_2 U331 ( .A1(n272), .A2(n271), .B(n274), .ZN(n273) );
  VHSR_IN_2 U332 ( .I(n273), .ZN(n308) );
  VHSR_NOR2_1 U333 ( .A1(n307), .A2(n308), .ZN(n305) );
  VHSR_NOR2_1 U334 ( .A1(n274), .A2(n305), .ZN(n299) );
  VHSR_CLKNAND2_2 U335 ( .A1(b[7]), .A2(a[2]), .ZN(n276) );
  VHSR_AOI21_2 U336 ( .A1(b[6]), .A2(a[3]), .B(n276), .ZN(n275) );
  VHSR_AOI31_2 U337 ( .A1(b[6]), .A2(n276), .A3(a[3]), .B(n275), .ZN(n277) );
  VHSR_IN_2 U338 ( .I(n277), .ZN(n278) );
  VHSR_OR2_2 U339 ( .A1(n279), .A2(n278), .Z(n280) );
  VHSR_MAOI222_2 U340 ( .A(n281), .B(n279), .C(n278), .ZN(n282) );
  VHSR_OAI21_2 U341 ( .A1(n281), .A2(n280), .B(n282), .ZN(n298) );
  VHSR_NOR2_1 U342 ( .A1(n299), .A2(n298), .ZN(n297) );
  VHSR_AOI211_2 U343 ( .A1(n283), .A2(n288), .B(n391), .C(n284), .ZN(n348) );
  VHSR_NOR2_1 U344 ( .A1(n284), .A2(n391), .ZN(n286) );
  VHSR_AOI21_2 U345 ( .A1(n288), .A2(n286), .B(n287), .ZN(n285) );
  VHSR_AOI31_2 U346 ( .A1(n288), .A2(n287), .A3(n286), .B(n285), .ZN(n356) );
  VHSR_NOR2_1 U347 ( .A1(n289), .A2(n390), .ZN(n291) );
  VHSR_IAO21_2 U348 ( .A1(n291), .A2(n290), .B(n349), .ZN(n355) );
  VHSR_AOI21_2 U349 ( .A1(n294), .A2(n293), .B(n292), .ZN(n295) );
  VHSR_XNOR2_2 U350 ( .A1(n296), .A2(n295), .ZN(n359) );
  VHSR_AOI21_2 U351 ( .A1(n299), .A2(n298), .B(n297), .ZN(n358) );
  VHSR_OAI21_2 U352 ( .A1(n302), .A2(n301), .B(n300), .ZN(n303) );
  VHSR_IN_2 U353 ( .I(n303), .ZN(n368) );
  VHSR_CLKNAND2_2 U354 ( .A1(n309), .A2(n304), .ZN(n306) );
  VHSR_AOI22_2 U355 ( .A1(n308), .A2(n307), .B1(n306), .B2(n305), .ZN(n367) );
  VHSR_IAO21_2 U356 ( .A1(n314), .A2(n310), .B(n309), .ZN(n395) );
  VHSR_OAI21_2 U357 ( .A1(n318), .A2(n312), .B(n311), .ZN(n313) );
  VHSR_IN_2 U358 ( .I(n313), .ZN(n394) );
  VHSR_AOI31_2 U359 ( .A1(n317), .A2(n316), .A3(n315), .B(n314), .ZN(n403) );
  VHSR_AOI31_2 U360 ( .A1(n321), .A2(n320), .A3(n319), .B(n318), .ZN(n402) );
  VHSR_AOI22_2 U361 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n322) );
  VHSR_NOR2_1 U362 ( .A1(n323), .A2(n322), .ZN(n405) );
  VHSR_AOI22_2 U363 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n325) );
  VHSR_NOR2_1 U364 ( .A1(n325), .A2(n324), .ZN(n404) );
  VHSR_CLKNAND2_2 U365 ( .A1(a[6]), .A2(b[6]), .ZN(n431) );
  VHSR_IN_2 U366 ( .I(n431), .ZN(n465) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[4]), .A2(b[6]), .ZN(n328) );
  VHSR_IN_2 U368 ( .I(n328), .ZN(n336) );
  VHSR_CLKNAND2_2 U369 ( .A1(a[5]), .A2(b[7]), .ZN(n327) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[6]), .A2(b[4]), .ZN(n329) );
  VHSR_IN_2 U371 ( .I(n329), .ZN(n337) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[5]), .ZN(n326) );
  VHSR_OAI22_2 U373 ( .A1(n336), .A2(n327), .B1(n337), .B2(n326), .ZN(n331) );
  VHSR_CLKNAND2_2 U374 ( .A1(n329), .A2(n328), .ZN(n351) );
  VHSR_CLKNAND2_2 U375 ( .A1(a[5]), .A2(b[5]), .ZN(n335) );
  VHSR_CLKNAND2_2 U376 ( .A1(a[7]), .A2(b[7]), .ZN(n466) );
  VHSR_NOR3_2 U377 ( .A1(n351), .A2(n335), .A3(n466), .ZN(n330) );
  VHSR_AOI31_2 U378 ( .A1(b[6]), .A2(a[6]), .A3(n331), .B(n330), .ZN(n416) );
  VHSR_OAI21_2 U379 ( .A1(n465), .A2(n331), .B(n416), .ZN(n344) );
  VHSR_NAND3_2 U380 ( .A1(a[7]), .A2(n337), .A3(b[5]), .ZN(n421) );
  VHSR_IN_2 U381 ( .I(n421), .ZN(n423) );
  VHSR_AOI22_2 U382 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n332) );
  VHSR_NOR2_1 U383 ( .A1(n423), .A2(n332), .ZN(n340) );
  VHSR_IN_2 U384 ( .I(a[4]), .ZN(n365) );
  VHSR_NAND4_2 U385 ( .A1(a[4]), .A2(a[5]), .A3(b[6]), .A4(b[7]), .ZN(n420) );
  VHSR_AOI22_2 U386 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n333) );
  VHSR_IN_2 U387 ( .I(n334), .ZN(n346) );
  VHSR_NOR2_1 U388 ( .A1(n444), .A2(n335), .ZN(n352) );
  VHSR_AOI22_2 U389 ( .A1(n337), .A2(n336), .B1(n352), .B2(n351), .ZN(n350) );
  VHSR_AD1_1 U390 ( .A(n340), .B(n339), .CI(n338), .CO(n341), .S(n334) );
  VHSR_NOR2_1 U391 ( .A1(n345), .A2(n341), .ZN(n343) );
  VHSR_CLKNAND2_2 U392 ( .A1(n345), .A2(n341), .ZN(n342) );
  VHSR_NOR2_1 U393 ( .A1(n343), .A2(n344), .ZN(n415) );
  VHSR_AOI22_2 U394 ( .A1(n344), .A2(n343), .B1(n342), .B2(n415), .ZN(n454) );
  VHSR_AOI21_2 U395 ( .A1(n350), .A2(n346), .B(n345), .ZN(n435) );
  VHSR_AD1_1 U396 ( .A(n349), .B(n348), .CI(n347), .CO(n455), .S(n434) );
  VHSR_OAI21_2 U397 ( .A1(n352), .A2(n351), .B(n350), .ZN(n353) );
  VHSR_IN_2 U398 ( .I(n353), .ZN(n438) );
  VHSR_AD1_1 U399 ( .A(n356), .B(n355), .CI(n354), .CO(n347), .S(n437) );
  VHSR_AD1_1 U400 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(n441) );
  VHSR_NOR2_1 U401 ( .A1(n361), .A2(n360), .ZN(n364) );
  VHSR_OAI21_2 U402 ( .A1(n365), .A2(n363), .B(n364), .ZN(n362) );
  VHSR_OAI31_2 U403 ( .A1(n365), .A2(n364), .A3(n363), .B(n362), .ZN(n440) );
  VHSR_AD1_1 U404 ( .A(n368), .B(n367), .CI(n366), .CO(n357), .S(n443) );
  VHSR_NAND3_2 U405 ( .A1(b[1]), .A2(a[3]), .A3(n379), .ZN(n384) );
  VHSR_IN_2 U406 ( .I(n399), .ZN(n392) );
  VHSR_OAI22_2 U407 ( .A1(n390), .A2(n371), .B1(n369), .B2(n391), .ZN(n370) );
  VHSR_OAI31_2 U408 ( .A1(n391), .A2(n390), .A3(n392), .B(n370), .ZN(n383) );
  VHSR_NAND4_2 U409 ( .A1(b[3]), .A2(b[2]), .A3(a[0]), .A4(a[1]), .ZN(n373) );
  VHSR_MAOI222_2 U410 ( .A(n384), .B(n383), .C(n373), .ZN(n389) );
  VHSR_OAI22_2 U411 ( .A1(n461), .A2(n371), .B1(n457), .B2(n391), .ZN(n372) );
  VHSR_AND2_2 U412 ( .A1(n384), .A2(n372), .Z(n377) );
  VHSR_AND3_2 U413 ( .A1(product[0]), .A2(b[1]), .A3(a[1]), .Z(n376) );
  VHSR_IN_2 U414 ( .I(n373), .ZN(n387) );
  VHSR_AOI22_2 U415 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n374) );
  VHSR_NOR2_1 U416 ( .A1(n387), .A2(n374), .ZN(n375) );
  VHSR_AD1_1 U417 ( .A(n377), .B(n376), .CI(n375), .CO(n413), .S(n378) );
  VHSR_IN_2 U418 ( .I(n378), .ZN(n463) );
  VHSR_NOR3_2 U419 ( .A1(n379), .A2(n391), .A3(n461), .ZN(n382) );
  VHSR_NOR3_2 U420 ( .A1(n380), .A2(n456), .A3(n390), .ZN(n381) );
  VHSR_OAI21_2 U421 ( .A1(n413), .A2(n462), .B(n411), .ZN(n414) );
  VHSR_IN_2 U422 ( .I(n414), .ZN(n410) );
  VHSR_AD1_1 U423 ( .A(n399), .B(n382), .CI(n381), .CO(n388), .S(n411) );
  VHSR_NOR2_1 U424 ( .A1(n410), .A2(n388), .ZN(n408) );
  VHSR_CLKNAND2_2 U425 ( .A1(n384), .A2(n383), .ZN(n386) );
  VHSR_IN_2 U426 ( .I(n389), .ZN(n385) );
  VHSR_OAI21_2 U427 ( .A1(n387), .A2(n386), .B(n385), .ZN(n406) );
  VHSR_NOR3_2 U428 ( .A1(n389), .A2(n409), .A3(n407), .ZN(n396) );
  VHSR_AOI211_2 U429 ( .A1(n396), .A2(n392), .B(n391), .C(n390), .ZN(n447) );
  VHSR_AD1_1 U430 ( .A(n395), .B(n394), .CI(n393), .CO(n366), .S(n446) );
  VHSR_CLKNAND2_2 U431 ( .A1(b[3]), .A2(a[3]), .ZN(n400) );
  VHSR_IN_2 U432 ( .I(n396), .ZN(n398) );
  VHSR_OAI21_2 U433 ( .A1(n400), .A2(n399), .B(n398), .ZN(n397) );
  VHSR_OAI31_2 U434 ( .A1(n400), .A2(n399), .A3(n398), .B(n397), .ZN(n452) );
  VHSR_AD1_1 U435 ( .A(n403), .B(n402), .CI(n401), .CO(n393), .S(n451) );
  VHSR_AD1_1 U436 ( .A(n405), .B(n470), .CI(n404), .CO(n401), .S(n449) );
  VHSR_OAI32_2 U437 ( .A1(n409), .A2(n408), .A3(n407), .B1(n406), .B2(n409), 
        .ZN(n448) );
  VHSR_IAO21_2 U438 ( .A1(n462), .A2(n411), .B(n410), .ZN(n412) );
  VHSR_OAI22_2 U439 ( .A1(n462), .A2(n414), .B1(n413), .B2(n412), .ZN(n476) );
  VHSR_AOI211_2 U440 ( .A1(n472), .A2(n471), .B(n470), .C(n476), .ZN(n474) );
  VHSR_CLKNAND2_2 U441 ( .A1(a[6]), .A2(b[7]), .ZN(n418) );
  VHSR_AOI21_2 U442 ( .A1(a[7]), .A2(b[6]), .B(n418), .ZN(n417) );
  VHSR_AOI31_2 U443 ( .A1(a[7]), .A2(n418), .A3(b[6]), .B(n417), .ZN(n419) );
  VHSR_CLKNAND2_2 U444 ( .A1(n420), .A2(n419), .ZN(n422) );
  VHSR_MAOI222_2 U445 ( .A(n421), .B(n420), .C(n419), .ZN(n429) );
  VHSR_IAO21_2 U446 ( .A1(n423), .A2(n422), .B(n429), .ZN(n428) );
  VHSR_XNOR2_2 U447 ( .A1(n427), .A2(n428), .ZN(n424) );
  VHSR_CLKNAND2_2 U448 ( .A1(n425), .A2(n424), .ZN(n467) );
  VHSR_OAI21_2 U449 ( .A1(n425), .A2(n424), .B(n467), .ZN(n426) );
  VHSR_NOR2_1 U450 ( .A1(n430), .A2(n429), .ZN(n468) );
  VHSR_AND3_2 U451 ( .A1(n468), .A2(n431), .A3(n467), .Z(n432) );
  VHSR_NOR2_1 U452 ( .A1(n466), .A2(n432), .ZN(product[15]) );
  VHSR_AD1_1 U453 ( .A(n452), .B(n451), .CI(n450), .CO(n445), .S(product[6])
         );
  VHSR_AD1_1 U454 ( .A(n455), .B(n454), .CI(n453), .CO(n425), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U455 ( .A1(n457), .A2(n456), .ZN(n460) );
  VHSR_OAI21_2 U456 ( .A1(n461), .A2(n459), .B(n460), .ZN(n458) );
  VHSR_OAI31_2 U457 ( .A1(n461), .A2(n460), .A3(n459), .B(n458), .ZN(
        product[1]) );
  VHSR_AOI21_2 U458 ( .A1(n464), .A2(n463), .B(n462), .ZN(product[3]) );
  VHSR_NOR2_1 U459 ( .A1(n466), .A2(n465), .ZN(n469) );
  VHSR_XOR3_2 U460 ( .A1(n469), .A2(n468), .A3(n467), .Z(product[14]) );
  VHSR_AOI21_2 U461 ( .A1(n472), .A2(n471), .B(n470), .ZN(n473) );
  VHSR_IN_2 U462 ( .I(n473), .ZN(n475) );
  VHSR_AOI21_2 U463 ( .A1(n476), .A2(n475), .B(n474), .ZN(product[4]) );
endmodule


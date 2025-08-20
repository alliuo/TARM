
module mul8_70 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[4] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , \intadd_0/SUM[0] , n252, n253, n254, n255, n256,
         n257, n258, n259, n260, n261, n262, n263, n264, n265, n266, n267,
         n268, n269, n270, n271, n272, n273, n274, n275, n276, n277, n278,
         n279, n280, n281, n282, n283, n284, n285, n286, n287, n288, n289,
         n290, n291, n292, n293, n294, n295, n296, n297, n298, n299, n300,
         n301, n302, n303, n304, n305, n306, n307, n308, n309, n310, n311,
         n312, n313, n314, n315, n316, n317, n318, n319, n320, n321, n322,
         n323, n324, n325, n326, n327, n328, n329, n330, n331, n332, n333,
         n334, n335, n336, n337, n338, n339, n340, n341, n342, n343, n344,
         n345, n346, n347, n348, n349, n350, n351, n352, n353, n354, n355,
         n356, n357, n358, n359, n360, n361, n362, n363, n364, n365, n366,
         n367, n368, n369, n370, n371, n372, n373, n374, n375, n376, n377,
         n378, n379, n380, n381, n382, n383, n384, n385, n386, n387, n388,
         n389, n390, n391, n392, n393, n394, n395, n396, n397, n398, n399,
         n400, n401, n402, n403, n404, n405, n406, n407, n408, n409, n410,
         n411, n412, n413, n414, n415, n416, n417, n418, n419, n420, n421,
         n422, n423, n424, n425, n426, n427, n428, n429, n430, n431, n432,
         n433, n434, n435, n436, n437, n438, n439, n440, n441, n442, n443,
         n444, n445, n446, n447, n448, n449, n450, n451, n452, n453, n454,
         n455, n456, n457, n458, n459, n460, n461, n462, n463, n464, n465,
         n466, n467, n468, n469, n470, n471, n472;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[9] = \intadd_0/SUM[4] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;
  assign product[5] = \intadd_0/SUM[0] ;

  VHSR_INOR3_2 U241 ( .A1(b[3]), .B1(n379), .B2(n456), .ZN(n380) );
  VHSR_INOR2_2 U242 ( .A1(n281), .B1(n274), .ZN(n277) );
  VHSR_INOR2_2 U243 ( .A1(n420), .B1(n332), .ZN(n336) );
  VHSR_INOR3_2 U244 ( .A1(product[0]), .B1(n461), .B2(n456), .ZN(n373) );
  VHSR_NOR2_1 U245 ( .A1(n376), .A2(n326), .ZN(n286) );
  VHSR_INOR2_2 U246 ( .A1(n270), .B1(n297), .ZN(n288) );
  VHSR_NOR2_1 U247 ( .A1(n457), .A2(n375), .ZN(n378) );
  VHSR_INAND3_2 U248 ( .A1(n403), .B1(n390), .B2(n399), .ZN(n396) );
  VHSR_NOR2_1 U249 ( .A1(n344), .A2(n348), .ZN(n343) );
  VHSR_INOR2_2 U250 ( .A1(n428), .B1(n427), .ZN(n430) );
  VHSR_NOR2_1 U251 ( .A1(n463), .A2(n464), .ZN(n462) );
  VHSR_NOR2_1 U252 ( .A1(n362), .A2(n404), .ZN(n443) );
  VHSR_IN_2 U253 ( .I(n426), .ZN(product[13]) );
  VHSR_NOR2_2 U254 ( .A1(n472), .A2(n471), .ZN(n470) );
  VHSR_INOR2_1 U255 ( .A1(n416), .B1(n415), .ZN(n427) );
  VHSR_INAND2_1 U256 ( .A1(n267), .B1(n266), .ZN(n268) );
  VHSR_INOR2_1 U257 ( .A1(n443), .B1(n334), .ZN(n337) );
  VHSR_NOR2_2 U258 ( .A1(n376), .A2(n459), .ZN(n379) );
  VHSR_NOR2_2 U259 ( .A1(n376), .A2(n375), .ZN(n397) );
  VHSR_NOR2_2 U260 ( .A1(n326), .A2(n404), .ZN(n335) );
  VHSR_AND2_2 U261 ( .A1(a[4]), .A2(b[6]), .Z(n325) );
  VHSR_AD1_1 U262 ( .A(n444), .B(n443), .CI(n442), .CO(n439), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U263 ( .A(n438), .B(n437), .CI(n436), .CO(n433), .S(product[10])
         );
  VHSR_AD1_1 U264 ( .A(n449), .B(n470), .CI(n448), .CO(n450), .S(
        \intadd_0/SUM[0] ) );
  VHSR_AD1_1 U265 ( .A(n447), .B(n446), .CI(n445), .CO(n442), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U266 ( .A(n441), .B(n440), .CI(n439), .CO(n436), .S(
        \intadd_0/SUM[4] ) );
  VHSR_AD1_1 U267 ( .A(n435), .B(n434), .CI(n433), .CO(n453), .S(product[11])
         );
  VHSR_IN_2 U268 ( .I(b[2]), .ZN(n376) );
  VHSR_IN_2 U269 ( .I(a[0]), .ZN(n459) );
  VHSR_IN_2 U270 ( .I(b[0]), .ZN(n457) );
  VHSR_IN_2 U271 ( .I(a[2]), .ZN(n375) );
  VHSR_NOR2_1 U272 ( .A1(n457), .A2(n459), .ZN(product[0]) );
  VHSR_IN_2 U273 ( .I(a[1]), .ZN(n456) );
  VHSR_IN_2 U274 ( .I(b[1]), .ZN(n461) );
  VHSR_NOR3_2 U275 ( .A1(product[0]), .A2(n456), .A3(n461), .ZN(n252) );
  VHSR_MAOI222_2 U276 ( .A(n379), .B(n378), .C(n252), .ZN(n464) );
  VHSR_OAI31_2 U277 ( .A1(n379), .A2(n378), .A3(n252), .B(n464), .ZN(n253) );
  VHSR_IN_2 U278 ( .I(n253), .ZN(product[2]) );
  VHSR_IN_2 U279 ( .I(b[7]), .ZN(n289) );
  VHSR_CLKNAND2_2 U280 ( .A1(b[6]), .A2(a[0]), .ZN(n320) );
  VHSR_NOR3_2 U281 ( .A1(n289), .A2(n320), .A3(n456), .ZN(n269) );
  VHSR_IN_2 U282 ( .I(b[4]), .ZN(n404) );
  VHSR_IN_2 U283 ( .I(b[5]), .ZN(n360) );
  VHSR_IN_2 U284 ( .I(a[3]), .ZN(n377) );
  VHSR_NOR4_2 U285 ( .A1(n404), .A2(n360), .A3(n377), .A4(n375), .ZN(n267) );
  VHSR_CLKNAND2_2 U286 ( .A1(b[7]), .A2(a[2]), .ZN(n255) );
  VHSR_AOI21_2 U287 ( .A1(b[6]), .A2(a[3]), .B(n255), .ZN(n254) );
  VHSR_AOI31_2 U288 ( .A1(b[6]), .A2(n255), .A3(a[3]), .B(n254), .ZN(n266) );
  VHSR_IN_2 U289 ( .I(n266), .ZN(n256) );
  VHSR_MAOI222_2 U290 ( .A(n269), .B(n267), .C(n256), .ZN(n270) );
  VHSR_CLKNAND2_2 U291 ( .A1(b[6]), .A2(a[2]), .ZN(n293) );
  VHSR_CLKNAND2_2 U292 ( .A1(b[4]), .A2(a[2]), .ZN(n319) );
  VHSR_NAND3_2 U293 ( .A1(a[3]), .A2(b[5]), .A3(n319), .ZN(n261) );
  VHSR_NAND3_2 U294 ( .A1(b[7]), .A2(a[1]), .A3(n320), .ZN(n263) );
  VHSR_MAOI222_2 U295 ( .A(n293), .B(n261), .C(n263), .ZN(n265) );
  VHSR_OAI211_2 U296 ( .A1(n404), .A2(n459), .B(b[5]), .C(a[1]), .ZN(n318) );
  VHSR_MAOI222_2 U297 ( .A(n320), .B(n319), .C(n318), .ZN(n317) );
  VHSR_NOR4_2 U298 ( .A1(n404), .A2(n360), .A3(n459), .A4(n456), .ZN(n324) );
  VHSR_AOI22_2 U299 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n257) );
  VHSR_NOR2_1 U300 ( .A1(n267), .A2(n257), .ZN(n260) );
  VHSR_AOI22_2 U301 ( .A1(b[6]), .A2(a[1]), .B1(b[7]), .B2(a[0]), .ZN(n258) );
  VHSR_NOR2_1 U302 ( .A1(n269), .A2(n258), .ZN(n259) );
  VHSR_AND2_2 U303 ( .A1(n317), .A2(n312), .Z(n311) );
  VHSR_AD1_1 U304 ( .A(n324), .B(n260), .CI(n259), .CO(n304), .S(n312) );
  VHSR_NOR2_1 U305 ( .A1(n311), .A2(n304), .ZN(n307) );
  VHSR_AND2_2 U306 ( .A1(n293), .A2(n261), .Z(n262) );
  VHSR_AOI21_2 U307 ( .A1(n263), .A2(n262), .B(n265), .ZN(n264) );
  VHSR_IN_2 U308 ( .I(n264), .ZN(n308) );
  VHSR_NOR2_1 U309 ( .A1(n307), .A2(n308), .ZN(n305) );
  VHSR_NOR2_1 U310 ( .A1(n265), .A2(n305), .ZN(n299) );
  VHSR_OAI21_2 U311 ( .A1(n269), .A2(n268), .B(n270), .ZN(n298) );
  VHSR_NOR2_1 U312 ( .A1(n299), .A2(n298), .ZN(n297) );
  VHSR_AOI211_2 U313 ( .A1(n288), .A2(n293), .B(n377), .C(n289), .ZN(n347) );
  VHSR_NAND4_2 U314 ( .A1(b[3]), .A2(b[2]), .A3(a[4]), .A4(a[5]), .ZN(n281) );
  VHSR_CLKNAND2_2 U315 ( .A1(b[3]), .A2(a[6]), .ZN(n272) );
  VHSR_AOI21_2 U316 ( .A1(a[7]), .A2(b[2]), .B(n272), .ZN(n271) );
  VHSR_AOI31_2 U317 ( .A1(a[7]), .A2(n272), .A3(b[2]), .B(n271), .ZN(n280) );
  VHSR_NOR2_1 U318 ( .A1(n281), .A2(n280), .ZN(n282) );
  VHSR_IN_2 U319 ( .I(a[6]), .ZN(n326) );
  VHSR_CLKNAND2_2 U320 ( .A1(b[2]), .A2(a[4]), .ZN(n316) );
  VHSR_AND3_2 U321 ( .A1(n316), .A2(b[3]), .A3(a[5]), .Z(n279) );
  VHSR_IN_2 U322 ( .I(a[7]), .ZN(n275) );
  VHSR_NOR2_1 U323 ( .A1(n275), .A2(n461), .ZN(n278) );
  VHSR_IN_2 U324 ( .I(n273), .ZN(n296) );
  VHSR_IN_2 U325 ( .I(a[4]), .ZN(n362) );
  VHSR_OAI211_2 U326 ( .A1(n362), .A2(n457), .B(a[5]), .C(b[1]), .ZN(n315) );
  VHSR_CLKNAND2_2 U327 ( .A1(a[6]), .A2(b[0]), .ZN(n314) );
  VHSR_MAOI222_2 U328 ( .A(n316), .B(n315), .C(n314), .ZN(n313) );
  VHSR_IN_2 U329 ( .I(a[5]), .ZN(n358) );
  VHSR_NOR4_2 U330 ( .A1(n362), .A2(n358), .A3(n461), .A4(n457), .ZN(n322) );
  VHSR_AOI22_2 U331 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n274) );
  VHSR_OAI22_2 U332 ( .A1(n275), .A2(n457), .B1(n326), .B2(n461), .ZN(n276) );
  VHSR_AND2_2 U333 ( .A1(n313), .A2(n310), .Z(n309) );
  VHSR_AD1_1 U334 ( .A(n322), .B(n277), .CI(n276), .CO(n301), .S(n310) );
  VHSR_AD1_1 U335 ( .A(n286), .B(n279), .CI(n278), .CO(n273), .S(n300) );
  VHSR_OAI21_2 U336 ( .A1(n309), .A2(n301), .B(n300), .ZN(n303) );
  VHSR_XNOR2_2 U337 ( .A1(n281), .A2(n280), .ZN(n295) );
  VHSR_MAOI222_2 U338 ( .A(n296), .B(n303), .C(n295), .ZN(n294) );
  VHSR_OR2_2 U339 ( .A1(n282), .A2(n294), .Z(n285) );
  VHSR_OAI211_2 U340 ( .A1(n285), .A2(n286), .B(b[3]), .C(a[7]), .ZN(n283) );
  VHSR_IN_2 U341 ( .I(n283), .ZN(n346) );
  VHSR_CLKNAND2_2 U342 ( .A1(a[7]), .A2(b[3]), .ZN(n287) );
  VHSR_OAI21_2 U343 ( .A1(n287), .A2(n286), .B(n285), .ZN(n284) );
  VHSR_OAI31_2 U344 ( .A1(n287), .A2(n286), .A3(n285), .B(n284), .ZN(n354) );
  VHSR_IN_2 U345 ( .I(n288), .ZN(n292) );
  VHSR_NOR2_1 U346 ( .A1(n289), .A2(n377), .ZN(n291) );
  VHSR_AOI21_2 U347 ( .A1(n293), .A2(n291), .B(n292), .ZN(n290) );
  VHSR_AOI31_2 U348 ( .A1(n293), .A2(n292), .A3(n291), .B(n290), .ZN(n353) );
  VHSR_AOI31_2 U349 ( .A1(n296), .A2(n303), .A3(n295), .B(n294), .ZN(n357) );
  VHSR_AOI21_2 U350 ( .A1(n299), .A2(n298), .B(n297), .ZN(n356) );
  VHSR_OAI32_2 U351 ( .A1(n301), .A2(n300), .A3(n309), .B1(n303), .B2(n301), 
        .ZN(n302) );
  VHSR_IAO21_2 U352 ( .A1(n309), .A2(n303), .B(n302), .ZN(n365) );
  VHSR_CLKNAND2_2 U353 ( .A1(n311), .A2(n304), .ZN(n306) );
  VHSR_AOI22_2 U354 ( .A1(n308), .A2(n307), .B1(n306), .B2(n305), .ZN(n364) );
  VHSR_IAO21_2 U355 ( .A1(n313), .A2(n310), .B(n309), .ZN(n368) );
  VHSR_IAO21_2 U356 ( .A1(n317), .A2(n312), .B(n311), .ZN(n367) );
  VHSR_AOI31_2 U357 ( .A1(n316), .A2(n315), .A3(n314), .B(n313), .ZN(n394) );
  VHSR_AOI31_2 U358 ( .A1(n320), .A2(n319), .A3(n318), .B(n317), .ZN(n393) );
  VHSR_CLKNAND2_2 U359 ( .A1(a[5]), .A2(b[0]), .ZN(n321) );
  VHSR_OAI32_2 U360 ( .A1(n322), .A2(n461), .A3(n362), .B1(n321), .B2(n322), 
        .ZN(n414) );
  VHSR_CLKNAND2_2 U361 ( .A1(n443), .A2(product[0]), .ZN(n406) );
  VHSR_IN_2 U362 ( .I(n406), .ZN(n413) );
  VHSR_CLKNAND2_2 U363 ( .A1(b[5]), .A2(a[0]), .ZN(n323) );
  VHSR_OAI32_2 U364 ( .A1(n324), .A2(n456), .A3(n404), .B1(n323), .B2(n324), 
        .ZN(n412) );
  VHSR_CLKNAND2_2 U365 ( .A1(a[6]), .A2(b[6]), .ZN(n431) );
  VHSR_IN_2 U366 ( .I(n431), .ZN(n465) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[5]), .A2(b[7]), .ZN(n328) );
  VHSR_CLKNAND2_2 U368 ( .A1(a[7]), .A2(b[5]), .ZN(n327) );
  VHSR_OAI22_2 U369 ( .A1(n325), .A2(n328), .B1(n335), .B2(n327), .ZN(n330) );
  VHSR_OR2_2 U370 ( .A1(n335), .A2(n325), .Z(n349) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[5]), .A2(b[5]), .ZN(n334) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[7]), .ZN(n466) );
  VHSR_NOR3_2 U373 ( .A1(n349), .A2(n334), .A3(n466), .ZN(n329) );
  VHSR_AOI31_2 U374 ( .A1(b[6]), .A2(a[6]), .A3(n330), .B(n329), .ZN(n416) );
  VHSR_OAI21_2 U375 ( .A1(n465), .A2(n330), .B(n416), .ZN(n342) );
  VHSR_NAND3_2 U376 ( .A1(a[7]), .A2(n335), .A3(b[5]), .ZN(n421) );
  VHSR_IN_2 U377 ( .I(n421), .ZN(n423) );
  VHSR_AOI22_2 U378 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n331) );
  VHSR_NOR2_1 U379 ( .A1(n423), .A2(n331), .ZN(n338) );
  VHSR_NAND4_2 U380 ( .A1(a[4]), .A2(a[5]), .A3(b[6]), .A4(b[7]), .ZN(n420) );
  VHSR_AOI22_2 U381 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n332) );
  VHSR_IN_2 U382 ( .I(n333), .ZN(n344) );
  VHSR_NOR2_1 U383 ( .A1(n443), .A2(n334), .ZN(n350) );
  VHSR_AOI22_2 U384 ( .A1(n335), .A2(n325), .B1(n350), .B2(n349), .ZN(n348) );
  VHSR_AD1_1 U385 ( .A(n338), .B(n337), .CI(n336), .CO(n339), .S(n333) );
  VHSR_NOR2_1 U386 ( .A1(n343), .A2(n339), .ZN(n341) );
  VHSR_CLKNAND2_2 U387 ( .A1(n343), .A2(n339), .ZN(n340) );
  VHSR_NOR2_1 U388 ( .A1(n341), .A2(n342), .ZN(n415) );
  VHSR_AOI22_2 U389 ( .A1(n342), .A2(n341), .B1(n340), .B2(n415), .ZN(n454) );
  VHSR_AOI21_2 U390 ( .A1(n348), .A2(n344), .B(n343), .ZN(n435) );
  VHSR_AD1_1 U391 ( .A(n347), .B(n346), .CI(n345), .CO(n455), .S(n434) );
  VHSR_OAI21_2 U392 ( .A1(n350), .A2(n349), .B(n348), .ZN(n351) );
  VHSR_IN_2 U393 ( .I(n351), .ZN(n438) );
  VHSR_AD1_1 U394 ( .A(n354), .B(n353), .CI(n352), .CO(n345), .S(n437) );
  VHSR_AD1_1 U395 ( .A(n357), .B(n356), .CI(n355), .CO(n352), .S(n441) );
  VHSR_NOR2_1 U396 ( .A1(n358), .A2(n404), .ZN(n361) );
  VHSR_OAI21_2 U397 ( .A1(n362), .A2(n360), .B(n361), .ZN(n359) );
  VHSR_OAI31_2 U398 ( .A1(n362), .A2(n361), .A3(n360), .B(n359), .ZN(n440) );
  VHSR_AD1_1 U399 ( .A(n365), .B(n364), .CI(n363), .CO(n355), .S(n444) );
  VHSR_AD1_1 U400 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(n447) );
  VHSR_NAND3_2 U401 ( .A1(b[3]), .A2(a[1]), .A3(n379), .ZN(n384) );
  VHSR_IN_2 U402 ( .I(n384), .ZN(n386) );
  VHSR_AOI22_2 U403 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n369) );
  VHSR_NOR2_1 U404 ( .A1(n386), .A2(n369), .ZN(n372) );
  VHSR_NAND3_2 U405 ( .A1(b[1]), .A2(a[3]), .A3(n378), .ZN(n383) );
  VHSR_IN_2 U406 ( .I(n383), .ZN(n385) );
  VHSR_AOI22_2 U407 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n370) );
  VHSR_NOR2_1 U408 ( .A1(n385), .A2(n370), .ZN(n371) );
  VHSR_AD1_1 U409 ( .A(n373), .B(n372), .CI(n371), .CO(n410), .S(n374) );
  VHSR_IN_2 U410 ( .I(n374), .ZN(n463) );
  VHSR_NOR3_2 U411 ( .A1(n378), .A2(n377), .A3(n461), .ZN(n381) );
  VHSR_OAI21_2 U412 ( .A1(n410), .A2(n462), .B(n408), .ZN(n411) );
  VHSR_IN_2 U413 ( .I(n411), .ZN(n407) );
  VHSR_AD1_1 U414 ( .A(n397), .B(n381), .CI(n380), .CO(n389), .S(n408) );
  VHSR_NOR2_1 U415 ( .A1(n407), .A2(n389), .ZN(n402) );
  VHSR_AOI22_2 U416 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n382) );
  VHSR_AOI31_2 U417 ( .A1(a[3]), .A2(b[3]), .A3(n397), .B(n382), .ZN(n388) );
  VHSR_CLKNAND2_2 U418 ( .A1(n384), .A2(n383), .ZN(n387) );
  VHSR_MAOI222_2 U419 ( .A(n386), .B(n385), .C(n388), .ZN(n390) );
  VHSR_OAI21_2 U420 ( .A1(n388), .A2(n387), .B(n390), .ZN(n400) );
  VHSR_NOR2_1 U421 ( .A1(n402), .A2(n400), .ZN(n403) );
  VHSR_CLKNAND2_2 U422 ( .A1(n389), .A2(n407), .ZN(n399) );
  VHSR_OAI211_2 U423 ( .A1(n396), .A2(n397), .B(a[3]), .C(b[3]), .ZN(n391) );
  VHSR_IN_2 U424 ( .I(n391), .ZN(n446) );
  VHSR_AD1_1 U425 ( .A(n394), .B(n393), .CI(n392), .CO(n366), .S(n452) );
  VHSR_CLKNAND2_2 U426 ( .A1(b[3]), .A2(a[3]), .ZN(n398) );
  VHSR_OAI21_2 U427 ( .A1(n398), .A2(n397), .B(n396), .ZN(n395) );
  VHSR_OAI31_2 U428 ( .A1(n398), .A2(n397), .A3(n396), .B(n395), .ZN(n451) );
  VHSR_IN_2 U429 ( .I(n399), .ZN(n401) );
  VHSR_OAI32_2 U430 ( .A1(n403), .A2(n402), .A3(n401), .B1(n400), .B2(n403), 
        .ZN(n449) );
  VHSR_NOR2_1 U431 ( .A1(n404), .A2(n459), .ZN(n405) );
  VHSR_AOI32_2 U432 ( .A1(b[0]), .A2(n406), .A3(a[4]), .B1(n405), .B2(n406), 
        .ZN(n472) );
  VHSR_IAO21_2 U433 ( .A1(n408), .A2(n462), .B(n407), .ZN(n409) );
  VHSR_OAI22_2 U434 ( .A1(n462), .A2(n411), .B1(n410), .B2(n409), .ZN(n471) );
  VHSR_AD1_1 U435 ( .A(n414), .B(n413), .CI(n412), .CO(n392), .S(n448) );
  VHSR_CLKNAND2_2 U436 ( .A1(a[6]), .A2(b[7]), .ZN(n418) );
  VHSR_AOI21_2 U437 ( .A1(a[7]), .A2(b[6]), .B(n418), .ZN(n417) );
  VHSR_AOI31_2 U438 ( .A1(a[7]), .A2(n418), .A3(b[6]), .B(n417), .ZN(n419) );
  VHSR_CLKNAND2_2 U439 ( .A1(n420), .A2(n419), .ZN(n422) );
  VHSR_MAOI222_2 U440 ( .A(n421), .B(n420), .C(n419), .ZN(n429) );
  VHSR_IAO21_2 U441 ( .A1(n423), .A2(n422), .B(n429), .ZN(n428) );
  VHSR_XNOR2_2 U442 ( .A1(n427), .A2(n428), .ZN(n424) );
  VHSR_CLKNAND2_2 U443 ( .A1(n425), .A2(n424), .ZN(n467) );
  VHSR_OAI21_2 U444 ( .A1(n425), .A2(n424), .B(n467), .ZN(n426) );
  VHSR_NOR2_1 U445 ( .A1(n430), .A2(n429), .ZN(n468) );
  VHSR_AND3_2 U446 ( .A1(n468), .A2(n431), .A3(n467), .Z(n432) );
  VHSR_NOR2_1 U447 ( .A1(n466), .A2(n432), .ZN(product[15]) );
  VHSR_AD1_1 U448 ( .A(n452), .B(n451), .CI(n450), .CO(n445), .S(product[6])
         );
  VHSR_AD1_1 U449 ( .A(n455), .B(n454), .CI(n453), .CO(n425), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U450 ( .A1(n457), .A2(n456), .ZN(n460) );
  VHSR_OAI21_2 U451 ( .A1(n461), .A2(n459), .B(n460), .ZN(n458) );
  VHSR_OAI31_2 U452 ( .A1(n461), .A2(n460), .A3(n459), .B(n458), .ZN(
        product[1]) );
  VHSR_AOI21_2 U453 ( .A1(n464), .A2(n463), .B(n462), .ZN(product[3]) );
  VHSR_NOR2_1 U454 ( .A1(n466), .A2(n465), .ZN(n469) );
  VHSR_XOR3_2 U455 ( .A1(n469), .A2(n468), .A3(n467), .Z(product[14]) );
  VHSR_AOI21_2 U456 ( .A1(n472), .A2(n471), .B(n470), .ZN(product[4]) );
endmodule


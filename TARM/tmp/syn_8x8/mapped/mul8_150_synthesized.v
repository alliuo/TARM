
module mul8_150 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n246, n247, n248, n249, n250, n251, n252, n253,
         n254, n255, n256, n257, n258, n259, n260, n261, n262, n263, n264,
         n265, n266, n267, n268, n269, n270, n271, n272, n273, n274, n275,
         n276, n277, n278, n279, n280, n281, n282, n283, n284, n285, n286,
         n287, n288, n289, n290, n291, n292, n293, n294, n295, n296, n297,
         n298, n299, n300, n301, n302, n303, n304, n305, n306, n307, n308,
         n309, n310, n311, n312, n313, n314, n315, n316, n317, n318, n319,
         n320, n321, n322, n323, n324, n325, n326, n327, n328, n329, n330,
         n331, n332, n333, n334, n335, n336, n337, n338, n339, n340, n341,
         n342, n343, n344, n345, n346, n347, n348, n349, n350, n351, n352,
         n353, n354, n355, n356, n357, n358, n359, n360, n361, n362, n363,
         n364, n365, n366, n367, n368, n369, n370, n371, n372, n373, n374,
         n375, n376, n377, n378, n379, n380, n381, n382, n383, n384, n385,
         n386, n387, n388, n389, n390, n391, n392, n393, n394, n395, n396,
         n397, n398, n399, n400, n401, n402, n403, n404, n405, n406, n407,
         n408, n409, n410, n411, n412, n413, n414, n415, n416, n417, n418,
         n419, n420, n421, n422, n423, n424, n425, n426, n427, n428, n429,
         n430, n431, n432, n433, n434, n435, n436, n437, n438, n439, n440,
         n441, n442, n443, n444, n445, n446, n447, n448, n449, n450, n451,
         n452, n453, n454, n455, n456, n457, n458, n459, n460, n461, n462,
         n463;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U237 ( .A1(n273), .B1(n271), .ZN(n274) );
  VHSR_INOR2_2 U238 ( .A1(n407), .B1(n326), .ZN(n332) );
  VHSR_NOR2_1 U239 ( .A1(n291), .A2(n290), .ZN(n289) );
  VHSR_NOR2_1 U240 ( .A1(n288), .A2(n284), .ZN(n282) );
  VHSR_NOR2_1 U241 ( .A1(n328), .A2(n368), .ZN(n450) );
  VHSR_NOR2_1 U242 ( .A1(n340), .A2(n344), .ZN(n339) );
  VHSR_NOR2_1 U243 ( .A1(n337), .A2(n338), .ZN(n402) );
  VHSR_IN_2 U244 ( .I(n368), .ZN(product[0]) );
  VHSR_IN_2 U245 ( .I(n413), .ZN(product[13]) );
  VHSR_AD1_2 U246 ( .A(n343), .B(n342), .CI(n341), .CO(n444), .S(n421) );
  VHSR_INOR3_1 U247 ( .A1(n282), .B1(n389), .B2(n324), .ZN(n343) );
  VHSR_NOR2_2 U248 ( .A1(n286), .A2(n285), .ZN(n284) );
  VHSR_NOR2_2 U249 ( .A1(n417), .A2(n416), .ZN(n448) );
  VHSR_INAND2_1 U250 ( .A1(n289), .B1(n276), .ZN(n279) );
  VHSR_INOR2_1 U251 ( .A1(n415), .B1(n414), .ZN(n417) );
  VHSR_CLKN_1 U252 ( .I(n392), .ZN(n384) );
  VHSR_INOR2_1 U253 ( .A1(n403), .B1(n402), .ZN(n414) );
  VHSR_INAND2_1 U254 ( .A1(n394), .B1(n383), .ZN(n392) );
  VHSR_NOR2_2 U255 ( .A1(n339), .A2(n335), .ZN(n337) );
  VHSR_INOR2_1 U256 ( .A1(n397), .B1(n374), .ZN(n396) );
  VHSR_MOAI22_1 U257 ( .A1(n324), .A2(n460), .B1(a[6]), .B2(b[2]), .ZN(n249)
         );
  VHSR_INOR3_1 U258 ( .A1(b[7]), .B1(n457), .B2(n309), .ZN(n275) );
  VHSR_AD1_1 U259 ( .A(n436), .B(n454), .CI(n435), .CO(n432), .S(product[5])
         );
  VHSR_AD1_1 U260 ( .A(n431), .B(n430), .CI(n429), .CO(n426), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U261 ( .A(n425), .B(n424), .CI(n423), .CO(n420), .S(product[10])
         );
  VHSR_AD1_1 U262 ( .A(n438), .B(n437), .CI(n461), .CO(n399), .S(product[3])
         );
  VHSR_AD1_1 U263 ( .A(n434), .B(n433), .CI(n432), .CO(n439), .S(product[6])
         );
  VHSR_AD1_1 U264 ( .A(n428), .B(n427), .CI(n426), .CO(n423), .S(product[9])
         );
  VHSR_AD1_1 U265 ( .A(n422), .B(n421), .CI(n420), .CO(n442), .S(
        \intadd_0/SUM[6] ) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[0]), .A2(a[0]), .ZN(n368) );
  VHSR_AOI22_2 U267 ( .A1(b[3]), .A2(a[6]), .B1(a[7]), .B2(b[2]), .ZN(n288) );
  VHSR_IN_2 U268 ( .I(b[3]), .ZN(n389) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[2]), .A2(a[4]), .ZN(n313) );
  VHSR_IN_2 U270 ( .I(a[5]), .ZN(n355) );
  VHSR_NOR3_2 U271 ( .A1(n389), .A2(n313), .A3(n355), .ZN(n286) );
  VHSR_AOI211_2 U272 ( .A1(a[4]), .A2(b[2]), .B(n389), .C(n355), .ZN(n248) );
  VHSR_AND2_2 U273 ( .A1(a[6]), .A2(b[2]), .Z(n247) );
  VHSR_IN_2 U274 ( .I(a[7]), .ZN(n324) );
  VHSR_IN_2 U275 ( .I(b[1]), .ZN(n460) );
  VHSR_NOR2_1 U276 ( .A1(n324), .A2(n460), .ZN(n246) );
  VHSR_MAOI222_2 U277 ( .A(n248), .B(n247), .C(n246), .ZN(n258) );
  VHSR_OAI21_2 U278 ( .A1(n249), .A2(n248), .B(n258), .ZN(n250) );
  VHSR_IN_2 U279 ( .I(n250), .ZN(n299) );
  VHSR_CLKNAND2_2 U280 ( .A1(a[6]), .A2(b[1]), .ZN(n255) );
  VHSR_IN_2 U281 ( .I(n255), .ZN(n252) );
  VHSR_IN_2 U282 ( .I(a[4]), .ZN(n359) );
  VHSR_IN_2 U283 ( .I(b[0]), .ZN(n458) );
  VHSR_NOR4_2 U284 ( .A1(n359), .A2(n355), .A3(n460), .A4(n458), .ZN(n317) );
  VHSR_AOI22_2 U285 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n251) );
  VHSR_NOR2_1 U286 ( .A1(n286), .A2(n251), .ZN(n253) );
  VHSR_MAOI222_2 U287 ( .A(n252), .B(n317), .C(n253), .ZN(n257) );
  VHSR_OAI21_2 U288 ( .A1(a[7]), .A2(a[6]), .B(b[0]), .ZN(n312) );
  VHSR_OAI211_2 U289 ( .A1(n359), .A2(n458), .B(a[5]), .C(b[1]), .ZN(n311) );
  VHSR_MAOI222_2 U290 ( .A(n313), .B(n312), .C(n311), .ZN(n310) );
  VHSR_NOR2_1 U291 ( .A1(n317), .A2(n253), .ZN(n256) );
  VHSR_IN_2 U292 ( .I(n257), .ZN(n254) );
  VHSR_AOI21_2 U293 ( .A1(n256), .A2(n255), .B(n254), .ZN(n304) );
  VHSR_CLKNAND2_2 U294 ( .A1(n310), .A2(n304), .ZN(n303) );
  VHSR_CLKNAND2_2 U295 ( .A1(n257), .A2(n303), .ZN(n298) );
  VHSR_CLKNAND2_2 U296 ( .A1(n299), .A2(n298), .ZN(n297) );
  VHSR_CLKNAND2_2 U297 ( .A1(n258), .A2(n297), .ZN(n285) );
  VHSR_CLKNAND2_2 U298 ( .A1(b[6]), .A2(a[2]), .ZN(n263) );
  VHSR_CLKNAND2_2 U299 ( .A1(b[6]), .A2(a[0]), .ZN(n309) );
  VHSR_NAND3_2 U300 ( .A1(a[1]), .A2(b[7]), .A3(n309), .ZN(n265) );
  VHSR_CLKNAND2_2 U301 ( .A1(b[4]), .A2(a[2]), .ZN(n308) );
  VHSR_NAND3_2 U302 ( .A1(a[3]), .A2(b[5]), .A3(n308), .ZN(n264) );
  VHSR_MAOI222_2 U303 ( .A(n263), .B(n265), .C(n264), .ZN(n268) );
  VHSR_CLKNAND2_2 U304 ( .A1(b[4]), .A2(a[0]), .ZN(n451) );
  VHSR_NAND3_2 U305 ( .A1(a[1]), .A2(b[5]), .A3(n451), .ZN(n307) );
  VHSR_MAOI222_2 U306 ( .A(n309), .B(n308), .C(n307), .ZN(n306) );
  VHSR_IN_2 U307 ( .I(b[4]), .ZN(n354) );
  VHSR_IN_2 U308 ( .I(b[5]), .ZN(n357) );
  VHSR_IN_2 U309 ( .I(a[3]), .ZN(n388) );
  VHSR_IN_2 U310 ( .I(a[2]), .ZN(n370) );
  VHSR_NOR4_2 U311 ( .A1(n354), .A2(n357), .A3(n388), .A4(n370), .ZN(n273) );
  VHSR_AOI22_2 U312 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n259) );
  VHSR_NOR2_1 U313 ( .A1(n273), .A2(n259), .ZN(n262) );
  VHSR_IN_2 U314 ( .I(a[1]), .ZN(n457) );
  VHSR_NOR3_2 U315 ( .A1(n357), .A2(n457), .A3(n451), .ZN(n315) );
  VHSR_AOI22_2 U316 ( .A1(b[6]), .A2(a[1]), .B1(b[7]), .B2(a[0]), .ZN(n260) );
  VHSR_NOR2_1 U317 ( .A1(n275), .A2(n260), .ZN(n261) );
  VHSR_AND2_2 U318 ( .A1(n306), .A2(n302), .Z(n301) );
  VHSR_AD1_1 U319 ( .A(n262), .B(n315), .CI(n261), .CO(n292), .S(n302) );
  VHSR_NOR2_1 U320 ( .A1(n301), .A2(n292), .ZN(n295) );
  VHSR_IN_2 U321 ( .I(n263), .ZN(n280) );
  VHSR_CLKNAND2_2 U322 ( .A1(n265), .A2(n264), .ZN(n267) );
  VHSR_IN_2 U323 ( .I(n268), .ZN(n266) );
  VHSR_OAI21_2 U324 ( .A1(n280), .A2(n267), .B(n266), .ZN(n296) );
  VHSR_NOR2_1 U325 ( .A1(n295), .A2(n296), .ZN(n293) );
  VHSR_NOR2_1 U326 ( .A1(n268), .A2(n293), .ZN(n291) );
  VHSR_CLKNAND2_2 U327 ( .A1(b[7]), .A2(a[2]), .ZN(n270) );
  VHSR_AOI21_2 U328 ( .A1(b[6]), .A2(a[3]), .B(n270), .ZN(n269) );
  VHSR_AOI31_2 U329 ( .A1(b[6]), .A2(n270), .A3(a[3]), .B(n269), .ZN(n271) );
  VHSR_IN_2 U330 ( .I(n271), .ZN(n272) );
  VHSR_MAOI222_2 U331 ( .A(n275), .B(n273), .C(n272), .ZN(n276) );
  VHSR_OAI21_2 U332 ( .A1(n275), .A2(n274), .B(n276), .ZN(n290) );
  VHSR_OAI211_2 U333 ( .A1(n279), .A2(n280), .B(a[3]), .C(b[7]), .ZN(n277) );
  VHSR_IN_2 U334 ( .I(n277), .ZN(n342) );
  VHSR_CLKNAND2_2 U335 ( .A1(b[7]), .A2(a[3]), .ZN(n281) );
  VHSR_OAI21_2 U336 ( .A1(n281), .A2(n280), .B(n279), .ZN(n278) );
  VHSR_OAI31_2 U337 ( .A1(n281), .A2(n280), .A3(n279), .B(n278), .ZN(n350) );
  VHSR_NOR2_1 U338 ( .A1(n389), .A2(n324), .ZN(n283) );
  VHSR_IAO21_2 U339 ( .A1(n283), .A2(n282), .B(n343), .ZN(n349) );
  VHSR_AOI21_2 U340 ( .A1(n286), .A2(n285), .B(n284), .ZN(n287) );
  VHSR_XNOR2_2 U341 ( .A1(n288), .A2(n287), .ZN(n353) );
  VHSR_AOI21_2 U342 ( .A1(n291), .A2(n290), .B(n289), .ZN(n352) );
  VHSR_CLKNAND2_2 U343 ( .A1(n301), .A2(n292), .ZN(n294) );
  VHSR_AOI22_2 U344 ( .A1(n296), .A2(n295), .B1(n294), .B2(n293), .ZN(n362) );
  VHSR_OAI21_2 U345 ( .A1(n299), .A2(n298), .B(n297), .ZN(n300) );
  VHSR_IN_2 U346 ( .I(n300), .ZN(n361) );
  VHSR_IAO21_2 U347 ( .A1(n306), .A2(n302), .B(n301), .ZN(n365) );
  VHSR_OAI21_2 U348 ( .A1(n310), .A2(n304), .B(n303), .ZN(n305) );
  VHSR_IN_2 U349 ( .I(n305), .ZN(n364) );
  VHSR_AOI31_2 U350 ( .A1(n309), .A2(n308), .A3(n307), .B(n306), .ZN(n387) );
  VHSR_AOI31_2 U351 ( .A1(n313), .A2(n312), .A3(n311), .B(n310), .ZN(n386) );
  VHSR_AOI22_2 U352 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n314) );
  VHSR_NOR2_1 U353 ( .A1(n315), .A2(n314), .ZN(n401) );
  VHSR_CLKNAND2_2 U354 ( .A1(a[4]), .A2(b[4]), .ZN(n328) );
  VHSR_CLKNAND2_2 U355 ( .A1(a[5]), .A2(b[0]), .ZN(n316) );
  VHSR_OAI32_2 U356 ( .A1(n317), .A2(n460), .A3(n359), .B1(n316), .B2(n317), 
        .ZN(n400) );
  VHSR_CLKNAND2_2 U357 ( .A1(a[6]), .A2(b[6]), .ZN(n418) );
  VHSR_IN_2 U358 ( .I(n418), .ZN(n445) );
  VHSR_CLKNAND2_2 U359 ( .A1(a[4]), .A2(b[6]), .ZN(n320) );
  VHSR_IN_2 U360 ( .I(n320), .ZN(n330) );
  VHSR_CLKNAND2_2 U361 ( .A1(a[5]), .A2(b[7]), .ZN(n319) );
  VHSR_CLKNAND2_2 U362 ( .A1(a[6]), .A2(b[4]), .ZN(n323) );
  VHSR_IN_2 U363 ( .I(n323), .ZN(n331) );
  VHSR_CLKNAND2_2 U364 ( .A1(a[7]), .A2(b[5]), .ZN(n318) );
  VHSR_OAI22_2 U365 ( .A1(n330), .A2(n319), .B1(n331), .B2(n318), .ZN(n322) );
  VHSR_CLKNAND2_2 U366 ( .A1(n323), .A2(n320), .ZN(n345) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[5]), .A2(b[5]), .ZN(n329) );
  VHSR_CLKNAND2_2 U368 ( .A1(a[7]), .A2(b[7]), .ZN(n446) );
  VHSR_NOR3_2 U369 ( .A1(n345), .A2(n329), .A3(n446), .ZN(n321) );
  VHSR_AOI31_2 U370 ( .A1(b[6]), .A2(a[6]), .A3(n322), .B(n321), .ZN(n403) );
  VHSR_OAI21_2 U371 ( .A1(n445), .A2(n322), .B(n403), .ZN(n338) );
  VHSR_NOR3_2 U372 ( .A1(n324), .A2(n323), .A3(n357), .ZN(n410) );
  VHSR_AOI22_2 U373 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n325) );
  VHSR_NOR2_1 U374 ( .A1(n410), .A2(n325), .ZN(n334) );
  VHSR_NOR2_1 U375 ( .A1(n329), .A2(n328), .ZN(n333) );
  VHSR_NAND4_2 U376 ( .A1(a[4]), .A2(a[5]), .A3(b[6]), .A4(b[7]), .ZN(n407) );
  VHSR_AOI22_2 U377 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n326) );
  VHSR_IN_2 U378 ( .I(n327), .ZN(n340) );
  VHSR_IN_2 U379 ( .I(n328), .ZN(n430) );
  VHSR_NOR2_1 U380 ( .A1(n430), .A2(n329), .ZN(n346) );
  VHSR_AOI22_2 U381 ( .A1(n331), .A2(n330), .B1(n346), .B2(n345), .ZN(n344) );
  VHSR_AD1_1 U382 ( .A(n334), .B(n333), .CI(n332), .CO(n335), .S(n327) );
  VHSR_CLKNAND2_2 U383 ( .A1(n339), .A2(n335), .ZN(n336) );
  VHSR_AOI22_2 U384 ( .A1(n338), .A2(n337), .B1(n336), .B2(n402), .ZN(n443) );
  VHSR_AOI21_2 U385 ( .A1(n344), .A2(n340), .B(n339), .ZN(n422) );
  VHSR_OAI21_2 U386 ( .A1(n346), .A2(n345), .B(n344), .ZN(n347) );
  VHSR_IN_2 U387 ( .I(n347), .ZN(n425) );
  VHSR_AD1_1 U388 ( .A(n350), .B(n349), .CI(n348), .CO(n341), .S(n424) );
  VHSR_AD1_1 U389 ( .A(n353), .B(n352), .CI(n351), .CO(n348), .S(n428) );
  VHSR_NOR2_1 U390 ( .A1(n355), .A2(n354), .ZN(n358) );
  VHSR_OAI21_2 U391 ( .A1(n359), .A2(n357), .B(n358), .ZN(n356) );
  VHSR_OAI31_2 U392 ( .A1(n359), .A2(n358), .A3(n357), .B(n356), .ZN(n427) );
  VHSR_AD1_1 U393 ( .A(n362), .B(n361), .CI(n360), .CO(n351), .S(n431) );
  VHSR_AD1_1 U394 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(n441) );
  VHSR_IN_2 U395 ( .I(b[2]), .ZN(n369) );
  VHSR_IN_2 U396 ( .I(a[0]), .ZN(n459) );
  VHSR_NOR4_2 U397 ( .A1(n389), .A2(n369), .A3(n457), .A4(n459), .ZN(n381) );
  VHSR_CLKNAND2_2 U398 ( .A1(b[2]), .A2(a[1]), .ZN(n366) );
  VHSR_OAI32_2 U399 ( .A1(n381), .A2(n459), .A3(n389), .B1(n366), .B2(n381), 
        .ZN(n438) );
  VHSR_NOR4_2 U400 ( .A1(n460), .A2(n458), .A3(n388), .A4(n370), .ZN(n380) );
  VHSR_CLKNAND2_2 U401 ( .A1(b[0]), .A2(a[3]), .ZN(n367) );
  VHSR_OAI32_2 U402 ( .A1(n380), .A2(n370), .A3(n460), .B1(n367), .B2(n380), 
        .ZN(n437) );
  VHSR_CLKNAND2_2 U403 ( .A1(b[1]), .A2(a[1]), .ZN(n462) );
  VHSR_AOI22_2 U404 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n463) );
  VHSR_CLKNAND2_2 U405 ( .A1(b[2]), .A2(a[2]), .ZN(n393) );
  VHSR_OAI22_2 U406 ( .A1(n462), .A2(n463), .B1(n368), .B2(n393), .ZN(n461) );
  VHSR_OAI211_2 U407 ( .A1(n369), .A2(n459), .B(b[3]), .C(a[1]), .ZN(n372) );
  VHSR_OAI211_2 U408 ( .A1(n458), .A2(n370), .B(b[1]), .C(a[3]), .ZN(n371) );
  VHSR_AND2_2 U409 ( .A1(n372), .A2(n371), .Z(n373) );
  VHSR_MAOI222_2 U410 ( .A(n393), .B(n372), .C(n371), .ZN(n374) );
  VHSR_AOI21_2 U411 ( .A1(n373), .A2(n393), .B(n374), .ZN(n398) );
  VHSR_CLKNAND2_2 U412 ( .A1(n399), .A2(n398), .ZN(n397) );
  VHSR_CLKNAND2_2 U413 ( .A1(b[2]), .A2(a[3]), .ZN(n376) );
  VHSR_AOI21_2 U414 ( .A1(b[3]), .A2(a[2]), .B(n376), .ZN(n375) );
  VHSR_AOI31_2 U415 ( .A1(b[3]), .A2(n376), .A3(a[2]), .B(n375), .ZN(n379) );
  VHSR_NOR2_1 U416 ( .A1(n381), .A2(n380), .ZN(n378) );
  VHSR_AOI22_2 U417 ( .A1(n381), .A2(n380), .B1(n379), .B2(n378), .ZN(n377) );
  VHSR_OAI21_2 U418 ( .A1(n379), .A2(n378), .B(n377), .ZN(n395) );
  VHSR_NOR2_1 U419 ( .A1(n396), .A2(n395), .ZN(n394) );
  VHSR_IN_2 U420 ( .I(n379), .ZN(n382) );
  VHSR_MAOI222_2 U421 ( .A(n382), .B(n381), .C(n380), .ZN(n383) );
  VHSR_AOI211_2 U422 ( .A1(n384), .A2(n393), .B(n388), .C(n389), .ZN(n440) );
  VHSR_AD1_1 U423 ( .A(n387), .B(n386), .CI(n385), .CO(n363), .S(n434) );
  VHSR_NOR2_1 U424 ( .A1(n389), .A2(n388), .ZN(n391) );
  VHSR_AOI21_2 U425 ( .A1(n393), .A2(n391), .B(n392), .ZN(n390) );
  VHSR_AOI31_2 U426 ( .A1(n393), .A2(n392), .A3(n391), .B(n390), .ZN(n433) );
  VHSR_AOI21_2 U427 ( .A1(n396), .A2(n395), .B(n394), .ZN(n436) );
  VHSR_CLKNAND2_2 U428 ( .A1(a[4]), .A2(b[0]), .ZN(n452) );
  VHSR_OAI21_2 U429 ( .A1(n399), .A2(n398), .B(n397), .ZN(n456) );
  VHSR_AOI211_2 U430 ( .A1(n452), .A2(n451), .B(n450), .C(n456), .ZN(n454) );
  VHSR_AD1_1 U431 ( .A(n401), .B(n450), .CI(n400), .CO(n385), .S(n435) );
  VHSR_CLKNAND2_2 U432 ( .A1(a[6]), .A2(b[7]), .ZN(n405) );
  VHSR_AOI21_2 U433 ( .A1(a[7]), .A2(b[6]), .B(n405), .ZN(n404) );
  VHSR_AOI31_2 U434 ( .A1(a[7]), .A2(n405), .A3(b[6]), .B(n404), .ZN(n406) );
  VHSR_CLKNAND2_2 U435 ( .A1(n407), .A2(n406), .ZN(n409) );
  VHSR_IN_2 U436 ( .I(n410), .ZN(n408) );
  VHSR_MAOI222_2 U437 ( .A(n408), .B(n407), .C(n406), .ZN(n416) );
  VHSR_IAO21_2 U438 ( .A1(n410), .A2(n409), .B(n416), .ZN(n415) );
  VHSR_XNOR2_2 U439 ( .A1(n414), .A2(n415), .ZN(n411) );
  VHSR_CLKNAND2_2 U440 ( .A1(n412), .A2(n411), .ZN(n447) );
  VHSR_OAI21_2 U441 ( .A1(n412), .A2(n411), .B(n447), .ZN(n413) );
  VHSR_AND3_2 U442 ( .A1(n448), .A2(n418), .A3(n447), .Z(n419) );
  VHSR_NOR2_1 U443 ( .A1(n446), .A2(n419), .ZN(product[15]) );
  VHSR_AD1_1 U444 ( .A(n441), .B(n440), .CI(n439), .CO(n429), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U445 ( .A(n444), .B(n443), .CI(n442), .CO(n412), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U446 ( .A1(n446), .A2(n445), .ZN(n449) );
  VHSR_XOR3_2 U447 ( .A1(n449), .A2(n448), .A3(n447), .Z(product[14]) );
  VHSR_AOI21_2 U448 ( .A1(n452), .A2(n451), .B(n450), .ZN(n453) );
  VHSR_IN_2 U449 ( .I(n453), .ZN(n455) );
  VHSR_AOI21_2 U450 ( .A1(n456), .A2(n455), .B(n454), .ZN(product[4]) );
  VHSR_OAI22_2 U451 ( .A1(n460), .A2(n459), .B1(n458), .B2(n457), .ZN(
        product[1]) );
  VHSR_AOI21_2 U452 ( .A1(n463), .A2(n462), .B(n461), .ZN(product[2]) );
endmodule



module mul8_12 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n234, n235, n236, n237, n238, n239, n240, n241,
         n242, n243, n244, n245, n246, n247, n248, n249, n250, n251, n252,
         n253, n254, n255, n256, n257, n258, n259, n260, n261, n262, n263,
         n264, n265, n266, n267, n268, n269, n270, n271, n272, n273, n274,
         n275, n276, n277, n278, n279, n280, n281, n282, n283, n284, n285,
         n286, n287, n288, n289, n290, n291, n292, n293, n294, n295, n296,
         n297, n298, n299, n300, n301, n302, n303, n304, n305, n306, n307,
         n308, n309, n310, n311, n312, n313, n314, n315, n316, n317, n318,
         n319, n320, n321, n322, n323, n324, n325, n326, n327, n328, n329,
         n330, n331, n332, n333, n334, n335, n336, n337, n338, n339, n340,
         n341, n342, n343, n344, n345, n346, n347, n348, n349, n350, n351,
         n352, n353, n354, n355, n356, n357, n358, n359, n360, n361, n362,
         n363, n364, n365, n366, n367, n368, n369, n370, n371, n372, n373,
         n374, n375, n376, n377, n378, n379, n380, n381, n382, n383, n384,
         n385, n386, n387, n388, n389, n390, n391, n392, n393, n394, n395,
         n396, n397, n398, n399, n400, n401, n402, n403, n404, n405, n406,
         n407, n408, n409, n410, n411, n412, n413, n414, n415, n416, n417,
         n418, n419, n420, n421, n422, n423, n424, n425, n426, n427, n428,
         n429, n430, n431, n432, n433, n434, n435, n436, n437;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U224 ( .A1(n261), .B1(n236), .ZN(n237) );
  VHSR_NOR2_1 U225 ( .A1(n344), .A2(n376), .ZN(n287) );
  VHSR_INAND2_2 U226 ( .A1(n386), .B1(n384), .ZN(n387) );
  VHSR_NOR2_1 U227 ( .A1(n274), .A2(n273), .ZN(n272) );
  VHSR_NOR2_1 U228 ( .A1(n293), .A2(n330), .ZN(n307) );
  VHSR_INAND2_2 U229 ( .A1(n369), .B1(n358), .ZN(n367) );
  VHSR_NOR2_1 U230 ( .A1(n316), .A2(n320), .ZN(n315) );
  VHSR_NOR2_1 U231 ( .A1(n259), .A2(n258), .ZN(n319) );
  VHSR_INOR2_2 U232 ( .A1(n395), .B1(n394), .ZN(n426) );
  VHSR_IN_2 U233 ( .I(n343), .ZN(product[0]) );
  VHSR_IN_2 U234 ( .I(n391), .ZN(product[13]) );
  VHSR_CLKN_1 U235 ( .I(n396), .ZN(n397) );
  VHSR_INAND3_1 U236 ( .A1(n423), .B1(n426), .B2(n425), .ZN(n396) );
  VHSR_INOR2_1 U237 ( .A1(n242), .B1(n272), .ZN(n262) );
  VHSR_INOR2_1 U238 ( .A1(n381), .B1(n380), .ZN(n393) );
  VHSR_CLKN_1 U239 ( .I(n367), .ZN(n359) );
  VHSR_INOR2_1 U240 ( .A1(n240), .B1(n278), .ZN(n274) );
  VHSR_INOR3_1 U241 ( .A1(n307), .B1(n298), .B2(n333), .ZN(n388) );
  VHSR_AD1_1 U242 ( .A(n414), .B(n428), .CI(n413), .CO(n410), .S(product[5])
         );
  VHSR_AD1_1 U243 ( .A(n409), .B(n408), .CI(n407), .CO(n404), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U244 ( .A(n403), .B(n402), .CI(n401), .CO(n398), .S(product[10])
         );
  VHSR_AD1_1 U245 ( .A(n416), .B(n415), .CI(n435), .CO(n374), .S(product[3])
         );
  VHSR_AD1_1 U246 ( .A(n412), .B(n411), .CI(n410), .CO(n417), .S(product[6])
         );
  VHSR_AD1_1 U247 ( .A(n406), .B(n405), .CI(n404), .CO(n401), .S(product[9])
         );
  VHSR_AD1_1 U248 ( .A(n400), .B(n399), .CI(n398), .CO(n420), .S(
        \intadd_0/SUM[6] ) );
  VHSR_CLKNAND2_2 U249 ( .A1(b[0]), .A2(a[0]), .ZN(n343) );
  VHSR_CLKNAND2_2 U250 ( .A1(b[3]), .A2(a[7]), .ZN(n259) );
  VHSR_IN_2 U251 ( .I(b[3]), .ZN(n364) );
  VHSR_IN_2 U252 ( .I(a[6]), .ZN(n293) );
  VHSR_IN_2 U253 ( .I(a[7]), .ZN(n298) );
  VHSR_IN_2 U254 ( .I(b[2]), .ZN(n344) );
  VHSR_OAI22_2 U255 ( .A1(n364), .A2(n293), .B1(n298), .B2(n344), .ZN(n264) );
  VHSR_IN_2 U256 ( .I(a[4]), .ZN(n376) );
  VHSR_CLKNAND2_2 U257 ( .A1(b[3]), .A2(a[5]), .ZN(n234) );
  VHSR_IN_2 U258 ( .I(b[1]), .ZN(n434) );
  VHSR_OAI22_2 U259 ( .A1(n287), .A2(n234), .B1(n298), .B2(n434), .ZN(n241) );
  VHSR_IN_2 U260 ( .I(a[5]), .ZN(n331) );
  VHSR_NOR4_2 U261 ( .A1(n287), .A2(n259), .A3(n331), .A4(n434), .ZN(n235) );
  VHSR_AOI31_2 U262 ( .A1(b[2]), .A2(a[6]), .A3(n241), .B(n235), .ZN(n242) );
  VHSR_IN_2 U263 ( .I(b[0]), .ZN(n432) );
  VHSR_NOR4_2 U264 ( .A1(n376), .A2(n331), .A3(n434), .A4(n432), .ZN(n292) );
  VHSR_NAND3_2 U265 ( .A1(b[3]), .A2(n287), .A3(a[5]), .ZN(n261) );
  VHSR_AOI22_2 U266 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n236) );
  VHSR_OAI22_2 U267 ( .A1(n298), .A2(n432), .B1(n293), .B2(n434), .ZN(n238) );
  VHSR_MAOI222_2 U268 ( .A(n292), .B(n237), .C(n238), .ZN(n240) );
  VHSR_NOR2_1 U269 ( .A1(n293), .A2(n432), .ZN(n286) );
  VHSR_AOI211_2 U270 ( .A1(a[4]), .A2(b[0]), .B(n331), .C(n434), .ZN(n285) );
  VHSR_MAOI222_2 U271 ( .A(n287), .B(n286), .C(n285), .ZN(n284) );
  VHSR_OR2_2 U272 ( .A1(n292), .A2(n237), .Z(n239) );
  VHSR_OAI21_2 U273 ( .A1(n239), .A2(n238), .B(n240), .ZN(n279) );
  VHSR_NOR2_1 U274 ( .A1(n284), .A2(n279), .ZN(n278) );
  VHSR_AOI32_2 U275 ( .A1(b[2]), .A2(n242), .A3(a[6]), .B1(n241), .B2(n242), 
        .ZN(n273) );
  VHSR_CLKNAND2_2 U276 ( .A1(n262), .A2(n261), .ZN(n260) );
  VHSR_CLKNAND2_2 U277 ( .A1(n264), .A2(n260), .ZN(n258) );
  VHSR_CLKNAND2_2 U278 ( .A1(b[6]), .A2(a[2]), .ZN(n250) );
  VHSR_IN_2 U279 ( .I(n250), .ZN(n257) );
  VHSR_IN_2 U280 ( .I(b[5]), .ZN(n333) );
  VHSR_IN_2 U281 ( .I(a[3]), .ZN(n363) );
  VHSR_CLKNAND2_2 U282 ( .A1(b[4]), .A2(a[2]), .ZN(n283) );
  VHSR_NOR3_2 U283 ( .A1(n333), .A2(n363), .A3(n283), .ZN(n267) );
  VHSR_CLKNAND2_2 U284 ( .A1(b[7]), .A2(a[3]), .ZN(n255) );
  VHSR_AOI22_2 U285 ( .A1(b[6]), .A2(a[3]), .B1(b[7]), .B2(a[2]), .ZN(n243) );
  VHSR_IAO21_2 U286 ( .A1(n255), .A2(n250), .B(n243), .ZN(n266) );
  VHSR_IN_2 U287 ( .I(a[1]), .ZN(n431) );
  VHSR_CLKNAND2_2 U288 ( .A1(b[4]), .A2(a[0]), .ZN(n375) );
  VHSR_NOR3_2 U289 ( .A1(n333), .A2(n431), .A3(n375), .ZN(n290) );
  VHSR_AOI22_2 U290 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n244) );
  VHSR_NOR2_1 U291 ( .A1(n267), .A2(n244), .ZN(n248) );
  VHSR_IN_2 U292 ( .I(b[6]), .ZN(n301) );
  VHSR_IN_2 U293 ( .I(b[7]), .ZN(n300) );
  VHSR_IN_2 U294 ( .I(a[0]), .ZN(n433) );
  VHSR_OAI22_2 U295 ( .A1(n301), .A2(n431), .B1(n300), .B2(n433), .ZN(n247) );
  VHSR_IN_2 U296 ( .I(n269), .ZN(n253) );
  VHSR_NOR2_1 U297 ( .A1(n300), .A2(n431), .ZN(n246) );
  VHSR_NAND3_2 U298 ( .A1(n283), .A2(a[3]), .A3(b[5]), .ZN(n249) );
  VHSR_IN_2 U299 ( .I(n249), .ZN(n245) );
  VHSR_MAOI222_2 U300 ( .A(n246), .B(n257), .C(n245), .ZN(n252) );
  VHSR_AD1_1 U301 ( .A(n290), .B(n248), .CI(n247), .CO(n269), .S(n276) );
  VHSR_IN_2 U302 ( .I(b[4]), .ZN(n330) );
  VHSR_OAI211_2 U303 ( .A1(n330), .A2(n433), .B(b[5]), .C(a[1]), .ZN(n282) );
  VHSR_CLKNAND2_2 U304 ( .A1(b[6]), .A2(a[0]), .ZN(n281) );
  VHSR_MAOI222_2 U305 ( .A(n283), .B(n282), .C(n281), .ZN(n280) );
  VHSR_CLKNAND2_2 U306 ( .A1(n276), .A2(n280), .ZN(n275) );
  VHSR_CLKNAND2_2 U307 ( .A1(n250), .A2(n249), .ZN(n251) );
  VHSR_AOI32_2 U308 ( .A1(a[1]), .A2(n252), .A3(b[7]), .B1(n251), .B2(n252), 
        .ZN(n268) );
  VHSR_AOI32_2 U309 ( .A1(n253), .A2(n252), .A3(n275), .B1(n268), .B2(n252), 
        .ZN(n265) );
  VHSR_IAO21_2 U310 ( .A1(n257), .A2(n256), .B(n255), .ZN(n318) );
  VHSR_OAI21_2 U311 ( .A1(n257), .A2(n255), .B(n256), .ZN(n254) );
  VHSR_OAI31_2 U312 ( .A1(n257), .A2(n256), .A3(n255), .B(n254), .ZN(n326) );
  VHSR_AOI21_2 U313 ( .A1(n259), .A2(n258), .B(n319), .ZN(n325) );
  VHSR_OAI21_2 U314 ( .A1(n262), .A2(n261), .B(n260), .ZN(n263) );
  VHSR_XNOR2_2 U315 ( .A1(n264), .A2(n263), .ZN(n329) );
  VHSR_AD1_1 U316 ( .A(n267), .B(n266), .CI(n265), .CO(n256), .S(n328) );
  VHSR_NOR2_1 U317 ( .A1(n269), .A2(n268), .ZN(n271) );
  VHSR_AOI22_2 U318 ( .A1(n269), .A2(n268), .B1(n275), .B2(n271), .ZN(n270) );
  VHSR_OAI21_2 U319 ( .A1(n275), .A2(n271), .B(n270), .ZN(n337) );
  VHSR_AOI21_2 U320 ( .A1(n274), .A2(n273), .B(n272), .ZN(n336) );
  VHSR_OAI21_2 U321 ( .A1(n276), .A2(n280), .B(n275), .ZN(n277) );
  VHSR_IN_2 U322 ( .I(n277), .ZN(n340) );
  VHSR_AOI21_2 U323 ( .A1(n284), .A2(n279), .B(n278), .ZN(n339) );
  VHSR_AOI31_2 U324 ( .A1(n283), .A2(n282), .A3(n281), .B(n280), .ZN(n362) );
  VHSR_OAI31_2 U325 ( .A1(n287), .A2(n286), .A3(n285), .B(n284), .ZN(n288) );
  VHSR_IN_2 U326 ( .I(n288), .ZN(n361) );
  VHSR_AOI22_2 U327 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n289) );
  VHSR_NOR2_1 U328 ( .A1(n290), .A2(n289), .ZN(n379) );
  VHSR_CLKNAND2_2 U329 ( .A1(a[4]), .A2(b[4]), .ZN(n304) );
  VHSR_NOR2_1 U330 ( .A1(n304), .A2(n343), .ZN(n378) );
  VHSR_CLKNAND2_2 U331 ( .A1(a[5]), .A2(b[0]), .ZN(n291) );
  VHSR_OAI32_2 U332 ( .A1(n292), .A2(n434), .A3(n376), .B1(n291), .B2(n292), 
        .ZN(n377) );
  VHSR_NOR2_1 U333 ( .A1(n293), .A2(n301), .ZN(n423) );
  VHSR_NOR2_1 U334 ( .A1(n376), .A2(n301), .ZN(n306) );
  VHSR_CLKNAND2_2 U335 ( .A1(a[5]), .A2(b[7]), .ZN(n295) );
  VHSR_CLKNAND2_2 U336 ( .A1(a[7]), .A2(b[5]), .ZN(n294) );
  VHSR_OAI22_2 U337 ( .A1(n306), .A2(n295), .B1(n307), .B2(n294), .ZN(n297) );
  VHSR_OR2_2 U338 ( .A1(n306), .A2(n307), .Z(n321) );
  VHSR_CLKNAND2_2 U339 ( .A1(a[5]), .A2(b[5]), .ZN(n305) );
  VHSR_CLKNAND2_2 U340 ( .A1(a[7]), .A2(b[7]), .ZN(n424) );
  VHSR_NOR3_2 U341 ( .A1(n321), .A2(n305), .A3(n424), .ZN(n296) );
  VHSR_AOI31_2 U342 ( .A1(b[6]), .A2(a[6]), .A3(n297), .B(n296), .ZN(n381) );
  VHSR_OAI21_2 U343 ( .A1(n423), .A2(n297), .B(n381), .ZN(n314) );
  VHSR_AOI22_2 U344 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n299) );
  VHSR_NOR2_1 U345 ( .A1(n388), .A2(n299), .ZN(n310) );
  VHSR_NOR2_1 U346 ( .A1(n305), .A2(n304), .ZN(n309) );
  VHSR_NOR4_2 U347 ( .A1(n376), .A2(n331), .A3(n301), .A4(n300), .ZN(n386) );
  VHSR_AOI22_2 U348 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n302) );
  VHSR_NOR2_1 U349 ( .A1(n386), .A2(n302), .ZN(n308) );
  VHSR_IN_2 U350 ( .I(n303), .ZN(n316) );
  VHSR_IN_2 U351 ( .I(n304), .ZN(n408) );
  VHSR_NOR2_1 U352 ( .A1(n408), .A2(n305), .ZN(n322) );
  VHSR_AOI22_2 U353 ( .A1(n307), .A2(n306), .B1(n322), .B2(n321), .ZN(n320) );
  VHSR_AD1_1 U354 ( .A(n310), .B(n309), .CI(n308), .CO(n311), .S(n303) );
  VHSR_NOR2_1 U355 ( .A1(n315), .A2(n311), .ZN(n313) );
  VHSR_CLKNAND2_2 U356 ( .A1(n315), .A2(n311), .ZN(n312) );
  VHSR_NOR2_1 U357 ( .A1(n313), .A2(n314), .ZN(n380) );
  VHSR_AOI22_2 U358 ( .A1(n314), .A2(n313), .B1(n312), .B2(n380), .ZN(n421) );
  VHSR_AOI21_2 U359 ( .A1(n320), .A2(n316), .B(n315), .ZN(n400) );
  VHSR_AD1_1 U360 ( .A(n319), .B(n318), .CI(n317), .CO(n422), .S(n399) );
  VHSR_OAI21_2 U361 ( .A1(n322), .A2(n321), .B(n320), .ZN(n323) );
  VHSR_IN_2 U362 ( .I(n323), .ZN(n403) );
  VHSR_AD1_1 U363 ( .A(n326), .B(n325), .CI(n324), .CO(n317), .S(n402) );
  VHSR_AD1_1 U364 ( .A(n329), .B(n328), .CI(n327), .CO(n324), .S(n406) );
  VHSR_NOR2_1 U365 ( .A1(n331), .A2(n330), .ZN(n334) );
  VHSR_OAI21_2 U366 ( .A1(n376), .A2(n333), .B(n334), .ZN(n332) );
  VHSR_OAI31_2 U367 ( .A1(n376), .A2(n334), .A3(n333), .B(n332), .ZN(n405) );
  VHSR_AD1_1 U368 ( .A(n337), .B(n336), .CI(n335), .CO(n327), .S(n409) );
  VHSR_AD1_1 U369 ( .A(n340), .B(n339), .CI(n338), .CO(n335), .S(n419) );
  VHSR_NOR4_2 U370 ( .A1(n364), .A2(n344), .A3(n431), .A4(n433), .ZN(n356) );
  VHSR_CLKNAND2_2 U371 ( .A1(b[2]), .A2(a[1]), .ZN(n341) );
  VHSR_OAI32_2 U372 ( .A1(n356), .A2(n364), .A3(n433), .B1(n341), .B2(n356), 
        .ZN(n416) );
  VHSR_IN_2 U373 ( .I(a[2]), .ZN(n345) );
  VHSR_NOR4_2 U374 ( .A1(n434), .A2(n432), .A3(n363), .A4(n345), .ZN(n355) );
  VHSR_CLKNAND2_2 U375 ( .A1(b[0]), .A2(a[3]), .ZN(n342) );
  VHSR_OAI32_2 U376 ( .A1(n355), .A2(n345), .A3(n434), .B1(n342), .B2(n355), 
        .ZN(n415) );
  VHSR_CLKNAND2_2 U377 ( .A1(b[1]), .A2(a[1]), .ZN(n436) );
  VHSR_AOI22_2 U378 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n437) );
  VHSR_CLKNAND2_2 U379 ( .A1(b[2]), .A2(a[2]), .ZN(n368) );
  VHSR_OAI22_2 U380 ( .A1(n436), .A2(n437), .B1(n343), .B2(n368), .ZN(n435) );
  VHSR_OAI211_2 U381 ( .A1(n344), .A2(n433), .B(b[3]), .C(a[1]), .ZN(n347) );
  VHSR_OAI211_2 U382 ( .A1(n432), .A2(n345), .B(b[1]), .C(a[3]), .ZN(n346) );
  VHSR_AND2_2 U383 ( .A1(n347), .A2(n346), .Z(n348) );
  VHSR_MAOI222_2 U384 ( .A(n368), .B(n347), .C(n346), .ZN(n349) );
  VHSR_AOI21_2 U385 ( .A1(n348), .A2(n368), .B(n349), .ZN(n373) );
  VHSR_AND2_2 U386 ( .A1(n374), .A2(n373), .Z(n372) );
  VHSR_NOR2_1 U387 ( .A1(n372), .A2(n349), .ZN(n371) );
  VHSR_CLKNAND2_2 U388 ( .A1(b[2]), .A2(a[3]), .ZN(n351) );
  VHSR_AOI21_2 U389 ( .A1(b[3]), .A2(a[2]), .B(n351), .ZN(n350) );
  VHSR_AOI31_2 U390 ( .A1(b[3]), .A2(n351), .A3(a[2]), .B(n350), .ZN(n354) );
  VHSR_NOR2_1 U391 ( .A1(n356), .A2(n355), .ZN(n353) );
  VHSR_AOI22_2 U392 ( .A1(n356), .A2(n355), .B1(n354), .B2(n353), .ZN(n352) );
  VHSR_OAI21_2 U393 ( .A1(n354), .A2(n353), .B(n352), .ZN(n370) );
  VHSR_NOR2_1 U394 ( .A1(n371), .A2(n370), .ZN(n369) );
  VHSR_IN_2 U395 ( .I(n354), .ZN(n357) );
  VHSR_MAOI222_2 U396 ( .A(n357), .B(n356), .C(n355), .ZN(n358) );
  VHSR_AOI211_2 U397 ( .A1(n359), .A2(n368), .B(n363), .C(n364), .ZN(n418) );
  VHSR_AD1_1 U398 ( .A(n362), .B(n361), .CI(n360), .CO(n338), .S(n412) );
  VHSR_NOR2_1 U399 ( .A1(n364), .A2(n363), .ZN(n366) );
  VHSR_AOI21_2 U400 ( .A1(n368), .A2(n366), .B(n367), .ZN(n365) );
  VHSR_AOI31_2 U401 ( .A1(n368), .A2(n367), .A3(n366), .B(n365), .ZN(n411) );
  VHSR_AOI21_2 U402 ( .A1(n371), .A2(n370), .B(n369), .ZN(n414) );
  VHSR_IAO21_2 U403 ( .A1(n374), .A2(n373), .B(n372), .ZN(n430) );
  VHSR_OAI32_2 U404 ( .A1(n378), .A2(n432), .A3(n376), .B1(n375), .B2(n378), 
        .ZN(n429) );
  VHSR_AND2_2 U405 ( .A1(n430), .A2(n429), .Z(n428) );
  VHSR_AD1_1 U406 ( .A(n379), .B(n378), .CI(n377), .CO(n360), .S(n413) );
  VHSR_CLKNAND2_2 U407 ( .A1(a[6]), .A2(b[7]), .ZN(n383) );
  VHSR_AOI21_2 U408 ( .A1(a[7]), .A2(b[6]), .B(n383), .ZN(n382) );
  VHSR_AOI31_2 U409 ( .A1(a[7]), .A2(n383), .A3(b[6]), .B(n382), .ZN(n384) );
  VHSR_IN_2 U410 ( .I(n384), .ZN(n385) );
  VHSR_MAOI222_2 U411 ( .A(n388), .B(n386), .C(n385), .ZN(n395) );
  VHSR_OAI21_2 U412 ( .A1(n388), .A2(n387), .B(n395), .ZN(n392) );
  VHSR_CLKXOR2_2 U413 ( .A1(n393), .A2(n392), .Z(n389) );
  VHSR_CLKNAND2_2 U414 ( .A1(n390), .A2(n389), .ZN(n425) );
  VHSR_OAI21_2 U415 ( .A1(n390), .A2(n389), .B(n425), .ZN(n391) );
  VHSR_NOR2_1 U416 ( .A1(n393), .A2(n392), .ZN(n394) );
  VHSR_NOR2_1 U417 ( .A1(n424), .A2(n397), .ZN(product[15]) );
  VHSR_AD1_1 U418 ( .A(n419), .B(n418), .CI(n417), .CO(n407), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U419 ( .A(n422), .B(n421), .CI(n420), .CO(n390), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U420 ( .A1(n424), .A2(n423), .ZN(n427) );
  VHSR_XOR3_2 U421 ( .A1(n427), .A2(n426), .A3(n425), .Z(product[14]) );
  VHSR_IAO21_2 U422 ( .A1(n430), .A2(n429), .B(n428), .ZN(product[4]) );
  VHSR_OAI22_2 U423 ( .A1(n434), .A2(n433), .B1(n432), .B2(n431), .ZN(
        product[1]) );
  VHSR_AOI21_2 U424 ( .A1(n437), .A2(n436), .B(n435), .ZN(product[2]) );
endmodule


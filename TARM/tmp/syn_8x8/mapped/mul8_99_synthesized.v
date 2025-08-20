
module mul8_99 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n236, n237, n238, n239, n240, n241, n242, n243,
         n244, n245, n246, n247, n248, n249, n250, n251, n252, n253, n254,
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
         n442;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U226 ( .A1(n249), .B1(n248), .ZN(n250) );
  VHSR_INOR2_2 U227 ( .A1(n389), .B1(n316), .ZN(n319) );
  VHSR_INOR2_2 U228 ( .A1(n252), .B1(n280), .ZN(n269) );
  VHSR_NOR2_1 U229 ( .A1(n330), .A2(n329), .ZN(n328) );
  VHSR_INOR3_2 U230 ( .A1(n268), .B1(n361), .B2(n313), .ZN(n326) );
  VHSR_NOR2_1 U231 ( .A1(n434), .A2(n433), .ZN(n432) );
  VHSR_NOR2_1 U232 ( .A1(n315), .A2(n374), .ZN(n409) );
  VHSR_IN_2 U233 ( .I(n395), .ZN(product[13]) );
  VHSR_INOR2_1 U234 ( .A1(n397), .B1(n396), .ZN(n399) );
  VHSR_MOAI22_1 U235 ( .A1(n313), .A2(n438), .B1(a[6]), .B2(b[2]), .ZN(n256)
         );
  VHSR_AD1_1 U236 ( .A(n416), .B(n415), .CI(n414), .CO(n411), .S(product[6])
         );
  VHSR_AD1_1 U237 ( .A(n407), .B(n406), .CI(n405), .CO(n402), .S(product[10])
         );
  VHSR_AD1_1 U238 ( .A(n420), .B(n419), .CI(n439), .CO(n373), .S(product[3])
         );
  VHSR_AD1_1 U239 ( .A(n432), .B(n418), .CI(n417), .CO(n414), .S(product[5])
         );
  VHSR_AD1_1 U240 ( .A(n413), .B(n412), .CI(n411), .CO(n408), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U241 ( .A(n410), .B(n409), .CI(n408), .CO(n421), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U242 ( .A(n404), .B(n403), .CI(n402), .CO(n424), .S(
        \intadd_0/SUM[6] ) );
  VHSR_IN_2 U243 ( .I(b[7]), .ZN(n270) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[6]), .A2(a[0]), .ZN(n300) );
  VHSR_IN_2 U245 ( .I(a[1]), .ZN(n435) );
  VHSR_NOR3_2 U246 ( .A1(n270), .A2(n300), .A3(n435), .ZN(n251) );
  VHSR_IN_2 U247 ( .I(b[4]), .ZN(n374) );
  VHSR_IN_2 U248 ( .I(b[5]), .ZN(n318) );
  VHSR_IN_2 U249 ( .I(a[3]), .ZN(n358) );
  VHSR_IN_2 U250 ( .I(a[2]), .ZN(n360) );
  VHSR_NOR4_2 U251 ( .A1(n374), .A2(n318), .A3(n358), .A4(n360), .ZN(n249) );
  VHSR_CLKNAND2_2 U252 ( .A1(b[7]), .A2(a[2]), .ZN(n237) );
  VHSR_AOI21_2 U253 ( .A1(b[6]), .A2(a[3]), .B(n237), .ZN(n236) );
  VHSR_AOI31_2 U254 ( .A1(b[6]), .A2(n237), .A3(a[3]), .B(n236), .ZN(n248) );
  VHSR_IN_2 U255 ( .I(n248), .ZN(n238) );
  VHSR_MAOI222_2 U256 ( .A(n251), .B(n249), .C(n238), .ZN(n252) );
  VHSR_CLKNAND2_2 U257 ( .A1(b[6]), .A2(a[2]), .ZN(n274) );
  VHSR_NAND3_2 U258 ( .A1(b[7]), .A2(a[1]), .A3(n300), .ZN(n243) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[4]), .A2(a[2]), .ZN(n299) );
  VHSR_NAND3_2 U260 ( .A1(a[3]), .A2(b[5]), .A3(n299), .ZN(n245) );
  VHSR_MAOI222_2 U261 ( .A(n274), .B(n243), .C(n245), .ZN(n247) );
  VHSR_IN_2 U262 ( .I(a[0]), .ZN(n437) );
  VHSR_OAI211_2 U263 ( .A1(n374), .A2(n437), .B(b[5]), .C(a[1]), .ZN(n298) );
  VHSR_MAOI222_2 U264 ( .A(n300), .B(n299), .C(n298), .ZN(n297) );
  VHSR_AOI22_2 U265 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n239) );
  VHSR_NOR2_1 U266 ( .A1(n249), .A2(n239), .ZN(n242) );
  VHSR_NOR4_2 U267 ( .A1(n374), .A2(n318), .A3(n437), .A4(n435), .ZN(n308) );
  VHSR_AOI22_2 U268 ( .A1(b[6]), .A2(a[1]), .B1(b[7]), .B2(a[0]), .ZN(n240) );
  VHSR_NOR2_1 U269 ( .A1(n251), .A2(n240), .ZN(n241) );
  VHSR_AND2_2 U270 ( .A1(n297), .A2(n293), .Z(n292) );
  VHSR_AD1_1 U271 ( .A(n242), .B(n308), .CI(n241), .CO(n287), .S(n293) );
  VHSR_NOR2_1 U272 ( .A1(n292), .A2(n287), .ZN(n290) );
  VHSR_AND2_2 U273 ( .A1(n274), .A2(n243), .Z(n244) );
  VHSR_AOI21_2 U274 ( .A1(n245), .A2(n244), .B(n247), .ZN(n246) );
  VHSR_IN_2 U275 ( .I(n246), .ZN(n291) );
  VHSR_NOR2_1 U276 ( .A1(n290), .A2(n291), .ZN(n288) );
  VHSR_NOR2_1 U277 ( .A1(n247), .A2(n288), .ZN(n282) );
  VHSR_OAI21_2 U278 ( .A1(n251), .A2(n250), .B(n252), .ZN(n281) );
  VHSR_NOR2_1 U279 ( .A1(n282), .A2(n281), .ZN(n280) );
  VHSR_AOI211_2 U280 ( .A1(n269), .A2(n274), .B(n358), .C(n270), .ZN(n327) );
  VHSR_AOI22_2 U281 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n279) );
  VHSR_IN_2 U282 ( .I(b[3]), .ZN(n361) );
  VHSR_IN_2 U283 ( .I(b[2]), .ZN(n359) );
  VHSR_IN_2 U284 ( .I(a[5]), .ZN(n339) );
  VHSR_IN_2 U285 ( .I(a[4]), .ZN(n315) );
  VHSR_NOR4_2 U286 ( .A1(n361), .A2(n359), .A3(n339), .A4(n315), .ZN(n277) );
  VHSR_IN_2 U287 ( .I(a[7]), .ZN(n313) );
  VHSR_IN_2 U288 ( .I(b[1]), .ZN(n438) );
  VHSR_NOR2_1 U289 ( .A1(n313), .A2(n438), .ZN(n254) );
  VHSR_AND2_2 U290 ( .A1(a[6]), .A2(b[2]), .Z(n253) );
  VHSR_AOI211_2 U291 ( .A1(b[2]), .A2(a[4]), .B(n361), .C(n339), .ZN(n255) );
  VHSR_MAOI222_2 U292 ( .A(n254), .B(n253), .C(n255), .ZN(n266) );
  VHSR_OAI21_2 U293 ( .A1(n256), .A2(n255), .B(n266), .ZN(n257) );
  VHSR_IN_2 U294 ( .I(n257), .ZN(n285) );
  VHSR_IN_2 U295 ( .I(b[0]), .ZN(n436) );
  VHSR_NOR4_2 U296 ( .A1(n339), .A2(n315), .A3(n438), .A4(n436), .ZN(n306) );
  VHSR_CLKNAND2_2 U297 ( .A1(b[2]), .A2(a[5]), .ZN(n259) );
  VHSR_CLKNAND2_2 U298 ( .A1(b[3]), .A2(a[4]), .ZN(n258) );
  VHSR_AOI21_2 U299 ( .A1(n259), .A2(n258), .B(n277), .ZN(n261) );
  VHSR_AOI22_2 U300 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n263) );
  VHSR_IN_2 U301 ( .I(n263), .ZN(n260) );
  VHSR_MAOI222_2 U302 ( .A(n306), .B(n261), .C(n260), .ZN(n265) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[2]), .A2(a[4]), .ZN(n304) );
  VHSR_OAI211_2 U304 ( .A1(n315), .A2(n436), .B(a[5]), .C(b[1]), .ZN(n303) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[6]), .A2(b[0]), .ZN(n302) );
  VHSR_MAOI222_2 U306 ( .A(n304), .B(n303), .C(n302), .ZN(n301) );
  VHSR_NOR2_1 U307 ( .A1(n306), .A2(n261), .ZN(n264) );
  VHSR_IN_2 U308 ( .I(n265), .ZN(n262) );
  VHSR_AOI21_2 U309 ( .A1(n264), .A2(n263), .B(n262), .ZN(n295) );
  VHSR_CLKNAND2_2 U310 ( .A1(n301), .A2(n295), .ZN(n294) );
  VHSR_CLKNAND2_2 U311 ( .A1(n265), .A2(n294), .ZN(n284) );
  VHSR_CLKNAND2_2 U312 ( .A1(n285), .A2(n284), .ZN(n283) );
  VHSR_CLKNAND2_2 U313 ( .A1(n266), .A2(n283), .ZN(n276) );
  VHSR_NOR2_1 U314 ( .A1(n277), .A2(n276), .ZN(n275) );
  VHSR_NOR2_1 U315 ( .A1(n279), .A2(n275), .ZN(n268) );
  VHSR_NOR2_1 U316 ( .A1(n361), .A2(n313), .ZN(n267) );
  VHSR_IAO21_2 U317 ( .A1(n268), .A2(n267), .B(n326), .ZN(n333) );
  VHSR_IN_2 U318 ( .I(n269), .ZN(n273) );
  VHSR_NOR2_1 U319 ( .A1(n270), .A2(n358), .ZN(n272) );
  VHSR_AOI21_2 U320 ( .A1(n274), .A2(n272), .B(n273), .ZN(n271) );
  VHSR_AOI31_2 U321 ( .A1(n274), .A2(n273), .A3(n272), .B(n271), .ZN(n332) );
  VHSR_AOI21_2 U322 ( .A1(n277), .A2(n276), .B(n275), .ZN(n278) );
  VHSR_XNOR2_2 U323 ( .A1(n279), .A2(n278), .ZN(n343) );
  VHSR_AOI21_2 U324 ( .A1(n282), .A2(n281), .B(n280), .ZN(n342) );
  VHSR_OAI21_2 U325 ( .A1(n285), .A2(n284), .B(n283), .ZN(n286) );
  VHSR_IN_2 U326 ( .I(n286), .ZN(n346) );
  VHSR_CLKNAND2_2 U327 ( .A1(n292), .A2(n287), .ZN(n289) );
  VHSR_AOI22_2 U328 ( .A1(n291), .A2(n290), .B1(n289), .B2(n288), .ZN(n345) );
  VHSR_IAO21_2 U329 ( .A1(n297), .A2(n293), .B(n292), .ZN(n349) );
  VHSR_OAI21_2 U330 ( .A1(n301), .A2(n295), .B(n294), .ZN(n296) );
  VHSR_IN_2 U331 ( .I(n296), .ZN(n348) );
  VHSR_AOI31_2 U332 ( .A1(n300), .A2(n299), .A3(n298), .B(n297), .ZN(n365) );
  VHSR_AOI31_2 U333 ( .A1(n304), .A2(n303), .A3(n302), .B(n301), .ZN(n364) );
  VHSR_NOR2_1 U334 ( .A1(n436), .A2(n437), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U335 ( .A1(n409), .A2(product[0]), .ZN(n376) );
  VHSR_IN_2 U336 ( .I(n376), .ZN(n383) );
  VHSR_CLKNAND2_2 U337 ( .A1(a[4]), .A2(b[1]), .ZN(n305) );
  VHSR_OAI32_2 U338 ( .A1(n306), .A2(n339), .A3(n436), .B1(n305), .B2(n306), 
        .ZN(n382) );
  VHSR_CLKNAND2_2 U339 ( .A1(b[5]), .A2(a[0]), .ZN(n307) );
  VHSR_OAI32_2 U340 ( .A1(n308), .A2(n435), .A3(n374), .B1(n307), .B2(n308), 
        .ZN(n381) );
  VHSR_CLKNAND2_2 U341 ( .A1(a[6]), .A2(b[6]), .ZN(n400) );
  VHSR_IN_2 U342 ( .I(n400), .ZN(n427) );
  VHSR_CLKNAND2_2 U343 ( .A1(a[6]), .A2(b[4]), .ZN(n337) );
  VHSR_NAND3_2 U344 ( .A1(a[7]), .A2(b[5]), .A3(n337), .ZN(n310) );
  VHSR_CLKNAND2_2 U345 ( .A1(a[4]), .A2(b[6]), .ZN(n336) );
  VHSR_NAND3_2 U346 ( .A1(b[7]), .A2(a[5]), .A3(n336), .ZN(n309) );
  VHSR_CLKNAND2_2 U347 ( .A1(n310), .A2(n309), .ZN(n312) );
  VHSR_MAOI222_2 U348 ( .A(n400), .B(n310), .C(n309), .ZN(n384) );
  VHSR_IN_2 U349 ( .I(n384), .ZN(n311) );
  VHSR_OAI21_2 U350 ( .A1(n427), .A2(n312), .B(n311), .ZN(n324) );
  VHSR_NOR3_2 U351 ( .A1(n313), .A2(n337), .A3(n318), .ZN(n392) );
  VHSR_AOI22_2 U352 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n314) );
  VHSR_NOR2_1 U353 ( .A1(n392), .A2(n314), .ZN(n320) );
  VHSR_NOR4_2 U354 ( .A1(n339), .A2(n315), .A3(n374), .A4(n318), .ZN(n340) );
  VHSR_NAND4_2 U355 ( .A1(a[5]), .A2(a[4]), .A3(b[6]), .A4(b[7]), .ZN(n389) );
  VHSR_AOI22_2 U356 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n316) );
  VHSR_IN_2 U357 ( .I(n317), .ZN(n330) );
  VHSR_OR3_2 U358 ( .A1(n409), .A2(n318), .A3(n339), .Z(n335) );
  VHSR_MAOI222_2 U359 ( .A(n337), .B(n336), .C(n335), .ZN(n334) );
  VHSR_IN_2 U360 ( .I(n334), .ZN(n329) );
  VHSR_AD1_1 U361 ( .A(n320), .B(n340), .CI(n319), .CO(n321), .S(n317) );
  VHSR_NOR2_1 U362 ( .A1(n328), .A2(n321), .ZN(n323) );
  VHSR_CLKNAND2_2 U363 ( .A1(n328), .A2(n321), .ZN(n322) );
  VHSR_NOR2_1 U364 ( .A1(n323), .A2(n324), .ZN(n385) );
  VHSR_AOI22_2 U365 ( .A1(n324), .A2(n323), .B1(n322), .B2(n385), .ZN(n425) );
  VHSR_AD1_1 U366 ( .A(n327), .B(n326), .CI(n325), .CO(n426), .S(n404) );
  VHSR_AOI21_2 U367 ( .A1(n330), .A2(n329), .B(n328), .ZN(n403) );
  VHSR_AD1_1 U368 ( .A(n333), .B(n332), .CI(n331), .CO(n325), .S(n407) );
  VHSR_AOI31_2 U369 ( .A1(n337), .A2(n336), .A3(n335), .B(n334), .ZN(n406) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[4]), .A2(b[5]), .ZN(n338) );
  VHSR_OAI32_2 U371 ( .A1(n340), .A2(n374), .A3(n339), .B1(n338), .B2(n340), 
        .ZN(n423) );
  VHSR_AD1_1 U372 ( .A(n343), .B(n342), .CI(n341), .CO(n331), .S(n422) );
  VHSR_AD1_1 U373 ( .A(n346), .B(n345), .CI(n344), .CO(n341), .S(n410) );
  VHSR_AD1_1 U374 ( .A(n349), .B(n348), .CI(n347), .CO(n344), .S(n413) );
  VHSR_CLKNAND2_2 U375 ( .A1(b[2]), .A2(a[0]), .ZN(n441) );
  VHSR_IN_2 U376 ( .I(n441), .ZN(n357) );
  VHSR_AOI22_2 U377 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n350) );
  VHSR_AOI31_2 U378 ( .A1(a[1]), .A2(b[3]), .A3(n357), .B(n350), .ZN(n420) );
  VHSR_CLKNAND2_2 U379 ( .A1(b[0]), .A2(a[2]), .ZN(n442) );
  VHSR_NOR3_2 U380 ( .A1(n438), .A2(n358), .A3(n442), .ZN(n356) );
  VHSR_AOI22_2 U381 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n351) );
  VHSR_NOR2_1 U382 ( .A1(n356), .A2(n351), .ZN(n419) );
  VHSR_CLKNAND2_2 U383 ( .A1(b[1]), .A2(a[1]), .ZN(n440) );
  VHSR_MAOI222_2 U384 ( .A(n442), .B(n441), .C(n440), .ZN(n439) );
  VHSR_NAND3_2 U385 ( .A1(n441), .A2(a[1]), .A3(b[3]), .ZN(n352) );
  VHSR_CLKNAND2_2 U386 ( .A1(b[2]), .A2(a[2]), .ZN(n366) );
  VHSR_AND2_2 U387 ( .A1(n352), .A2(n366), .Z(n354) );
  VHSR_NAND3_2 U388 ( .A1(a[3]), .A2(b[1]), .A3(n442), .ZN(n353) );
  VHSR_MAOI222_2 U389 ( .A(n353), .B(n366), .C(n352), .ZN(n355) );
  VHSR_AOI21_2 U390 ( .A1(n354), .A2(n353), .B(n355), .ZN(n372) );
  VHSR_AOI21_2 U391 ( .A1(n373), .A2(n372), .B(n355), .ZN(n379) );
  VHSR_AOI31_2 U392 ( .A1(a[1]), .A2(b[3]), .A3(n357), .B(n356), .ZN(n378) );
  VHSR_CLKNAND2_2 U393 ( .A1(b[3]), .A2(a[3]), .ZN(n371) );
  VHSR_OAI22_2 U394 ( .A1(n361), .A2(n360), .B1(n359), .B2(n358), .ZN(n362) );
  VHSR_OAI21_2 U395 ( .A1(n371), .A2(n366), .B(n362), .ZN(n377) );
  VHSR_AOI21_2 U396 ( .A1(n367), .A2(n366), .B(n371), .ZN(n412) );
  VHSR_AD1_1 U397 ( .A(n365), .B(n364), .CI(n363), .CO(n347), .S(n416) );
  VHSR_IN_2 U398 ( .I(n366), .ZN(n370) );
  VHSR_IN_2 U399 ( .I(n367), .ZN(n369) );
  VHSR_OAI21_2 U400 ( .A1(n371), .A2(n370), .B(n369), .ZN(n368) );
  VHSR_OAI31_2 U401 ( .A1(n371), .A2(n370), .A3(n369), .B(n368), .ZN(n415) );
  VHSR_XNOR2_2 U402 ( .A1(n373), .A2(n372), .ZN(n434) );
  VHSR_NOR2_1 U403 ( .A1(n374), .A2(n437), .ZN(n375) );
  VHSR_AOI32_2 U404 ( .A1(b[0]), .A2(n376), .A3(a[4]), .B1(n375), .B2(n376), 
        .ZN(n433) );
  VHSR_AD1_1 U405 ( .A(n379), .B(n378), .CI(n377), .CO(n367), .S(n380) );
  VHSR_IN_2 U406 ( .I(n380), .ZN(n418) );
  VHSR_AD1_1 U407 ( .A(n383), .B(n382), .CI(n381), .CO(n363), .S(n417) );
  VHSR_NOR2_1 U408 ( .A1(n385), .A2(n384), .ZN(n396) );
  VHSR_CLKNAND2_2 U409 ( .A1(a[7]), .A2(b[6]), .ZN(n387) );
  VHSR_AOI21_2 U410 ( .A1(a[6]), .A2(b[7]), .B(n387), .ZN(n386) );
  VHSR_AOI31_2 U411 ( .A1(a[6]), .A2(n387), .A3(b[7]), .B(n386), .ZN(n388) );
  VHSR_CLKNAND2_2 U412 ( .A1(n389), .A2(n388), .ZN(n391) );
  VHSR_IN_2 U413 ( .I(n392), .ZN(n390) );
  VHSR_MAOI222_2 U414 ( .A(n390), .B(n389), .C(n388), .ZN(n398) );
  VHSR_IAO21_2 U415 ( .A1(n392), .A2(n391), .B(n398), .ZN(n397) );
  VHSR_XNOR2_2 U416 ( .A1(n396), .A2(n397), .ZN(n393) );
  VHSR_CLKNAND2_2 U417 ( .A1(n394), .A2(n393), .ZN(n429) );
  VHSR_OAI21_2 U418 ( .A1(n394), .A2(n393), .B(n429), .ZN(n395) );
  VHSR_CLKNAND2_2 U419 ( .A1(a[7]), .A2(b[7]), .ZN(n428) );
  VHSR_NOR2_1 U420 ( .A1(n399), .A2(n398), .ZN(n430) );
  VHSR_AND3_2 U421 ( .A1(n430), .A2(n400), .A3(n429), .Z(n401) );
  VHSR_NOR2_1 U422 ( .A1(n428), .A2(n401), .ZN(product[15]) );
  VHSR_AD1_1 U423 ( .A(n423), .B(n422), .CI(n421), .CO(n405), .S(product[9])
         );
  VHSR_AD1_1 U424 ( .A(n426), .B(n425), .CI(n424), .CO(n394), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U425 ( .A1(n428), .A2(n427), .ZN(n431) );
  VHSR_XOR3_2 U426 ( .A1(n431), .A2(n430), .A3(n429), .Z(product[14]) );
  VHSR_AOI21_2 U427 ( .A1(n434), .A2(n433), .B(n432), .ZN(product[4]) );
  VHSR_OAI22_2 U428 ( .A1(n438), .A2(n437), .B1(n436), .B2(n435), .ZN(
        product[1]) );
  VHSR_AOI31_2 U429 ( .A1(n442), .A2(n441), .A3(n440), .B(n439), .ZN(
        product[2]) );
endmodule


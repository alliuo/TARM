
module mul8_128 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[4] ,
         \intadd_0/SUM[2] , n237, n238, n239, n240, n241, n242, n243, n244,
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
         n443, n444, n445, n446, n447, n448, n449;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[9] = \intadd_0/SUM[4] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR3_2 U227 ( .A1(n293), .B1(n338), .B2(n372), .ZN(n258) );
  VHSR_NOR2_1 U228 ( .A1(n274), .A2(n273), .ZN(n272) );
  VHSR_NOR2_1 U229 ( .A1(n353), .A2(n391), .ZN(n265) );
  VHSR_NOR2_1 U230 ( .A1(n276), .A2(n272), .ZN(n267) );
  VHSR_NOR2_1 U231 ( .A1(n312), .A2(n351), .ZN(n436) );
  VHSR_INOR2_2 U232 ( .A1(n364), .B1(n378), .ZN(n371) );
  VHSR_INOR3_2 U233 ( .A1(n267), .B1(n373), .B2(n389), .ZN(n327) );
  VHSR_INOR2_2 U234 ( .A1(n403), .B1(n402), .ZN(n434) );
  VHSR_IN_2 U235 ( .I(n351), .ZN(product[0]) );
  VHSR_IN_2 U236 ( .I(n399), .ZN(product[13]) );
  VHSR_INOR2_1 U237 ( .A1(n387), .B1(n386), .ZN(n401) );
  VHSR_INOR2_1 U238 ( .A1(n381), .B1(n357), .ZN(n380) );
  VHSR_NOR2_2 U239 ( .A1(n324), .A2(n328), .ZN(n323) );
  VHSR_INAND2_1 U240 ( .A1(n394), .B1(n392), .ZN(n395) );
  VHSR_AD1_1 U241 ( .A(n422), .B(n440), .CI(n421), .CO(n418), .S(product[5])
         );
  VHSR_AD1_1 U242 ( .A(n417), .B(n416), .CI(n415), .CO(n412), .S(product[8])
         );
  VHSR_AD1_1 U243 ( .A(n411), .B(n410), .CI(n409), .CO(n406), .S(product[10])
         );
  VHSR_AD1_1 U244 ( .A(n424), .B(n423), .CI(n447), .CO(n383), .S(product[3])
         );
  VHSR_AD1_1 U245 ( .A(n420), .B(n419), .CI(n418), .CO(n425), .S(product[6])
         );
  VHSR_AD1_1 U246 ( .A(n414), .B(n413), .CI(n412), .CO(n409), .S(
        \intadd_0/SUM[4] ) );
  VHSR_AD1_1 U247 ( .A(n408), .B(n407), .CI(n406), .CO(n428), .S(
        \intadd_0/SUM[6] ) );
  VHSR_CLKNAND2_2 U248 ( .A1(b[0]), .A2(a[0]), .ZN(n351) );
  VHSR_AOI22_2 U249 ( .A1(b[3]), .A2(a[6]), .B1(a[7]), .B2(b[2]), .ZN(n276) );
  VHSR_IN_2 U250 ( .I(b[3]), .ZN(n373) );
  VHSR_CLKNAND2_2 U251 ( .A1(b[2]), .A2(a[4]), .ZN(n297) );
  VHSR_IN_2 U252 ( .I(a[5]), .ZN(n336) );
  VHSR_NOR3_2 U253 ( .A1(n373), .A2(n297), .A3(n336), .ZN(n274) );
  VHSR_IN_2 U254 ( .I(a[7]), .ZN(n389) );
  VHSR_IN_2 U255 ( .I(b[1]), .ZN(n446) );
  VHSR_NOR2_1 U256 ( .A1(n389), .A2(n446), .ZN(n240) );
  VHSR_AND2_2 U257 ( .A1(a[6]), .A2(b[2]), .Z(n237) );
  VHSR_AOI211_2 U258 ( .A1(a[4]), .A2(b[2]), .B(n373), .C(n336), .ZN(n238) );
  VHSR_MAOI222_2 U259 ( .A(n240), .B(n237), .C(n238), .ZN(n250) );
  VHSR_AOI21_2 U260 ( .A1(b[2]), .A2(a[6]), .B(n238), .ZN(n239) );
  VHSR_IN_2 U261 ( .I(n239), .ZN(n241) );
  VHSR_OAI21_2 U262 ( .A1(n241), .A2(n240), .B(n250), .ZN(n242) );
  VHSR_IN_2 U263 ( .I(n242), .ZN(n283) );
  VHSR_IN_2 U264 ( .I(a[4]), .ZN(n340) );
  VHSR_IN_2 U265 ( .I(b[0]), .ZN(n444) );
  VHSR_NOR4_2 U266 ( .A1(n340), .A2(n336), .A3(n446), .A4(n444), .ZN(n301) );
  VHSR_AOI22_2 U267 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n243) );
  VHSR_NOR2_1 U268 ( .A1(n274), .A2(n243), .ZN(n245) );
  VHSR_AOI22_2 U269 ( .A1(a[7]), .A2(b[0]), .B1(a[6]), .B2(b[1]), .ZN(n247) );
  VHSR_IN_2 U270 ( .I(n247), .ZN(n244) );
  VHSR_MAOI222_2 U271 ( .A(n301), .B(n245), .C(n244), .ZN(n249) );
  VHSR_OAI211_2 U272 ( .A1(n340), .A2(n444), .B(a[5]), .C(b[1]), .ZN(n296) );
  VHSR_CLKNAND2_2 U273 ( .A1(a[6]), .A2(b[0]), .ZN(n295) );
  VHSR_MAOI222_2 U274 ( .A(n297), .B(n296), .C(n295), .ZN(n294) );
  VHSR_NOR2_1 U275 ( .A1(n301), .A2(n245), .ZN(n248) );
  VHSR_IN_2 U276 ( .I(n249), .ZN(n246) );
  VHSR_AOI21_2 U277 ( .A1(n248), .A2(n247), .B(n246), .ZN(n288) );
  VHSR_CLKNAND2_2 U278 ( .A1(n294), .A2(n288), .ZN(n287) );
  VHSR_CLKNAND2_2 U279 ( .A1(n249), .A2(n287), .ZN(n282) );
  VHSR_CLKNAND2_2 U280 ( .A1(n283), .A2(n282), .ZN(n281) );
  VHSR_CLKNAND2_2 U281 ( .A1(n250), .A2(n281), .ZN(n273) );
  VHSR_NAND4_2 U282 ( .A1(a[3]), .A2(a[2]), .A3(b[5]), .A4(b[4]), .ZN(n260) );
  VHSR_CLKNAND2_2 U283 ( .A1(b[7]), .A2(a[2]), .ZN(n252) );
  VHSR_AOI21_2 U284 ( .A1(a[3]), .A2(b[6]), .B(n252), .ZN(n251) );
  VHSR_AOI31_2 U285 ( .A1(a[3]), .A2(n252), .A3(b[6]), .B(n251), .ZN(n259) );
  VHSR_NOR2_1 U286 ( .A1(n260), .A2(n259), .ZN(n261) );
  VHSR_CLKNAND2_2 U287 ( .A1(a[2]), .A2(b[4]), .ZN(n293) );
  VHSR_IN_2 U288 ( .I(b[5]), .ZN(n338) );
  VHSR_IN_2 U289 ( .I(a[3]), .ZN(n372) );
  VHSR_IN_2 U290 ( .I(a[2]), .ZN(n353) );
  VHSR_IN_2 U291 ( .I(b[6]), .ZN(n391) );
  VHSR_IN_2 U292 ( .I(b[7]), .ZN(n309) );
  VHSR_IN_2 U293 ( .I(a[1]), .ZN(n443) );
  VHSR_NOR2_1 U294 ( .A1(n309), .A2(n443), .ZN(n257) );
  VHSR_IN_2 U295 ( .I(n253), .ZN(n271) );
  VHSR_CLKNAND2_2 U296 ( .A1(b[6]), .A2(a[0]), .ZN(n292) );
  VHSR_CLKNAND2_2 U297 ( .A1(b[4]), .A2(a[0]), .ZN(n437) );
  VHSR_NAND3_2 U298 ( .A1(b[5]), .A2(a[1]), .A3(n437), .ZN(n291) );
  VHSR_MAOI222_2 U299 ( .A(n293), .B(n292), .C(n291), .ZN(n290) );
  VHSR_IN_2 U300 ( .I(b[4]), .ZN(n335) );
  VHSR_OAI22_2 U301 ( .A1(n372), .A2(n335), .B1(n353), .B2(n338), .ZN(n254) );
  VHSR_AND2_2 U302 ( .A1(n260), .A2(n254), .Z(n256) );
  VHSR_IN_2 U303 ( .I(a[0]), .ZN(n445) );
  VHSR_OAI22_2 U304 ( .A1(n309), .A2(n445), .B1(n391), .B2(n443), .ZN(n255) );
  VHSR_NOR3_2 U305 ( .A1(n338), .A2(n437), .A3(n443), .ZN(n299) );
  VHSR_AND2_2 U306 ( .A1(n290), .A2(n286), .Z(n285) );
  VHSR_AD1_1 U307 ( .A(n256), .B(n255), .CI(n299), .CO(n280), .S(n286) );
  VHSR_AD1_1 U308 ( .A(n258), .B(n265), .CI(n257), .CO(n253), .S(n277) );
  VHSR_OAI21_2 U309 ( .A1(n285), .A2(n280), .B(n277), .ZN(n279) );
  VHSR_XNOR2_2 U310 ( .A1(n260), .A2(n259), .ZN(n270) );
  VHSR_MAOI222_2 U311 ( .A(n271), .B(n279), .C(n270), .ZN(n269) );
  VHSR_OR2_2 U312 ( .A1(n261), .A2(n269), .Z(n264) );
  VHSR_OAI211_2 U313 ( .A1(n264), .A2(n265), .B(b[7]), .C(a[3]), .ZN(n262) );
  VHSR_IN_2 U314 ( .I(n262), .ZN(n326) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[3]), .A2(b[7]), .ZN(n266) );
  VHSR_OAI21_2 U316 ( .A1(n266), .A2(n265), .B(n264), .ZN(n263) );
  VHSR_OAI31_2 U317 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n334) );
  VHSR_NOR2_1 U318 ( .A1(n373), .A2(n389), .ZN(n268) );
  VHSR_IAO21_2 U319 ( .A1(n268), .A2(n267), .B(n327), .ZN(n333) );
  VHSR_AOI31_2 U320 ( .A1(n271), .A2(n279), .A3(n270), .B(n269), .ZN(n343) );
  VHSR_AOI21_2 U321 ( .A1(n274), .A2(n273), .B(n272), .ZN(n275) );
  VHSR_XNOR2_2 U322 ( .A1(n276), .A2(n275), .ZN(n342) );
  VHSR_OAI32_2 U323 ( .A1(n285), .A2(n280), .A3(n277), .B1(n279), .B2(n285), 
        .ZN(n278) );
  VHSR_IAO21_2 U324 ( .A1(n280), .A2(n279), .B(n278), .ZN(n346) );
  VHSR_OAI21_2 U325 ( .A1(n283), .A2(n282), .B(n281), .ZN(n284) );
  VHSR_IN_2 U326 ( .I(n284), .ZN(n345) );
  VHSR_IAO21_2 U327 ( .A1(n290), .A2(n286), .B(n285), .ZN(n367) );
  VHSR_OAI21_2 U328 ( .A1(n294), .A2(n288), .B(n287), .ZN(n289) );
  VHSR_IN_2 U329 ( .I(n289), .ZN(n366) );
  VHSR_AOI31_2 U330 ( .A1(n293), .A2(n292), .A3(n291), .B(n290), .ZN(n370) );
  VHSR_AOI31_2 U331 ( .A1(n297), .A2(n296), .A3(n295), .B(n294), .ZN(n369) );
  VHSR_AOI22_2 U332 ( .A1(b[5]), .A2(a[0]), .B1(b[4]), .B2(a[1]), .ZN(n298) );
  VHSR_NOR2_1 U333 ( .A1(n299), .A2(n298), .ZN(n385) );
  VHSR_CLKNAND2_2 U334 ( .A1(a[4]), .A2(b[4]), .ZN(n312) );
  VHSR_CLKNAND2_2 U335 ( .A1(a[5]), .A2(b[0]), .ZN(n300) );
  VHSR_OAI32_2 U336 ( .A1(n301), .A2(n446), .A3(n340), .B1(n300), .B2(n301), 
        .ZN(n384) );
  VHSR_CLKNAND2_2 U337 ( .A1(a[6]), .A2(b[6]), .ZN(n404) );
  VHSR_IN_2 U338 ( .I(n404), .ZN(n431) );
  VHSR_CLKNAND2_2 U339 ( .A1(a[4]), .A2(b[6]), .ZN(n308) );
  VHSR_IN_2 U340 ( .I(n308), .ZN(n314) );
  VHSR_CLKNAND2_2 U341 ( .A1(a[5]), .A2(b[7]), .ZN(n303) );
  VHSR_CLKNAND2_2 U342 ( .A1(a[6]), .A2(b[4]), .ZN(n306) );
  VHSR_IN_2 U343 ( .I(n306), .ZN(n315) );
  VHSR_CLKNAND2_2 U344 ( .A1(a[7]), .A2(b[5]), .ZN(n302) );
  VHSR_OAI22_2 U345 ( .A1(n314), .A2(n303), .B1(n315), .B2(n302), .ZN(n305) );
  VHSR_CLKNAND2_2 U346 ( .A1(n306), .A2(n308), .ZN(n329) );
  VHSR_CLKNAND2_2 U347 ( .A1(a[5]), .A2(b[5]), .ZN(n313) );
  VHSR_NOR4_2 U348 ( .A1(n389), .A2(n309), .A3(n329), .A4(n313), .ZN(n304) );
  VHSR_AOI21_2 U349 ( .A1(n431), .A2(n305), .B(n304), .ZN(n387) );
  VHSR_OAI21_2 U350 ( .A1(n431), .A2(n305), .B(n387), .ZN(n322) );
  VHSR_NOR3_2 U351 ( .A1(n389), .A2(n338), .A3(n306), .ZN(n396) );
  VHSR_AOI22_2 U352 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n307) );
  VHSR_NOR2_1 U353 ( .A1(n396), .A2(n307), .ZN(n318) );
  VHSR_NOR2_1 U354 ( .A1(n313), .A2(n312), .ZN(n317) );
  VHSR_NOR3_2 U355 ( .A1(n336), .A2(n309), .A3(n308), .ZN(n394) );
  VHSR_AOI22_2 U356 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n310) );
  VHSR_NOR2_1 U357 ( .A1(n394), .A2(n310), .ZN(n316) );
  VHSR_IN_2 U358 ( .I(n311), .ZN(n324) );
  VHSR_IN_2 U359 ( .I(n312), .ZN(n416) );
  VHSR_NOR2_1 U360 ( .A1(n416), .A2(n313), .ZN(n330) );
  VHSR_AOI22_2 U361 ( .A1(n315), .A2(n314), .B1(n330), .B2(n329), .ZN(n328) );
  VHSR_AD1_1 U362 ( .A(n318), .B(n317), .CI(n316), .CO(n319), .S(n311) );
  VHSR_NOR2_1 U363 ( .A1(n323), .A2(n319), .ZN(n321) );
  VHSR_CLKNAND2_2 U364 ( .A1(n323), .A2(n319), .ZN(n320) );
  VHSR_NOR2_1 U365 ( .A1(n321), .A2(n322), .ZN(n386) );
  VHSR_AOI22_2 U366 ( .A1(n322), .A2(n321), .B1(n320), .B2(n386), .ZN(n429) );
  VHSR_AOI21_2 U367 ( .A1(n328), .A2(n324), .B(n323), .ZN(n408) );
  VHSR_AD1_1 U368 ( .A(n327), .B(n326), .CI(n325), .CO(n430), .S(n407) );
  VHSR_OAI21_2 U369 ( .A1(n330), .A2(n329), .B(n328), .ZN(n331) );
  VHSR_IN_2 U370 ( .I(n331), .ZN(n411) );
  VHSR_AD1_1 U371 ( .A(n334), .B(n333), .CI(n332), .CO(n325), .S(n410) );
  VHSR_NOR2_1 U372 ( .A1(n336), .A2(n335), .ZN(n339) );
  VHSR_OAI21_2 U373 ( .A1(n340), .A2(n338), .B(n339), .ZN(n337) );
  VHSR_OAI31_2 U374 ( .A1(n340), .A2(n339), .A3(n338), .B(n337), .ZN(n414) );
  VHSR_AD1_1 U375 ( .A(n343), .B(n342), .CI(n341), .CO(n332), .S(n413) );
  VHSR_AD1_1 U376 ( .A(n346), .B(n345), .CI(n344), .CO(n341), .S(n417) );
  VHSR_IN_2 U377 ( .I(b[2]), .ZN(n352) );
  VHSR_NOR2_1 U378 ( .A1(n352), .A2(n372), .ZN(n348) );
  VHSR_OAI21_2 U379 ( .A1(n373), .A2(n353), .B(n348), .ZN(n347) );
  VHSR_OAI31_2 U380 ( .A1(n373), .A2(n348), .A3(n353), .B(n347), .ZN(n360) );
  VHSR_NOR4_2 U381 ( .A1(n373), .A2(n352), .A3(n445), .A4(n443), .ZN(n358) );
  VHSR_NOR4_2 U382 ( .A1(n446), .A2(n444), .A3(n372), .A4(n353), .ZN(n359) );
  VHSR_MAOI222_2 U383 ( .A(n360), .B(n358), .C(n359), .ZN(n364) );
  VHSR_CLKNAND2_2 U384 ( .A1(b[2]), .A2(a[1]), .ZN(n349) );
  VHSR_OAI32_2 U385 ( .A1(n358), .A2(n445), .A3(n373), .B1(n349), .B2(n358), 
        .ZN(n424) );
  VHSR_CLKNAND2_2 U386 ( .A1(b[0]), .A2(a[3]), .ZN(n350) );
  VHSR_OAI32_2 U387 ( .A1(n359), .A2(n353), .A3(n446), .B1(n350), .B2(n359), 
        .ZN(n423) );
  VHSR_CLKNAND2_2 U388 ( .A1(b[1]), .A2(a[1]), .ZN(n448) );
  VHSR_AOI22_2 U389 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n449) );
  VHSR_CLKNAND2_2 U390 ( .A1(b[2]), .A2(a[2]), .ZN(n377) );
  VHSR_OAI22_2 U391 ( .A1(n448), .A2(n449), .B1(n351), .B2(n377), .ZN(n447) );
  VHSR_OAI211_2 U392 ( .A1(n445), .A2(n352), .B(b[3]), .C(a[1]), .ZN(n355) );
  VHSR_OAI211_2 U393 ( .A1(n444), .A2(n353), .B(b[1]), .C(a[3]), .ZN(n354) );
  VHSR_AND2_2 U394 ( .A1(n355), .A2(n354), .Z(n356) );
  VHSR_MAOI222_2 U395 ( .A(n377), .B(n355), .C(n354), .ZN(n357) );
  VHSR_AOI21_2 U396 ( .A1(n356), .A2(n377), .B(n357), .ZN(n382) );
  VHSR_CLKNAND2_2 U397 ( .A1(n383), .A2(n382), .ZN(n381) );
  VHSR_IN_2 U398 ( .I(n358), .ZN(n363) );
  VHSR_NOR2_1 U399 ( .A1(n360), .A2(n359), .ZN(n362) );
  VHSR_AOI22_2 U400 ( .A1(n360), .A2(n359), .B1(n363), .B2(n362), .ZN(n361) );
  VHSR_OAI21_2 U401 ( .A1(n363), .A2(n362), .B(n361), .ZN(n379) );
  VHSR_NOR2_1 U402 ( .A1(n380), .A2(n379), .ZN(n378) );
  VHSR_AOI211_2 U403 ( .A1(n371), .A2(n377), .B(n372), .C(n373), .ZN(n427) );
  VHSR_AD1_1 U404 ( .A(n367), .B(n366), .CI(n365), .CO(n344), .S(n426) );
  VHSR_AD1_1 U405 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(n420) );
  VHSR_IN_2 U406 ( .I(n371), .ZN(n376) );
  VHSR_NOR2_1 U407 ( .A1(n373), .A2(n372), .ZN(n375) );
  VHSR_AOI21_2 U408 ( .A1(n377), .A2(n375), .B(n376), .ZN(n374) );
  VHSR_AOI31_2 U409 ( .A1(n377), .A2(n376), .A3(n375), .B(n374), .ZN(n419) );
  VHSR_AOI21_2 U410 ( .A1(n380), .A2(n379), .B(n378), .ZN(n422) );
  VHSR_CLKNAND2_2 U411 ( .A1(a[4]), .A2(b[0]), .ZN(n438) );
  VHSR_OAI21_2 U412 ( .A1(n383), .A2(n382), .B(n381), .ZN(n442) );
  VHSR_AOI211_2 U413 ( .A1(n438), .A2(n437), .B(n436), .C(n442), .ZN(n440) );
  VHSR_AD1_1 U414 ( .A(n385), .B(n436), .CI(n384), .CO(n368), .S(n421) );
  VHSR_CLKNAND2_2 U415 ( .A1(b[7]), .A2(a[6]), .ZN(n390) );
  VHSR_OAI21_2 U416 ( .A1(n391), .A2(n389), .B(n390), .ZN(n388) );
  VHSR_OAI31_2 U417 ( .A1(n391), .A2(n390), .A3(n389), .B(n388), .ZN(n392) );
  VHSR_IN_2 U418 ( .I(n392), .ZN(n393) );
  VHSR_MAOI222_2 U419 ( .A(n396), .B(n394), .C(n393), .ZN(n403) );
  VHSR_OAI21_2 U420 ( .A1(n396), .A2(n395), .B(n403), .ZN(n400) );
  VHSR_CLKXOR2_2 U421 ( .A1(n401), .A2(n400), .Z(n397) );
  VHSR_CLKNAND2_2 U422 ( .A1(n398), .A2(n397), .ZN(n433) );
  VHSR_OAI21_2 U423 ( .A1(n398), .A2(n397), .B(n433), .ZN(n399) );
  VHSR_CLKNAND2_2 U424 ( .A1(a[7]), .A2(b[7]), .ZN(n432) );
  VHSR_NOR2_1 U425 ( .A1(n401), .A2(n400), .ZN(n402) );
  VHSR_AND3_2 U426 ( .A1(n434), .A2(n404), .A3(n433), .Z(n405) );
  VHSR_NOR2_1 U427 ( .A1(n432), .A2(n405), .ZN(product[15]) );
  VHSR_AD1_1 U428 ( .A(n427), .B(n426), .CI(n425), .CO(n415), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U429 ( .A(n430), .B(n429), .CI(n428), .CO(n398), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U430 ( .A1(n432), .A2(n431), .ZN(n435) );
  VHSR_XOR3_2 U431 ( .A1(n435), .A2(n434), .A3(n433), .Z(product[14]) );
  VHSR_AOI21_2 U432 ( .A1(n438), .A2(n437), .B(n436), .ZN(n439) );
  VHSR_IN_2 U433 ( .I(n439), .ZN(n441) );
  VHSR_AOI21_2 U434 ( .A1(n442), .A2(n441), .B(n440), .ZN(product[4]) );
  VHSR_OAI22_2 U435 ( .A1(n446), .A2(n445), .B1(n444), .B2(n443), .ZN(
        product[1]) );
  VHSR_AOI21_2 U436 ( .A1(n449), .A2(n448), .B(n447), .ZN(product[2]) );
endmodule


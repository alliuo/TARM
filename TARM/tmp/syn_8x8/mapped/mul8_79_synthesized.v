
module mul8_79 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n220, n221,
         n222, n223, n224, n225, n226, n227, n228, n229, n230, n231, n232,
         n233, n234, n235, n236, n237, n238, n239, n240, n241, n242, n243,
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
         n409, n410, n411, n412, n413, n414, n415;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U210 ( .A1(n253), .A2(n252), .ZN(n251) );
  VHSR_NOR2_1 U211 ( .A1(n255), .A2(n251), .ZN(n244) );
  VHSR_INOR3_2 U212 ( .A1(n244), .B1(n332), .B2(n286), .ZN(n304) );
  VHSR_NOR2_1 U213 ( .A1(n408), .A2(n407), .ZN(n406) );
  VHSR_INOR2_2 U214 ( .A1(n373), .B1(n372), .ZN(n404) );
  VHSR_IN_2 U215 ( .I(n369), .ZN(product[13]) );
  VHSR_NOR2_2 U216 ( .A1(n359), .A2(n358), .ZN(n371) );
  VHSR_INAND2_1 U217 ( .A1(n364), .B1(n362), .ZN(n365) );
  VHSR_AD1_1 U218 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(product[9])
         );
  VHSR_AD1_1 U219 ( .A(n391), .B(n390), .CI(n413), .CO(n347), .S(product[3])
         );
  VHSR_AD1_1 U220 ( .A(n406), .B(n389), .CI(n388), .CO(n392), .S(product[5])
         );
  VHSR_AD1_1 U221 ( .A(n387), .B(n386), .CI(n385), .CO(n382), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U222 ( .A(n384), .B(n383), .CI(n382), .CO(n379), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U223 ( .A(n378), .B(n377), .CI(n376), .CO(n395), .S(product[10])
         );
  VHSR_AOI22_2 U224 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n255) );
  VHSR_IN_2 U225 ( .I(b[3]), .ZN(n332) );
  VHSR_IN_2 U226 ( .I(b[2]), .ZN(n330) );
  VHSR_IN_2 U227 ( .I(a[5]), .ZN(n291) );
  VHSR_IN_2 U228 ( .I(a[4]), .ZN(n290) );
  VHSR_NOR4_2 U229 ( .A1(n332), .A2(n330), .A3(n291), .A4(n290), .ZN(n253) );
  VHSR_IN_2 U230 ( .I(a[7]), .ZN(n286) );
  VHSR_IN_2 U231 ( .I(b[1]), .ZN(n412) );
  VHSR_NOR2_1 U232 ( .A1(n286), .A2(n412), .ZN(n221) );
  VHSR_AOI211_2 U233 ( .A1(b[2]), .A2(a[4]), .B(n332), .C(n291), .ZN(n222) );
  VHSR_CLKNAND2_2 U234 ( .A1(a[6]), .A2(b[2]), .ZN(n224) );
  VHSR_IN_2 U235 ( .I(n224), .ZN(n220) );
  VHSR_MAOI222_2 U236 ( .A(n221), .B(n222), .C(n220), .ZN(n234) );
  VHSR_AOI21_2 U237 ( .A1(b[1]), .A2(a[7]), .B(n222), .ZN(n225) );
  VHSR_IN_2 U238 ( .I(n234), .ZN(n223) );
  VHSR_AOI21_2 U239 ( .A1(n225), .A2(n224), .B(n223), .ZN(n262) );
  VHSR_CLKNAND2_2 U240 ( .A1(a[6]), .A2(b[1]), .ZN(n231) );
  VHSR_IN_2 U241 ( .I(n231), .ZN(n228) );
  VHSR_IN_2 U242 ( .I(b[0]), .ZN(n410) );
  VHSR_NOR4_2 U243 ( .A1(n291), .A2(n290), .A3(n412), .A4(n410), .ZN(n280) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[2]), .A2(a[5]), .ZN(n227) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[3]), .A2(a[4]), .ZN(n226) );
  VHSR_AOI21_2 U246 ( .A1(n227), .A2(n226), .B(n253), .ZN(n229) );
  VHSR_MAOI222_2 U247 ( .A(n228), .B(n280), .C(n229), .ZN(n233) );
  VHSR_CLKNAND2_2 U248 ( .A1(b[2]), .A2(a[4]), .ZN(n276) );
  VHSR_OAI21_2 U249 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n275) );
  VHSR_OAI211_2 U250 ( .A1(n290), .A2(n410), .B(a[5]), .C(b[1]), .ZN(n274) );
  VHSR_MAOI222_2 U251 ( .A(n276), .B(n275), .C(n274), .ZN(n273) );
  VHSR_NOR2_1 U252 ( .A1(n280), .A2(n229), .ZN(n232) );
  VHSR_IN_2 U253 ( .I(n233), .ZN(n230) );
  VHSR_AOI21_2 U254 ( .A1(n232), .A2(n231), .B(n230), .ZN(n265) );
  VHSR_CLKNAND2_2 U255 ( .A1(n273), .A2(n265), .ZN(n264) );
  VHSR_CLKNAND2_2 U256 ( .A1(n233), .A2(n264), .ZN(n261) );
  VHSR_CLKNAND2_2 U257 ( .A1(n262), .A2(n261), .ZN(n260) );
  VHSR_CLKNAND2_2 U258 ( .A1(n234), .A2(n260), .ZN(n252) );
  VHSR_IN_2 U259 ( .I(b[7]), .ZN(n288) );
  VHSR_IN_2 U260 ( .I(a[3]), .ZN(n329) );
  VHSR_IN_2 U261 ( .I(b[6]), .ZN(n289) );
  VHSR_IN_2 U262 ( .I(a[2]), .ZN(n331) );
  VHSR_OAI22_2 U263 ( .A1(n289), .A2(n329), .B1(n288), .B2(n331), .ZN(n250) );
  VHSR_AOI22_2 U264 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n241) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[2]), .ZN(n272) );
  VHSR_NAND3_2 U266 ( .A1(a[3]), .A2(b[5]), .A3(n272), .ZN(n240) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[7]), .A2(a[2]), .ZN(n235) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[6]), .A2(a[1]), .ZN(n237) );
  VHSR_OAI22_2 U269 ( .A1(n241), .A2(n240), .B1(n235), .B2(n237), .ZN(n242) );
  VHSR_IN_2 U270 ( .I(b[4]), .ZN(n348) );
  VHSR_IN_2 U271 ( .I(a[0]), .ZN(n411) );
  VHSR_OAI211_2 U272 ( .A1(n348), .A2(n411), .B(b[5]), .C(a[1]), .ZN(n271) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[6]), .A2(a[0]), .ZN(n270) );
  VHSR_MAOI222_2 U274 ( .A(n272), .B(n271), .C(n270), .ZN(n269) );
  VHSR_NAND4_2 U275 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n247) );
  VHSR_IN_2 U276 ( .I(b[5]), .ZN(n285) );
  VHSR_OAI22_2 U277 ( .A1(n348), .A2(n329), .B1(n285), .B2(n331), .ZN(n236) );
  VHSR_AND2_2 U278 ( .A1(n247), .A2(n236), .Z(n239) );
  VHSR_OAI21_2 U279 ( .A1(n288), .A2(n411), .B(n237), .ZN(n238) );
  VHSR_IN_2 U280 ( .I(a[1]), .ZN(n409) );
  VHSR_NOR4_2 U281 ( .A1(n348), .A2(n285), .A3(n409), .A4(n411), .ZN(n278) );
  VHSR_AND2_2 U282 ( .A1(n269), .A2(n268), .Z(n267) );
  VHSR_AD1_1 U283 ( .A(n239), .B(n238), .CI(n278), .CO(n256), .S(n268) );
  VHSR_AOI21_2 U284 ( .A1(n241), .A2(n240), .B(n242), .ZN(n259) );
  VHSR_OAI32_2 U285 ( .A1(n242), .A2(n267), .A3(n256), .B1(n259), .B2(n242), 
        .ZN(n248) );
  VHSR_CLKNAND2_2 U286 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_CLKNAND2_2 U287 ( .A1(n250), .A2(n246), .ZN(n245) );
  VHSR_NOR3_2 U288 ( .A1(n288), .A2(n329), .A3(n245), .ZN(n303) );
  VHSR_NOR2_1 U289 ( .A1(n332), .A2(n286), .ZN(n243) );
  VHSR_IAO21_2 U290 ( .A1(n244), .A2(n243), .B(n304), .ZN(n307) );
  VHSR_OAI32_2 U291 ( .A1(n303), .A2(n329), .A3(n288), .B1(n245), .B2(n303), 
        .ZN(n306) );
  VHSR_OAI21_2 U292 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U293 ( .A1(n250), .A2(n249), .ZN(n314) );
  VHSR_AOI21_2 U294 ( .A1(n253), .A2(n252), .B(n251), .ZN(n254) );
  VHSR_XNOR2_2 U295 ( .A1(n255), .A2(n254), .ZN(n313) );
  VHSR_NOR2_1 U296 ( .A1(n267), .A2(n256), .ZN(n258) );
  VHSR_AOI22_2 U297 ( .A1(n267), .A2(n256), .B1(n259), .B2(n258), .ZN(n257) );
  VHSR_OAI21_2 U298 ( .A1(n259), .A2(n258), .B(n257), .ZN(n319) );
  VHSR_OAI21_2 U299 ( .A1(n262), .A2(n261), .B(n260), .ZN(n263) );
  VHSR_IN_2 U300 ( .I(n263), .ZN(n318) );
  VHSR_OAI21_2 U301 ( .A1(n273), .A2(n265), .B(n264), .ZN(n266) );
  VHSR_IN_2 U302 ( .I(n266), .ZN(n336) );
  VHSR_IAO21_2 U303 ( .A1(n269), .A2(n268), .B(n267), .ZN(n335) );
  VHSR_AOI31_2 U304 ( .A1(n272), .A2(n271), .A3(n270), .B(n269), .ZN(n345) );
  VHSR_AOI31_2 U305 ( .A1(n276), .A2(n275), .A3(n274), .B(n273), .ZN(n344) );
  VHSR_CLKNAND2_2 U306 ( .A1(b[5]), .A2(a[0]), .ZN(n277) );
  VHSR_OAI32_2 U307 ( .A1(n278), .A2(n409), .A3(n348), .B1(n277), .B2(n278), 
        .ZN(n353) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[4]), .ZN(n293) );
  VHSR_IN_2 U309 ( .I(n293), .ZN(n383) );
  VHSR_NOR2_1 U310 ( .A1(n410), .A2(n411), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U311 ( .A1(n383), .A2(product[0]), .ZN(n350) );
  VHSR_IN_2 U312 ( .I(n350), .ZN(n352) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[4]), .A2(b[1]), .ZN(n279) );
  VHSR_OAI32_2 U314 ( .A1(n280), .A2(n291), .A3(n410), .B1(n279), .B2(n280), 
        .ZN(n351) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[6]), .A2(b[6]), .ZN(n374) );
  VHSR_IN_2 U316 ( .I(n374), .ZN(n401) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[6]), .A2(b[4]), .ZN(n311) );
  VHSR_NAND3_2 U318 ( .A1(a[7]), .A2(b[5]), .A3(n311), .ZN(n282) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[4]), .A2(b[6]), .ZN(n310) );
  VHSR_NAND3_2 U320 ( .A1(b[7]), .A2(a[5]), .A3(n310), .ZN(n281) );
  VHSR_CLKNAND2_2 U321 ( .A1(n282), .A2(n281), .ZN(n284) );
  VHSR_MAOI222_2 U322 ( .A(n374), .B(n282), .C(n281), .ZN(n358) );
  VHSR_IN_2 U323 ( .I(n358), .ZN(n283) );
  VHSR_OAI21_2 U324 ( .A1(n401), .A2(n284), .B(n283), .ZN(n299) );
  VHSR_NOR3_2 U325 ( .A1(n291), .A2(n285), .A3(n293), .ZN(n315) );
  VHSR_NOR3_2 U326 ( .A1(n286), .A2(n311), .A3(n285), .ZN(n366) );
  VHSR_AOI22_2 U327 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n287) );
  VHSR_NOR2_1 U328 ( .A1(n366), .A2(n287), .ZN(n295) );
  VHSR_NOR4_2 U329 ( .A1(n291), .A2(n290), .A3(n289), .A4(n288), .ZN(n364) );
  VHSR_AOI22_2 U330 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n292) );
  VHSR_NOR2_1 U331 ( .A1(n364), .A2(n292), .ZN(n294) );
  VHSR_NAND3_2 U332 ( .A1(b[5]), .A2(a[5]), .A3(n293), .ZN(n309) );
  VHSR_MAOI222_2 U333 ( .A(n311), .B(n310), .C(n309), .ZN(n308) );
  VHSR_AND2_2 U334 ( .A1(n301), .A2(n308), .Z(n300) );
  VHSR_AD1_1 U335 ( .A(n315), .B(n295), .CI(n294), .CO(n296), .S(n301) );
  VHSR_NOR2_1 U336 ( .A1(n300), .A2(n296), .ZN(n298) );
  VHSR_CLKNAND2_2 U337 ( .A1(n300), .A2(n296), .ZN(n297) );
  VHSR_NOR2_1 U338 ( .A1(n298), .A2(n299), .ZN(n359) );
  VHSR_AOI22_2 U339 ( .A1(n299), .A2(n298), .B1(n297), .B2(n359), .ZN(n399) );
  VHSR_IAO21_2 U340 ( .A1(n301), .A2(n308), .B(n300), .ZN(n397) );
  VHSR_AD1_1 U341 ( .A(n304), .B(n303), .CI(n302), .CO(n400), .S(n396) );
  VHSR_AD1_1 U342 ( .A(n307), .B(n306), .CI(n305), .CO(n302), .S(n378) );
  VHSR_AOI31_2 U343 ( .A1(n311), .A2(n310), .A3(n309), .B(n308), .ZN(n377) );
  VHSR_AD1_1 U344 ( .A(n314), .B(n313), .CI(n312), .CO(n305), .S(n381) );
  VHSR_AOI22_2 U345 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n316) );
  VHSR_NOR2_1 U346 ( .A1(n316), .A2(n315), .ZN(n380) );
  VHSR_AD1_1 U347 ( .A(n319), .B(n318), .CI(n317), .CO(n312), .S(n384) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[2]), .A2(a[0]), .ZN(n414) );
  VHSR_NOR3_2 U349 ( .A1(n332), .A2(n409), .A3(n414), .ZN(n328) );
  VHSR_AOI22_2 U350 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n320) );
  VHSR_NOR2_1 U351 ( .A1(n328), .A2(n320), .ZN(n391) );
  VHSR_NOR4_2 U352 ( .A1(n412), .A2(n410), .A3(n331), .A4(n329), .ZN(n327) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[0]), .A2(a[3]), .ZN(n321) );
  VHSR_OAI32_2 U354 ( .A1(n327), .A2(n331), .A3(n412), .B1(n321), .B2(n327), 
        .ZN(n390) );
  VHSR_AOI22_2 U355 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n415) );
  VHSR_NOR2_1 U356 ( .A1(n415), .A2(n414), .ZN(n413) );
  VHSR_OAI211_2 U357 ( .A1(n331), .A2(n410), .B(b[1]), .C(a[3]), .ZN(n323) );
  VHSR_IN_2 U358 ( .I(n323), .ZN(n322) );
  VHSR_AOI31_2 U359 ( .A1(a[1]), .A2(b[3]), .A3(n414), .B(n322), .ZN(n325) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[2]), .A2(a[2]), .ZN(n342) );
  VHSR_NAND3_2 U361 ( .A1(a[1]), .A2(b[3]), .A3(n414), .ZN(n324) );
  VHSR_MAOI222_2 U362 ( .A(n342), .B(n324), .C(n323), .ZN(n326) );
  VHSR_AOI21_2 U363 ( .A1(n325), .A2(n342), .B(n326), .ZN(n346) );
  VHSR_AOI21_2 U364 ( .A1(n347), .A2(n346), .B(n326), .ZN(n356) );
  VHSR_NOR2_1 U365 ( .A1(n328), .A2(n327), .ZN(n355) );
  VHSR_CLKNAND2_2 U366 ( .A1(b[3]), .A2(a[3]), .ZN(n338) );
  VHSR_OAI22_2 U367 ( .A1(n332), .A2(n331), .B1(n330), .B2(n329), .ZN(n333) );
  VHSR_OAI21_2 U368 ( .A1(n338), .A2(n342), .B(n333), .ZN(n354) );
  VHSR_AOI21_2 U369 ( .A1(n337), .A2(n342), .B(n338), .ZN(n387) );
  VHSR_AD1_1 U370 ( .A(n336), .B(n335), .CI(n334), .CO(n317), .S(n386) );
  VHSR_IN_2 U371 ( .I(n337), .ZN(n341) );
  VHSR_IN_2 U372 ( .I(n338), .ZN(n340) );
  VHSR_AOI21_2 U373 ( .A1(n342), .A2(n340), .B(n341), .ZN(n339) );
  VHSR_AOI31_2 U374 ( .A1(n342), .A2(n341), .A3(n340), .B(n339), .ZN(n394) );
  VHSR_AD1_1 U375 ( .A(n345), .B(n344), .CI(n343), .CO(n334), .S(n393) );
  VHSR_XNOR2_2 U376 ( .A1(n347), .A2(n346), .ZN(n408) );
  VHSR_NOR2_1 U377 ( .A1(n348), .A2(n411), .ZN(n349) );
  VHSR_AOI32_2 U378 ( .A1(b[0]), .A2(n350), .A3(a[4]), .B1(n349), .B2(n350), 
        .ZN(n407) );
  VHSR_AD1_1 U379 ( .A(n353), .B(n352), .CI(n351), .CO(n343), .S(n389) );
  VHSR_AD1_1 U380 ( .A(n356), .B(n355), .CI(n354), .CO(n337), .S(n357) );
  VHSR_IN_2 U381 ( .I(n357), .ZN(n388) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[7]), .A2(b[6]), .ZN(n361) );
  VHSR_AOI21_2 U383 ( .A1(a[6]), .A2(b[7]), .B(n361), .ZN(n360) );
  VHSR_AOI31_2 U384 ( .A1(a[6]), .A2(n361), .A3(b[7]), .B(n360), .ZN(n362) );
  VHSR_IN_2 U385 ( .I(n362), .ZN(n363) );
  VHSR_MAOI222_2 U386 ( .A(n366), .B(n364), .C(n363), .ZN(n373) );
  VHSR_OAI21_2 U387 ( .A1(n366), .A2(n365), .B(n373), .ZN(n370) );
  VHSR_CLKXOR2_2 U388 ( .A1(n371), .A2(n370), .Z(n367) );
  VHSR_CLKNAND2_2 U389 ( .A1(n368), .A2(n367), .ZN(n403) );
  VHSR_OAI21_2 U390 ( .A1(n368), .A2(n367), .B(n403), .ZN(n369) );
  VHSR_CLKNAND2_2 U391 ( .A1(a[7]), .A2(b[7]), .ZN(n402) );
  VHSR_NOR2_1 U392 ( .A1(n371), .A2(n370), .ZN(n372) );
  VHSR_AND3_2 U393 ( .A1(n404), .A2(n374), .A3(n403), .Z(n375) );
  VHSR_NOR2_1 U394 ( .A1(n402), .A2(n375), .ZN(product[15]) );
  VHSR_AD1_1 U395 ( .A(n394), .B(n393), .CI(n392), .CO(n385), .S(product[6])
         );
  VHSR_AD1_1 U396 ( .A(n397), .B(n396), .CI(n395), .CO(n398), .S(product[11])
         );
  VHSR_AD1_1 U397 ( .A(n400), .B(n399), .CI(n398), .CO(n368), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U398 ( .A1(n402), .A2(n401), .ZN(n405) );
  VHSR_XOR3_2 U399 ( .A1(n405), .A2(n404), .A3(n403), .Z(product[14]) );
  VHSR_AOI21_2 U400 ( .A1(n408), .A2(n407), .B(n406), .ZN(product[4]) );
  VHSR_OAI22_2 U401 ( .A1(n412), .A2(n411), .B1(n410), .B2(n409), .ZN(
        product[1]) );
  VHSR_AOI21_2 U402 ( .A1(n415), .A2(n414), .B(n413), .ZN(product[2]) );
endmodule


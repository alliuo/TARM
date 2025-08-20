
module mul8_46 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n223, n224,
         n225, n226, n227, n228, n229, n230, n231, n232, n233, n234, n235,
         n236, n237, n238, n239, n240, n241, n242, n243, n244, n245, n246,
         n247, n248, n249, n250, n251, n252, n253, n254, n255, n256, n257,
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
         n412, n413, n414;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U213 ( .A1(n273), .B1(a[5]), .B2(b[3]), .ZN(n223) );
  VHSR_INOR2_2 U214 ( .A1(n233), .B1(n259), .ZN(n252) );
  VHSR_INOR2_2 U215 ( .A1(n328), .B1(n342), .ZN(n331) );
  VHSR_INOR2_2 U216 ( .A1(n231), .B1(n262), .ZN(n261) );
  VHSR_NOR2_1 U217 ( .A1(n243), .A2(n242), .ZN(n303) );
  VHSR_NOR2_1 U218 ( .A1(n289), .A2(n348), .ZN(n382) );
  VHSR_IN_2 U219 ( .I(n368), .ZN(product[13]) );
  VHSR_INOR2_1 U220 ( .A1(n372), .B1(n371), .ZN(n403) );
  VHSR_AD1_1 U221 ( .A(n380), .B(n379), .CI(n378), .CO(n375), .S(product[9])
         );
  VHSR_AD1_1 U222 ( .A(n387), .B(n386), .CI(n412), .CO(n347), .S(product[3])
         );
  VHSR_AD1_1 U223 ( .A(n405), .B(n385), .CI(n384), .CO(n388), .S(product[5])
         );
  VHSR_AD1_1 U224 ( .A(n383), .B(n382), .CI(n381), .CO(n378), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n377), .B(n376), .CI(n375), .CO(n394), .S(product[10])
         );
  VHSR_CLKNAND2_2 U226 ( .A1(b[3]), .A2(a[7]), .ZN(n243) );
  VHSR_IN_2 U227 ( .I(b[3]), .ZN(n335) );
  VHSR_IN_2 U228 ( .I(a[6]), .ZN(n279) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n285) );
  VHSR_IN_2 U230 ( .I(b[2]), .ZN(n320) );
  VHSR_OAI22_2 U231 ( .A1(n335), .A2(n279), .B1(n285), .B2(n320), .ZN(n254) );
  VHSR_IN_2 U232 ( .I(b[1]), .ZN(n411) );
  VHSR_IN_2 U233 ( .I(a[4]), .ZN(n289) );
  VHSR_NOR2_1 U234 ( .A1(n320), .A2(n289), .ZN(n273) );
  VHSR_OAI21_2 U235 ( .A1(n411), .A2(n285), .B(n223), .ZN(n232) );
  VHSR_IN_2 U236 ( .I(a[5]), .ZN(n290) );
  VHSR_NOR4_2 U237 ( .A1(n273), .A2(n290), .A3(n243), .A4(n411), .ZN(n224) );
  VHSR_AOI31_2 U238 ( .A1(b[2]), .A2(a[6]), .A3(n232), .B(n224), .ZN(n233) );
  VHSR_NOR2_1 U239 ( .A1(n279), .A2(n411), .ZN(n228) );
  VHSR_IN_2 U240 ( .I(b[0]), .ZN(n409) );
  VHSR_NOR4_2 U241 ( .A1(n290), .A2(n289), .A3(n411), .A4(n409), .ZN(n278) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[2]), .A2(a[5]), .ZN(n227) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[3]), .A2(a[4]), .ZN(n226) );
  VHSR_NAND4_2 U244 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n251) );
  VHSR_IN_2 U245 ( .I(n251), .ZN(n225) );
  VHSR_AOI21_2 U246 ( .A1(n227), .A2(n226), .B(n225), .ZN(n229) );
  VHSR_MAOI222_2 U247 ( .A(n228), .B(n278), .C(n229), .ZN(n231) );
  VHSR_AOI211_2 U248 ( .A1(a[4]), .A2(b[0]), .B(n290), .C(n411), .ZN(n272) );
  VHSR_AOI21_2 U249 ( .A1(n285), .A2(n279), .B(n409), .ZN(n271) );
  VHSR_MAOI222_2 U250 ( .A(n273), .B(n272), .C(n271), .ZN(n270) );
  VHSR_OR2_2 U251 ( .A1(n278), .A2(n229), .Z(n230) );
  VHSR_AOI32_2 U252 ( .A1(b[1]), .A2(n231), .A3(a[6]), .B1(n230), .B2(n231), 
        .ZN(n263) );
  VHSR_NOR2_1 U253 ( .A1(n270), .A2(n263), .ZN(n262) );
  VHSR_AOI32_2 U254 ( .A1(b[2]), .A2(n233), .A3(a[6]), .B1(n232), .B2(n233), 
        .ZN(n260) );
  VHSR_NOR2_1 U255 ( .A1(n261), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U256 ( .A1(n252), .A2(n251), .ZN(n250) );
  VHSR_CLKNAND2_2 U257 ( .A1(n254), .A2(n250), .ZN(n242) );
  VHSR_IN_2 U258 ( .I(b[7]), .ZN(n287) );
  VHSR_IN_2 U259 ( .I(a[3]), .ZN(n319) );
  VHSR_IN_2 U260 ( .I(b[6]), .ZN(n288) );
  VHSR_IN_2 U261 ( .I(a[2]), .ZN(n327) );
  VHSR_OAI22_2 U262 ( .A1(n288), .A2(n319), .B1(n287), .B2(n327), .ZN(n249) );
  VHSR_AOI22_2 U263 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n240) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[4]), .A2(a[2]), .ZN(n269) );
  VHSR_NAND3_2 U265 ( .A1(a[3]), .A2(b[5]), .A3(n269), .ZN(n239) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[7]), .A2(a[2]), .ZN(n234) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[1]), .ZN(n236) );
  VHSR_OAI22_2 U268 ( .A1(n240), .A2(n239), .B1(n234), .B2(n236), .ZN(n241) );
  VHSR_IN_2 U269 ( .I(b[4]), .ZN(n348) );
  VHSR_IN_2 U270 ( .I(a[0]), .ZN(n410) );
  VHSR_OAI211_2 U271 ( .A1(n348), .A2(n410), .B(b[5]), .C(a[1]), .ZN(n268) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[0]), .ZN(n267) );
  VHSR_MAOI222_2 U273 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_NAND4_2 U274 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n246) );
  VHSR_IN_2 U275 ( .I(b[5]), .ZN(n284) );
  VHSR_OAI22_2 U276 ( .A1(n348), .A2(n319), .B1(n284), .B2(n327), .ZN(n235) );
  VHSR_AND2_2 U277 ( .A1(n246), .A2(n235), .Z(n238) );
  VHSR_OAI21_2 U278 ( .A1(n287), .A2(n410), .B(n236), .ZN(n237) );
  VHSR_IN_2 U279 ( .I(a[1]), .ZN(n408) );
  VHSR_NOR4_2 U280 ( .A1(n348), .A2(n284), .A3(n408), .A4(n410), .ZN(n276) );
  VHSR_AND2_2 U281 ( .A1(n266), .A2(n265), .Z(n264) );
  VHSR_AD1_1 U282 ( .A(n238), .B(n237), .CI(n276), .CO(n255), .S(n265) );
  VHSR_AOI21_2 U283 ( .A1(n240), .A2(n239), .B(n241), .ZN(n258) );
  VHSR_OAI32_2 U284 ( .A1(n241), .A2(n264), .A3(n255), .B1(n258), .B2(n241), 
        .ZN(n247) );
  VHSR_CLKNAND2_2 U285 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_CLKNAND2_2 U286 ( .A1(n249), .A2(n245), .ZN(n244) );
  VHSR_NOR3_2 U287 ( .A1(n287), .A2(n319), .A3(n244), .ZN(n302) );
  VHSR_AOI21_2 U288 ( .A1(n243), .A2(n242), .B(n303), .ZN(n306) );
  VHSR_OAI32_2 U289 ( .A1(n302), .A2(n319), .A3(n287), .B1(n244), .B2(n302), 
        .ZN(n305) );
  VHSR_OAI21_2 U290 ( .A1(n247), .A2(n246), .B(n245), .ZN(n248) );
  VHSR_XNOR2_2 U291 ( .A1(n249), .A2(n248), .ZN(n313) );
  VHSR_OAI21_2 U292 ( .A1(n252), .A2(n251), .B(n250), .ZN(n253) );
  VHSR_XNOR2_2 U293 ( .A1(n254), .A2(n253), .ZN(n312) );
  VHSR_NOR2_1 U294 ( .A1(n264), .A2(n255), .ZN(n257) );
  VHSR_AOI22_2 U295 ( .A1(n264), .A2(n255), .B1(n258), .B2(n257), .ZN(n256) );
  VHSR_OAI21_2 U296 ( .A1(n258), .A2(n257), .B(n256), .ZN(n318) );
  VHSR_AOI21_2 U297 ( .A1(n261), .A2(n260), .B(n259), .ZN(n317) );
  VHSR_AOI21_2 U298 ( .A1(n270), .A2(n263), .B(n262), .ZN(n338) );
  VHSR_IAO21_2 U299 ( .A1(n266), .A2(n265), .B(n264), .ZN(n337) );
  VHSR_AOI31_2 U300 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n345) );
  VHSR_OAI31_2 U301 ( .A1(n273), .A2(n272), .A3(n271), .B(n270), .ZN(n274) );
  VHSR_IN_2 U302 ( .I(n274), .ZN(n344) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[5]), .A2(a[0]), .ZN(n275) );
  VHSR_OAI32_2 U304 ( .A1(n276), .A2(n408), .A3(n348), .B1(n275), .B2(n276), 
        .ZN(n353) );
  VHSR_NOR2_1 U305 ( .A1(n409), .A2(n410), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U306 ( .A1(n382), .A2(product[0]), .ZN(n350) );
  VHSR_IN_2 U307 ( .I(n350), .ZN(n352) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[1]), .ZN(n277) );
  VHSR_OAI32_2 U309 ( .A1(n278), .A2(n409), .A3(n290), .B1(n277), .B2(n278), 
        .ZN(n351) );
  VHSR_NOR2_1 U310 ( .A1(n279), .A2(n288), .ZN(n400) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[4]), .A2(b[6]), .ZN(n309) );
  VHSR_NAND3_2 U312 ( .A1(b[7]), .A2(a[5]), .A3(n309), .ZN(n281) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[6]), .A2(b[4]), .ZN(n310) );
  VHSR_NAND3_2 U314 ( .A1(a[7]), .A2(b[5]), .A3(n310), .ZN(n280) );
  VHSR_CLKNAND2_2 U315 ( .A1(n281), .A2(n280), .ZN(n283) );
  VHSR_IN_2 U316 ( .I(n400), .ZN(n373) );
  VHSR_MAOI222_2 U317 ( .A(n373), .B(n281), .C(n280), .ZN(n357) );
  VHSR_IN_2 U318 ( .I(n357), .ZN(n282) );
  VHSR_OAI21_2 U319 ( .A1(n400), .A2(n283), .B(n282), .ZN(n298) );
  VHSR_IN_2 U320 ( .I(n382), .ZN(n292) );
  VHSR_NOR3_2 U321 ( .A1(n290), .A2(n284), .A3(n292), .ZN(n314) );
  VHSR_NOR3_2 U322 ( .A1(n285), .A2(n310), .A3(n284), .ZN(n365) );
  VHSR_AOI22_2 U323 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n286) );
  VHSR_NOR2_1 U324 ( .A1(n365), .A2(n286), .ZN(n294) );
  VHSR_NOR4_2 U325 ( .A1(n290), .A2(n289), .A3(n288), .A4(n287), .ZN(n363) );
  VHSR_AOI22_2 U326 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n291) );
  VHSR_NOR2_1 U327 ( .A1(n363), .A2(n291), .ZN(n293) );
  VHSR_NAND3_2 U328 ( .A1(b[5]), .A2(a[5]), .A3(n292), .ZN(n308) );
  VHSR_MAOI222_2 U329 ( .A(n310), .B(n309), .C(n308), .ZN(n307) );
  VHSR_AND2_2 U330 ( .A1(n300), .A2(n307), .Z(n299) );
  VHSR_AD1_1 U331 ( .A(n314), .B(n294), .CI(n293), .CO(n295), .S(n300) );
  VHSR_NOR2_1 U332 ( .A1(n299), .A2(n295), .ZN(n297) );
  VHSR_CLKNAND2_2 U333 ( .A1(n299), .A2(n295), .ZN(n296) );
  VHSR_NOR2_1 U334 ( .A1(n297), .A2(n298), .ZN(n358) );
  VHSR_AOI22_2 U335 ( .A1(n298), .A2(n297), .B1(n296), .B2(n358), .ZN(n398) );
  VHSR_IAO21_2 U336 ( .A1(n300), .A2(n307), .B(n299), .ZN(n396) );
  VHSR_AD1_1 U337 ( .A(n303), .B(n302), .CI(n301), .CO(n399), .S(n395) );
  VHSR_AD1_1 U338 ( .A(n306), .B(n305), .CI(n304), .CO(n301), .S(n377) );
  VHSR_AOI31_2 U339 ( .A1(n310), .A2(n309), .A3(n308), .B(n307), .ZN(n376) );
  VHSR_AD1_1 U340 ( .A(n313), .B(n312), .CI(n311), .CO(n304), .S(n380) );
  VHSR_AOI22_2 U341 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n315) );
  VHSR_NOR2_1 U342 ( .A1(n315), .A2(n314), .ZN(n379) );
  VHSR_AD1_1 U343 ( .A(n318), .B(n317), .CI(n316), .CO(n311), .S(n383) );
  VHSR_NOR2_1 U344 ( .A1(n320), .A2(n327), .ZN(n342) );
  VHSR_NOR2_1 U345 ( .A1(n320), .A2(n319), .ZN(n322) );
  VHSR_OAI21_2 U346 ( .A1(n335), .A2(n327), .B(n322), .ZN(n321) );
  VHSR_OAI31_2 U347 ( .A1(n335), .A2(n322), .A3(n327), .B(n321), .ZN(n356) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[2]), .A2(a[0]), .ZN(n413) );
  VHSR_NOR3_2 U349 ( .A1(n335), .A2(n408), .A3(n413), .ZN(n324) );
  VHSR_AOI22_2 U350 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n323) );
  VHSR_NOR2_1 U351 ( .A1(n324), .A2(n323), .ZN(n387) );
  VHSR_NAND4_2 U352 ( .A1(b[1]), .A2(b[0]), .A3(a[2]), .A4(a[3]), .ZN(n334) );
  VHSR_IN_2 U353 ( .I(n334), .ZN(n326) );
  VHSR_CLKNAND2_2 U354 ( .A1(b[0]), .A2(a[3]), .ZN(n325) );
  VHSR_OAI32_2 U355 ( .A1(n326), .A2(n327), .A3(n411), .B1(n325), .B2(n326), 
        .ZN(n386) );
  VHSR_AOI22_2 U356 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n414) );
  VHSR_NOR2_1 U357 ( .A1(n414), .A2(n413), .ZN(n412) );
  VHSR_OAI211_2 U358 ( .A1(n327), .A2(n409), .B(b[1]), .C(a[3]), .ZN(n328) );
  VHSR_NAND3_2 U359 ( .A1(a[1]), .A2(b[3]), .A3(n413), .ZN(n330) );
  VHSR_IN_2 U360 ( .I(n342), .ZN(n329) );
  VHSR_MAOI222_2 U361 ( .A(n330), .B(n329), .C(n328), .ZN(n332) );
  VHSR_AOI21_2 U362 ( .A1(n331), .A2(n330), .B(n332), .ZN(n346) );
  VHSR_AOI21_2 U363 ( .A1(n347), .A2(n346), .B(n332), .ZN(n333) );
  VHSR_IN_2 U364 ( .I(n333), .ZN(n355) );
  VHSR_OAI31_2 U365 ( .A1(n408), .A2(n335), .A3(n413), .B(n334), .ZN(n354) );
  VHSR_CLKNAND2_2 U366 ( .A1(b[3]), .A2(a[3]), .ZN(n340) );
  VHSR_IAO21_2 U367 ( .A1(n342), .A2(n341), .B(n340), .ZN(n393) );
  VHSR_AD1_1 U368 ( .A(n338), .B(n337), .CI(n336), .CO(n316), .S(n392) );
  VHSR_OAI21_2 U369 ( .A1(n342), .A2(n340), .B(n341), .ZN(n339) );
  VHSR_OAI31_2 U370 ( .A1(n342), .A2(n341), .A3(n340), .B(n339), .ZN(n390) );
  VHSR_AD1_1 U371 ( .A(n345), .B(n344), .CI(n343), .CO(n336), .S(n389) );
  VHSR_XNOR2_2 U372 ( .A1(n347), .A2(n346), .ZN(n407) );
  VHSR_NOR2_1 U373 ( .A1(n348), .A2(n410), .ZN(n349) );
  VHSR_AOI32_2 U374 ( .A1(b[0]), .A2(n350), .A3(a[4]), .B1(n349), .B2(n350), 
        .ZN(n406) );
  VHSR_NOR2_1 U375 ( .A1(n407), .A2(n406), .ZN(n405) );
  VHSR_AD1_1 U376 ( .A(n353), .B(n352), .CI(n351), .CO(n343), .S(n385) );
  VHSR_AD1_1 U377 ( .A(n356), .B(n355), .CI(n354), .CO(n341), .S(n384) );
  VHSR_NOR2_1 U378 ( .A1(n358), .A2(n357), .ZN(n370) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[6]), .A2(b[7]), .ZN(n360) );
  VHSR_AOI21_2 U380 ( .A1(a[7]), .A2(b[6]), .B(n360), .ZN(n359) );
  VHSR_AOI31_2 U381 ( .A1(a[7]), .A2(n360), .A3(b[6]), .B(n359), .ZN(n361) );
  VHSR_IN_2 U382 ( .I(n361), .ZN(n362) );
  VHSR_OR2_2 U383 ( .A1(n363), .A2(n362), .Z(n364) );
  VHSR_MAOI222_2 U384 ( .A(n365), .B(n363), .C(n362), .ZN(n372) );
  VHSR_OAI21_2 U385 ( .A1(n365), .A2(n364), .B(n372), .ZN(n369) );
  VHSR_CLKXOR2_2 U386 ( .A1(n370), .A2(n369), .Z(n366) );
  VHSR_CLKNAND2_2 U387 ( .A1(n367), .A2(n366), .ZN(n402) );
  VHSR_OAI21_2 U388 ( .A1(n367), .A2(n366), .B(n402), .ZN(n368) );
  VHSR_CLKNAND2_2 U389 ( .A1(a[7]), .A2(b[7]), .ZN(n401) );
  VHSR_NOR2_1 U390 ( .A1(n370), .A2(n369), .ZN(n371) );
  VHSR_AND3_2 U391 ( .A1(n403), .A2(n373), .A3(n402), .Z(n374) );
  VHSR_NOR2_1 U392 ( .A1(n401), .A2(n374), .ZN(product[15]) );
  VHSR_AD1_1 U393 ( .A(n390), .B(n389), .CI(n388), .CO(n391), .S(product[6])
         );
  VHSR_AD1_1 U394 ( .A(n393), .B(n392), .CI(n391), .CO(n381), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U395 ( .A(n396), .B(n395), .CI(n394), .CO(n397), .S(product[11])
         );
  VHSR_AD1_1 U396 ( .A(n399), .B(n398), .CI(n397), .CO(n367), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U397 ( .A1(n401), .A2(n400), .ZN(n404) );
  VHSR_XOR3_2 U398 ( .A1(n404), .A2(n403), .A3(n402), .Z(product[14]) );
  VHSR_AOI21_2 U399 ( .A1(n407), .A2(n406), .B(n405), .ZN(product[4]) );
  VHSR_OAI22_2 U400 ( .A1(n411), .A2(n410), .B1(n409), .B2(n408), .ZN(
        product[1]) );
  VHSR_AOI21_2 U401 ( .A1(n414), .A2(n413), .B(n412), .ZN(product[2]) );
endmodule


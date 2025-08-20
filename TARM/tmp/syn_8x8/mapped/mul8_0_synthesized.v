
module mul8_0 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n225, n226, n227, n228, n229, n230, n231, n232,
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
         n409, n410, n411, n412, n413, n414, n415, n416, n417, n418, n419,
         n420, n421, n422, n423;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U214 ( .A1(n257), .A2(n256), .ZN(n255) );
  VHSR_NOR2_1 U215 ( .A1(n259), .A2(n255), .ZN(n250) );
  VHSR_NOR2_1 U216 ( .A1(n340), .A2(n341), .ZN(n351) );
  VHSR_INOR3_2 U217 ( .A1(n250), .B1(n342), .B2(n293), .ZN(n309) );
  VHSR_NOR2_1 U218 ( .A1(n415), .A2(n414), .ZN(n413) );
  VHSR_INOR2_2 U219 ( .A1(n380), .B1(n379), .ZN(n411) );
  VHSR_IN_2 U220 ( .I(n376), .ZN(product[13]) );
  VHSR_NOR2_2 U221 ( .A1(n366), .A2(n365), .ZN(n378) );
  VHSR_INAND2_1 U222 ( .A1(n371), .B1(n369), .ZN(n372) );
  VHSR_MOAI22_1 U223 ( .A1(n293), .A2(n419), .B1(a[6]), .B2(b[2]), .ZN(n238)
         );
  VHSR_AD1_1 U224 ( .A(n394), .B(n393), .CI(n392), .CO(n389), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n388), .B(n387), .CI(n386), .CO(n383), .S(product[10])
         );
  VHSR_AD1_1 U226 ( .A(n401), .B(n400), .CI(n420), .CO(n354), .S(product[3])
         );
  VHSR_AD1_1 U227 ( .A(n413), .B(n399), .CI(n398), .CO(n402), .S(product[5])
         );
  VHSR_AD1_1 U228 ( .A(n397), .B(n396), .CI(n395), .CO(n392), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U229 ( .A(n391), .B(n390), .CI(n389), .CO(n386), .S(product[9])
         );
  VHSR_AD1_1 U230 ( .A(n385), .B(n384), .CI(n383), .CO(n405), .S(
        \intadd_0/SUM[6] ) );
  VHSR_CLKNAND2_2 U231 ( .A1(b[6]), .A2(a[2]), .ZN(n226) );
  VHSR_IN_2 U232 ( .I(n226), .ZN(n254) );
  VHSR_IN_2 U233 ( .I(b[5]), .ZN(n295) );
  VHSR_CLKNAND2_2 U234 ( .A1(b[4]), .A2(a[2]), .ZN(n280) );
  VHSR_IN_2 U235 ( .I(a[3]), .ZN(n339) );
  VHSR_NOR3_2 U236 ( .A1(n295), .A2(n280), .A3(n339), .ZN(n262) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[7]), .A2(a[3]), .ZN(n252) );
  VHSR_AOI22_2 U238 ( .A1(b[6]), .A2(a[3]), .B1(b[7]), .B2(a[2]), .ZN(n225) );
  VHSR_IAO21_2 U239 ( .A1(n252), .A2(n226), .B(n225), .ZN(n261) );
  VHSR_IN_2 U240 ( .I(b[4]), .ZN(n355) );
  VHSR_IN_2 U241 ( .I(a[0]), .ZN(n418) );
  VHSR_OAI211_2 U242 ( .A1(n355), .A2(n418), .B(b[5]), .C(a[1]), .ZN(n279) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[6]), .A2(a[0]), .ZN(n278) );
  VHSR_MAOI222_2 U244 ( .A(n280), .B(n279), .C(n278), .ZN(n277) );
  VHSR_IN_2 U245 ( .I(a[1]), .ZN(n416) );
  VHSR_NOR4_2 U246 ( .A1(n355), .A2(n295), .A3(n416), .A4(n418), .ZN(n286) );
  VHSR_AOI22_2 U247 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n227) );
  VHSR_NOR2_1 U248 ( .A1(n227), .A2(n262), .ZN(n230) );
  VHSR_IN_2 U249 ( .I(b[6]), .ZN(n297) );
  VHSR_IN_2 U250 ( .I(b[7]), .ZN(n296) );
  VHSR_OAI22_2 U251 ( .A1(n297), .A2(n416), .B1(n296), .B2(n418), .ZN(n229) );
  VHSR_CLKNAND2_2 U252 ( .A1(n277), .A2(n272), .ZN(n271) );
  VHSR_NOR2_1 U253 ( .A1(n296), .A2(n416), .ZN(n228) );
  VHSR_AOI211_2 U254 ( .A1(b[4]), .A2(a[2]), .B(n295), .C(n339), .ZN(n231) );
  VHSR_MAOI222_2 U255 ( .A(n228), .B(n254), .C(n231), .ZN(n234) );
  VHSR_AD1_1 U256 ( .A(n286), .B(n230), .CI(n229), .CO(n268), .S(n272) );
  VHSR_IN_2 U257 ( .I(n268), .ZN(n233) );
  VHSR_OR2_2 U258 ( .A1(n231), .A2(n254), .Z(n232) );
  VHSR_AOI32_2 U259 ( .A1(a[1]), .A2(n234), .A3(b[7]), .B1(n232), .B2(n234), 
        .ZN(n267) );
  VHSR_AOI32_2 U260 ( .A1(n271), .A2(n234), .A3(n233), .B1(n267), .B2(n234), 
        .ZN(n260) );
  VHSR_IAO21_2 U261 ( .A1(n254), .A2(n253), .B(n252), .ZN(n310) );
  VHSR_AOI22_2 U262 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n259) );
  VHSR_IN_2 U263 ( .I(b[3]), .ZN(n342) );
  VHSR_IN_2 U264 ( .I(b[2]), .ZN(n340) );
  VHSR_IN_2 U265 ( .I(a[5]), .ZN(n299) );
  VHSR_IN_2 U266 ( .I(a[4]), .ZN(n298) );
  VHSR_NOR4_2 U267 ( .A1(n342), .A2(n340), .A3(n299), .A4(n298), .ZN(n257) );
  VHSR_IN_2 U268 ( .I(a[7]), .ZN(n293) );
  VHSR_IN_2 U269 ( .I(b[1]), .ZN(n419) );
  VHSR_NOR2_1 U270 ( .A1(n293), .A2(n419), .ZN(n236) );
  VHSR_AND2_2 U271 ( .A1(a[6]), .A2(b[2]), .Z(n235) );
  VHSR_AOI211_2 U272 ( .A1(b[2]), .A2(a[4]), .B(n342), .C(n299), .ZN(n237) );
  VHSR_MAOI222_2 U273 ( .A(n236), .B(n235), .C(n237), .ZN(n248) );
  VHSR_OAI21_2 U274 ( .A1(n238), .A2(n237), .B(n248), .ZN(n239) );
  VHSR_IN_2 U275 ( .I(n239), .ZN(n265) );
  VHSR_IN_2 U276 ( .I(b[0]), .ZN(n417) );
  VHSR_NOR4_2 U277 ( .A1(n299), .A2(n298), .A3(n419), .A4(n417), .ZN(n288) );
  VHSR_CLKNAND2_2 U278 ( .A1(b[2]), .A2(a[5]), .ZN(n241) );
  VHSR_CLKNAND2_2 U279 ( .A1(b[3]), .A2(a[4]), .ZN(n240) );
  VHSR_AOI21_2 U280 ( .A1(n241), .A2(n240), .B(n257), .ZN(n243) );
  VHSR_AOI22_2 U281 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n245) );
  VHSR_IN_2 U282 ( .I(n245), .ZN(n242) );
  VHSR_MAOI222_2 U283 ( .A(n288), .B(n243), .C(n242), .ZN(n247) );
  VHSR_CLKNAND2_2 U284 ( .A1(b[2]), .A2(a[4]), .ZN(n284) );
  VHSR_OAI211_2 U285 ( .A1(n298), .A2(n417), .B(a[5]), .C(b[1]), .ZN(n283) );
  VHSR_CLKNAND2_2 U286 ( .A1(a[6]), .A2(b[0]), .ZN(n282) );
  VHSR_MAOI222_2 U287 ( .A(n284), .B(n283), .C(n282), .ZN(n281) );
  VHSR_NOR2_1 U288 ( .A1(n288), .A2(n243), .ZN(n246) );
  VHSR_IN_2 U289 ( .I(n247), .ZN(n244) );
  VHSR_AOI21_2 U290 ( .A1(n246), .A2(n245), .B(n244), .ZN(n275) );
  VHSR_CLKNAND2_2 U291 ( .A1(n281), .A2(n275), .ZN(n274) );
  VHSR_CLKNAND2_2 U292 ( .A1(n247), .A2(n274), .ZN(n264) );
  VHSR_CLKNAND2_2 U293 ( .A1(n265), .A2(n264), .ZN(n263) );
  VHSR_CLKNAND2_2 U294 ( .A1(n248), .A2(n263), .ZN(n256) );
  VHSR_NOR2_1 U295 ( .A1(n342), .A2(n293), .ZN(n249) );
  VHSR_IAO21_2 U296 ( .A1(n250), .A2(n249), .B(n309), .ZN(n315) );
  VHSR_OAI21_2 U297 ( .A1(n254), .A2(n252), .B(n253), .ZN(n251) );
  VHSR_OAI31_2 U298 ( .A1(n254), .A2(n253), .A3(n252), .B(n251), .ZN(n314) );
  VHSR_AOI21_2 U299 ( .A1(n257), .A2(n256), .B(n255), .ZN(n258) );
  VHSR_XNOR2_2 U300 ( .A1(n259), .A2(n258), .ZN(n322) );
  VHSR_AD1_1 U301 ( .A(n262), .B(n261), .CI(n260), .CO(n253), .S(n321) );
  VHSR_OAI21_2 U302 ( .A1(n265), .A2(n264), .B(n263), .ZN(n266) );
  VHSR_IN_2 U303 ( .I(n266), .ZN(n327) );
  VHSR_NOR2_1 U304 ( .A1(n268), .A2(n267), .ZN(n270) );
  VHSR_AOI22_2 U305 ( .A1(n268), .A2(n267), .B1(n271), .B2(n270), .ZN(n269) );
  VHSR_OAI21_2 U306 ( .A1(n271), .A2(n270), .B(n269), .ZN(n326) );
  VHSR_OAI21_2 U307 ( .A1(n277), .A2(n272), .B(n271), .ZN(n273) );
  VHSR_IN_2 U308 ( .I(n273), .ZN(n330) );
  VHSR_OAI21_2 U309 ( .A1(n281), .A2(n275), .B(n274), .ZN(n276) );
  VHSR_IN_2 U310 ( .I(n276), .ZN(n329) );
  VHSR_AOI31_2 U311 ( .A1(n280), .A2(n279), .A3(n278), .B(n277), .ZN(n347) );
  VHSR_AOI31_2 U312 ( .A1(n284), .A2(n283), .A3(n282), .B(n281), .ZN(n346) );
  VHSR_CLKNAND2_2 U313 ( .A1(b[5]), .A2(a[0]), .ZN(n285) );
  VHSR_OAI32_2 U314 ( .A1(n286), .A2(n416), .A3(n355), .B1(n285), .B2(n286), 
        .ZN(n364) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[4]), .A2(b[1]), .ZN(n287) );
  VHSR_OAI32_2 U316 ( .A1(n288), .A2(n299), .A3(n417), .B1(n287), .B2(n288), 
        .ZN(n363) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[4]), .A2(b[4]), .ZN(n301) );
  VHSR_IN_2 U318 ( .I(n301), .ZN(n393) );
  VHSR_NOR2_1 U319 ( .A1(n417), .A2(n418), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U320 ( .A1(n393), .A2(product[0]), .ZN(n357) );
  VHSR_IN_2 U321 ( .I(n357), .ZN(n362) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[6]), .A2(b[6]), .ZN(n381) );
  VHSR_IN_2 U323 ( .I(n381), .ZN(n408) );
  VHSR_CLKNAND2_2 U324 ( .A1(a[6]), .A2(b[4]), .ZN(n319) );
  VHSR_NAND3_2 U325 ( .A1(a[7]), .A2(b[5]), .A3(n319), .ZN(n290) );
  VHSR_CLKNAND2_2 U326 ( .A1(a[4]), .A2(b[6]), .ZN(n318) );
  VHSR_NAND3_2 U327 ( .A1(b[7]), .A2(a[5]), .A3(n318), .ZN(n289) );
  VHSR_CLKNAND2_2 U328 ( .A1(n290), .A2(n289), .ZN(n292) );
  VHSR_MAOI222_2 U329 ( .A(n381), .B(n290), .C(n289), .ZN(n365) );
  VHSR_IN_2 U330 ( .I(n365), .ZN(n291) );
  VHSR_OAI21_2 U331 ( .A1(n408), .A2(n292), .B(n291), .ZN(n307) );
  VHSR_NOR3_2 U332 ( .A1(n293), .A2(n319), .A3(n295), .ZN(n373) );
  VHSR_AOI22_2 U333 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n294) );
  VHSR_NOR2_1 U334 ( .A1(n373), .A2(n294), .ZN(n303) );
  VHSR_NOR3_2 U335 ( .A1(n299), .A2(n295), .A3(n301), .ZN(n324) );
  VHSR_NOR4_2 U336 ( .A1(n299), .A2(n298), .A3(n297), .A4(n296), .ZN(n371) );
  VHSR_AOI22_2 U337 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n300) );
  VHSR_NOR2_1 U338 ( .A1(n371), .A2(n300), .ZN(n302) );
  VHSR_NAND3_2 U339 ( .A1(b[5]), .A2(a[5]), .A3(n301), .ZN(n317) );
  VHSR_MAOI222_2 U340 ( .A(n319), .B(n318), .C(n317), .ZN(n316) );
  VHSR_AND2_2 U341 ( .A1(n312), .A2(n316), .Z(n311) );
  VHSR_AD1_1 U342 ( .A(n303), .B(n324), .CI(n302), .CO(n304), .S(n312) );
  VHSR_NOR2_1 U343 ( .A1(n311), .A2(n304), .ZN(n306) );
  VHSR_CLKNAND2_2 U344 ( .A1(n311), .A2(n304), .ZN(n305) );
  VHSR_NOR2_1 U345 ( .A1(n306), .A2(n307), .ZN(n366) );
  VHSR_AOI22_2 U346 ( .A1(n307), .A2(n306), .B1(n305), .B2(n366), .ZN(n406) );
  VHSR_AD1_1 U347 ( .A(n310), .B(n309), .CI(n308), .CO(n407), .S(n385) );
  VHSR_IAO21_2 U348 ( .A1(n312), .A2(n316), .B(n311), .ZN(n384) );
  VHSR_AD1_1 U349 ( .A(n315), .B(n314), .CI(n313), .CO(n308), .S(n388) );
  VHSR_AOI31_2 U350 ( .A1(n319), .A2(n318), .A3(n317), .B(n316), .ZN(n387) );
  VHSR_AD1_1 U351 ( .A(n322), .B(n321), .CI(n320), .CO(n313), .S(n391) );
  VHSR_AOI22_2 U352 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n323) );
  VHSR_NOR2_1 U353 ( .A1(n324), .A2(n323), .ZN(n390) );
  VHSR_AD1_1 U354 ( .A(n327), .B(n326), .CI(n325), .CO(n320), .S(n394) );
  VHSR_AD1_1 U355 ( .A(n330), .B(n329), .CI(n328), .CO(n325), .S(n397) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[2]), .A2(a[0]), .ZN(n422) );
  VHSR_NOR3_2 U357 ( .A1(n342), .A2(n416), .A3(n422), .ZN(n337) );
  VHSR_AOI22_2 U358 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n331) );
  VHSR_NOR2_1 U359 ( .A1(n337), .A2(n331), .ZN(n401) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[0]), .A2(a[2]), .ZN(n423) );
  VHSR_IN_2 U361 ( .I(n423), .ZN(n338) );
  VHSR_AOI22_2 U362 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n332) );
  VHSR_AOI31_2 U363 ( .A1(a[3]), .A2(b[1]), .A3(n338), .B(n332), .ZN(n400) );
  VHSR_CLKNAND2_2 U364 ( .A1(b[1]), .A2(a[1]), .ZN(n421) );
  VHSR_MAOI222_2 U365 ( .A(n423), .B(n422), .C(n421), .ZN(n420) );
  VHSR_NAND3_2 U366 ( .A1(n423), .A2(a[3]), .A3(b[1]), .ZN(n335) );
  VHSR_NAND3_2 U367 ( .A1(a[1]), .A2(b[3]), .A3(n422), .ZN(n333) );
  VHSR_IN_2 U368 ( .I(a[2]), .ZN(n341) );
  VHSR_IN_2 U369 ( .I(n351), .ZN(n344) );
  VHSR_AND2_2 U370 ( .A1(n333), .A2(n344), .Z(n334) );
  VHSR_MAOI222_2 U371 ( .A(n335), .B(n344), .C(n333), .ZN(n336) );
  VHSR_AOI21_2 U372 ( .A1(n335), .A2(n334), .B(n336), .ZN(n353) );
  VHSR_AOI21_2 U373 ( .A1(n354), .A2(n353), .B(n336), .ZN(n360) );
  VHSR_AOI31_2 U374 ( .A1(a[3]), .A2(b[1]), .A3(n338), .B(n337), .ZN(n359) );
  VHSR_CLKNAND2_2 U375 ( .A1(b[3]), .A2(a[3]), .ZN(n352) );
  VHSR_OAI22_2 U376 ( .A1(n342), .A2(n341), .B1(n340), .B2(n339), .ZN(n343) );
  VHSR_OAI21_2 U377 ( .A1(n352), .A2(n344), .B(n343), .ZN(n358) );
  VHSR_AOI21_2 U378 ( .A1(n348), .A2(n344), .B(n352), .ZN(n396) );
  VHSR_AD1_1 U379 ( .A(n347), .B(n346), .CI(n345), .CO(n328), .S(n404) );
  VHSR_IN_2 U380 ( .I(n348), .ZN(n350) );
  VHSR_OAI21_2 U381 ( .A1(n352), .A2(n351), .B(n350), .ZN(n349) );
  VHSR_OAI31_2 U382 ( .A1(n352), .A2(n351), .A3(n350), .B(n349), .ZN(n403) );
  VHSR_XNOR2_2 U383 ( .A1(n354), .A2(n353), .ZN(n415) );
  VHSR_NOR2_1 U384 ( .A1(n355), .A2(n418), .ZN(n356) );
  VHSR_AOI32_2 U385 ( .A1(b[0]), .A2(n357), .A3(a[4]), .B1(n356), .B2(n357), 
        .ZN(n414) );
  VHSR_AD1_1 U386 ( .A(n360), .B(n359), .CI(n358), .CO(n348), .S(n361) );
  VHSR_IN_2 U387 ( .I(n361), .ZN(n399) );
  VHSR_AD1_1 U388 ( .A(n364), .B(n363), .CI(n362), .CO(n345), .S(n398) );
  VHSR_CLKNAND2_2 U389 ( .A1(a[7]), .A2(b[6]), .ZN(n368) );
  VHSR_AOI21_2 U390 ( .A1(a[6]), .A2(b[7]), .B(n368), .ZN(n367) );
  VHSR_AOI31_2 U391 ( .A1(a[6]), .A2(n368), .A3(b[7]), .B(n367), .ZN(n369) );
  VHSR_IN_2 U392 ( .I(n369), .ZN(n370) );
  VHSR_MAOI222_2 U393 ( .A(n373), .B(n371), .C(n370), .ZN(n380) );
  VHSR_OAI21_2 U394 ( .A1(n373), .A2(n372), .B(n380), .ZN(n377) );
  VHSR_CLKXOR2_2 U395 ( .A1(n378), .A2(n377), .Z(n374) );
  VHSR_CLKNAND2_2 U396 ( .A1(n375), .A2(n374), .ZN(n410) );
  VHSR_OAI21_2 U397 ( .A1(n375), .A2(n374), .B(n410), .ZN(n376) );
  VHSR_CLKNAND2_2 U398 ( .A1(a[7]), .A2(b[7]), .ZN(n409) );
  VHSR_NOR2_1 U399 ( .A1(n378), .A2(n377), .ZN(n379) );
  VHSR_AND3_2 U400 ( .A1(n411), .A2(n381), .A3(n410), .Z(n382) );
  VHSR_NOR2_1 U401 ( .A1(n409), .A2(n382), .ZN(product[15]) );
  VHSR_AD1_1 U402 ( .A(n404), .B(n403), .CI(n402), .CO(n395), .S(product[6])
         );
  VHSR_AD1_1 U403 ( .A(n407), .B(n406), .CI(n405), .CO(n375), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U404 ( .A1(n409), .A2(n408), .ZN(n412) );
  VHSR_XOR3_2 U405 ( .A1(n412), .A2(n411), .A3(n410), .Z(product[14]) );
  VHSR_AOI21_2 U406 ( .A1(n415), .A2(n414), .B(n413), .ZN(product[4]) );
  VHSR_OAI22_2 U407 ( .A1(n419), .A2(n418), .B1(n417), .B2(n416), .ZN(
        product[1]) );
  VHSR_AOI31_2 U408 ( .A1(n423), .A2(n422), .A3(n421), .B(n420), .ZN(
        product[2]) );
endmodule


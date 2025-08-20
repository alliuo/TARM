
module mul8_144 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n219, n220,
         n221, n222, n223, n224, n225, n226, n227, n228, n229, n230, n231,
         n232, n233, n234, n235, n236, n237, n238, n239, n240, n241, n242,
         n243, n244, n245, n246, n247, n248, n249, n250, n251, n252, n253,
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
         n408, n409, n410, n411, n412, n413, n414;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U209 ( .A1(n240), .B1(n259), .ZN(n246) );
  VHSR_NOR2_1 U210 ( .A1(n253), .A2(n249), .ZN(n242) );
  VHSR_NOR2_1 U211 ( .A1(n331), .A2(n332), .ZN(n342) );
  VHSR_NOR2_1 U212 ( .A1(n297), .A2(n298), .ZN(n357) );
  VHSR_NOR2_1 U213 ( .A1(n406), .A2(n405), .ZN(n404) );
  VHSR_INOR2_2 U214 ( .A1(n371), .B1(n370), .ZN(n402) );
  VHSR_IN_2 U215 ( .I(n367), .ZN(product[13]) );
  VHSR_INOR3_1 U216 ( .A1(n242), .B1(n333), .B2(n285), .ZN(n303) );
  VHSR_NOR2_2 U217 ( .A1(n251), .A2(n250), .ZN(n249) );
  VHSR_NOR2_2 U218 ( .A1(n357), .A2(n356), .ZN(n369) );
  VHSR_NOR2_2 U219 ( .A1(n299), .A2(n295), .ZN(n297) );
  VHSR_INAND2_1 U220 ( .A1(n362), .B1(n360), .ZN(n363) );
  VHSR_MOAI22_1 U221 ( .A1(n285), .A2(n410), .B1(a[6]), .B2(b[2]), .ZN(n222)
         );
  VHSR_AD1_1 U222 ( .A(n379), .B(n378), .CI(n377), .CO(n374), .S(product[9])
         );
  VHSR_AD1_1 U223 ( .A(n389), .B(n388), .CI(n411), .CO(n345), .S(product[3])
         );
  VHSR_AD1_1 U224 ( .A(n404), .B(n387), .CI(n386), .CO(n390), .S(product[5])
         );
  VHSR_AD1_1 U225 ( .A(n385), .B(n384), .CI(n383), .CO(n380), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U226 ( .A(n382), .B(n381), .CI(n380), .CO(n377), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U227 ( .A(n376), .B(n375), .CI(n374), .CO(n393), .S(product[10])
         );
  VHSR_AOI22_2 U228 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n253) );
  VHSR_IN_2 U229 ( .I(b[3]), .ZN(n333) );
  VHSR_IN_2 U230 ( .I(b[2]), .ZN(n331) );
  VHSR_IN_2 U231 ( .I(a[5]), .ZN(n290) );
  VHSR_IN_2 U232 ( .I(a[4]), .ZN(n289) );
  VHSR_NOR4_2 U233 ( .A1(n333), .A2(n331), .A3(n290), .A4(n289), .ZN(n251) );
  VHSR_IN_2 U234 ( .I(a[7]), .ZN(n285) );
  VHSR_IN_2 U235 ( .I(b[1]), .ZN(n410) );
  VHSR_NOR2_1 U236 ( .A1(n285), .A2(n410), .ZN(n220) );
  VHSR_AND2_2 U237 ( .A1(a[6]), .A2(b[2]), .Z(n219) );
  VHSR_AOI211_2 U238 ( .A1(b[2]), .A2(a[4]), .B(n333), .C(n290), .ZN(n221) );
  VHSR_MAOI222_2 U239 ( .A(n220), .B(n219), .C(n221), .ZN(n232) );
  VHSR_OAI21_2 U240 ( .A1(n222), .A2(n221), .B(n232), .ZN(n223) );
  VHSR_IN_2 U241 ( .I(n223), .ZN(n256) );
  VHSR_IN_2 U242 ( .I(b[0]), .ZN(n408) );
  VHSR_NOR4_2 U243 ( .A1(n290), .A2(n289), .A3(n410), .A4(n408), .ZN(n279) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[2]), .A2(a[5]), .ZN(n225) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[3]), .A2(a[4]), .ZN(n224) );
  VHSR_AOI21_2 U246 ( .A1(n225), .A2(n224), .B(n251), .ZN(n227) );
  VHSR_AOI22_2 U247 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n229) );
  VHSR_IN_2 U248 ( .I(n229), .ZN(n226) );
  VHSR_MAOI222_2 U249 ( .A(n279), .B(n227), .C(n226), .ZN(n231) );
  VHSR_CLKNAND2_2 U250 ( .A1(b[2]), .A2(a[4]), .ZN(n275) );
  VHSR_OAI211_2 U251 ( .A1(n289), .A2(n408), .B(a[5]), .C(b[1]), .ZN(n274) );
  VHSR_CLKNAND2_2 U252 ( .A1(a[6]), .A2(b[0]), .ZN(n273) );
  VHSR_MAOI222_2 U253 ( .A(n275), .B(n274), .C(n273), .ZN(n272) );
  VHSR_NOR2_1 U254 ( .A1(n279), .A2(n227), .ZN(n230) );
  VHSR_IN_2 U255 ( .I(n231), .ZN(n228) );
  VHSR_AOI21_2 U256 ( .A1(n230), .A2(n229), .B(n228), .ZN(n266) );
  VHSR_CLKNAND2_2 U257 ( .A1(n272), .A2(n266), .ZN(n265) );
  VHSR_CLKNAND2_2 U258 ( .A1(n231), .A2(n265), .ZN(n255) );
  VHSR_CLKNAND2_2 U259 ( .A1(n256), .A2(n255), .ZN(n254) );
  VHSR_CLKNAND2_2 U260 ( .A1(n232), .A2(n254), .ZN(n250) );
  VHSR_IN_2 U261 ( .I(b[7]), .ZN(n287) );
  VHSR_IN_2 U262 ( .I(a[3]), .ZN(n330) );
  VHSR_IN_2 U263 ( .I(b[6]), .ZN(n288) );
  VHSR_IN_2 U264 ( .I(a[2]), .ZN(n332) );
  VHSR_OAI22_2 U265 ( .A1(n288), .A2(n330), .B1(n287), .B2(n332), .ZN(n248) );
  VHSR_NOR2_1 U266 ( .A1(n287), .A2(n332), .ZN(n234) );
  VHSR_IN_2 U267 ( .I(a[1]), .ZN(n407) );
  VHSR_NOR2_1 U268 ( .A1(n288), .A2(n407), .ZN(n233) );
  VHSR_IN_2 U269 ( .I(b[5]), .ZN(n284) );
  VHSR_AOI211_2 U270 ( .A1(b[4]), .A2(a[2]), .B(n284), .C(n330), .ZN(n239) );
  VHSR_OAI22_2 U271 ( .A1(n288), .A2(n332), .B1(n287), .B2(n407), .ZN(n238) );
  VHSR_AOI22_2 U272 ( .A1(n234), .A2(n233), .B1(n239), .B2(n238), .ZN(n240) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[4]), .A2(a[2]), .ZN(n271) );
  VHSR_IN_2 U274 ( .I(b[4]), .ZN(n346) );
  VHSR_IN_2 U275 ( .I(a[0]), .ZN(n409) );
  VHSR_OAI211_2 U276 ( .A1(n346), .A2(n409), .B(b[5]), .C(a[1]), .ZN(n270) );
  VHSR_CLKNAND2_2 U277 ( .A1(b[6]), .A2(a[0]), .ZN(n269) );
  VHSR_MAOI222_2 U278 ( .A(n271), .B(n270), .C(n269), .ZN(n268) );
  VHSR_NOR4_2 U279 ( .A1(n346), .A2(n284), .A3(n407), .A4(n409), .ZN(n277) );
  VHSR_NAND4_2 U280 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n245) );
  VHSR_OAI22_2 U281 ( .A1(n346), .A2(n330), .B1(n284), .B2(n332), .ZN(n235) );
  VHSR_AND2_2 U282 ( .A1(n245), .A2(n235), .Z(n237) );
  VHSR_OAI22_2 U283 ( .A1(n288), .A2(n407), .B1(n287), .B2(n409), .ZN(n236) );
  VHSR_AND2_2 U284 ( .A1(n268), .A2(n264), .Z(n263) );
  VHSR_AD1_1 U285 ( .A(n277), .B(n237), .CI(n236), .CO(n258), .S(n264) );
  VHSR_NOR2_1 U286 ( .A1(n263), .A2(n258), .ZN(n261) );
  VHSR_OAI21_2 U287 ( .A1(n239), .A2(n238), .B(n240), .ZN(n262) );
  VHSR_NOR2_1 U288 ( .A1(n261), .A2(n262), .ZN(n259) );
  VHSR_CLKNAND2_2 U289 ( .A1(n246), .A2(n245), .ZN(n244) );
  VHSR_CLKNAND2_2 U290 ( .A1(n248), .A2(n244), .ZN(n243) );
  VHSR_NOR3_2 U291 ( .A1(n287), .A2(n330), .A3(n243), .ZN(n302) );
  VHSR_NOR2_1 U292 ( .A1(n333), .A2(n285), .ZN(n241) );
  VHSR_IAO21_2 U293 ( .A1(n242), .A2(n241), .B(n303), .ZN(n306) );
  VHSR_OAI32_2 U294 ( .A1(n302), .A2(n330), .A3(n287), .B1(n243), .B2(n302), 
        .ZN(n305) );
  VHSR_OAI21_2 U295 ( .A1(n246), .A2(n245), .B(n244), .ZN(n247) );
  VHSR_XNOR2_2 U296 ( .A1(n248), .A2(n247), .ZN(n313) );
  VHSR_AOI21_2 U297 ( .A1(n251), .A2(n250), .B(n249), .ZN(n252) );
  VHSR_XNOR2_2 U298 ( .A1(n253), .A2(n252), .ZN(n312) );
  VHSR_OAI21_2 U299 ( .A1(n256), .A2(n255), .B(n254), .ZN(n257) );
  VHSR_IN_2 U300 ( .I(n257), .ZN(n318) );
  VHSR_CLKNAND2_2 U301 ( .A1(n263), .A2(n258), .ZN(n260) );
  VHSR_AOI22_2 U302 ( .A1(n262), .A2(n261), .B1(n260), .B2(n259), .ZN(n317) );
  VHSR_IAO21_2 U303 ( .A1(n268), .A2(n264), .B(n263), .ZN(n321) );
  VHSR_OAI21_2 U304 ( .A1(n272), .A2(n266), .B(n265), .ZN(n267) );
  VHSR_IN_2 U305 ( .I(n267), .ZN(n320) );
  VHSR_AOI31_2 U306 ( .A1(n271), .A2(n270), .A3(n269), .B(n268), .ZN(n338) );
  VHSR_AOI31_2 U307 ( .A1(n275), .A2(n274), .A3(n273), .B(n272), .ZN(n337) );
  VHSR_CLKNAND2_2 U308 ( .A1(b[5]), .A2(a[0]), .ZN(n276) );
  VHSR_OAI32_2 U309 ( .A1(n277), .A2(n407), .A3(n346), .B1(n276), .B2(n277), 
        .ZN(n355) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[4]), .A2(b[1]), .ZN(n278) );
  VHSR_OAI32_2 U311 ( .A1(n279), .A2(n290), .A3(n408), .B1(n278), .B2(n279), 
        .ZN(n354) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[4]), .A2(b[4]), .ZN(n292) );
  VHSR_IN_2 U313 ( .I(n292), .ZN(n381) );
  VHSR_NOR2_1 U314 ( .A1(n408), .A2(n409), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U315 ( .A1(n381), .A2(product[0]), .ZN(n348) );
  VHSR_IN_2 U316 ( .I(n348), .ZN(n353) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[6]), .A2(b[6]), .ZN(n372) );
  VHSR_IN_2 U318 ( .I(n372), .ZN(n399) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[6]), .A2(b[4]), .ZN(n310) );
  VHSR_NAND3_2 U320 ( .A1(a[7]), .A2(b[5]), .A3(n310), .ZN(n281) );
  VHSR_CLKNAND2_2 U321 ( .A1(a[4]), .A2(b[6]), .ZN(n309) );
  VHSR_NAND3_2 U322 ( .A1(b[7]), .A2(a[5]), .A3(n309), .ZN(n280) );
  VHSR_CLKNAND2_2 U323 ( .A1(n281), .A2(n280), .ZN(n283) );
  VHSR_MAOI222_2 U324 ( .A(n372), .B(n281), .C(n280), .ZN(n356) );
  VHSR_IN_2 U325 ( .I(n356), .ZN(n282) );
  VHSR_OAI21_2 U326 ( .A1(n399), .A2(n283), .B(n282), .ZN(n298) );
  VHSR_NOR3_2 U327 ( .A1(n290), .A2(n284), .A3(n292), .ZN(n314) );
  VHSR_NOR3_2 U328 ( .A1(n285), .A2(n310), .A3(n284), .ZN(n364) );
  VHSR_AOI22_2 U329 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n286) );
  VHSR_NOR2_1 U330 ( .A1(n364), .A2(n286), .ZN(n294) );
  VHSR_NOR4_2 U331 ( .A1(n290), .A2(n289), .A3(n288), .A4(n287), .ZN(n362) );
  VHSR_AOI22_2 U332 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n291) );
  VHSR_NOR2_1 U333 ( .A1(n362), .A2(n291), .ZN(n293) );
  VHSR_NAND3_2 U334 ( .A1(b[5]), .A2(a[5]), .A3(n292), .ZN(n308) );
  VHSR_MAOI222_2 U335 ( .A(n310), .B(n309), .C(n308), .ZN(n307) );
  VHSR_AND2_2 U336 ( .A1(n300), .A2(n307), .Z(n299) );
  VHSR_AD1_1 U337 ( .A(n314), .B(n294), .CI(n293), .CO(n295), .S(n300) );
  VHSR_CLKNAND2_2 U338 ( .A1(n299), .A2(n295), .ZN(n296) );
  VHSR_AOI22_2 U339 ( .A1(n298), .A2(n297), .B1(n296), .B2(n357), .ZN(n397) );
  VHSR_IAO21_2 U340 ( .A1(n300), .A2(n307), .B(n299), .ZN(n395) );
  VHSR_AD1_1 U341 ( .A(n303), .B(n302), .CI(n301), .CO(n398), .S(n394) );
  VHSR_AD1_1 U342 ( .A(n306), .B(n305), .CI(n304), .CO(n301), .S(n376) );
  VHSR_AOI31_2 U343 ( .A1(n310), .A2(n309), .A3(n308), .B(n307), .ZN(n375) );
  VHSR_AD1_1 U344 ( .A(n313), .B(n312), .CI(n311), .CO(n304), .S(n379) );
  VHSR_AOI22_2 U345 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n315) );
  VHSR_NOR2_1 U346 ( .A1(n315), .A2(n314), .ZN(n378) );
  VHSR_AD1_1 U347 ( .A(n318), .B(n317), .CI(n316), .CO(n311), .S(n382) );
  VHSR_AD1_1 U348 ( .A(n321), .B(n320), .CI(n319), .CO(n316), .S(n385) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[2]), .A2(a[0]), .ZN(n413) );
  VHSR_NOR3_2 U350 ( .A1(n333), .A2(n407), .A3(n413), .ZN(n328) );
  VHSR_AOI22_2 U351 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n322) );
  VHSR_NOR2_1 U352 ( .A1(n328), .A2(n322), .ZN(n389) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[0]), .A2(a[2]), .ZN(n414) );
  VHSR_IN_2 U354 ( .I(n414), .ZN(n329) );
  VHSR_AOI22_2 U355 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n323) );
  VHSR_AOI31_2 U356 ( .A1(a[3]), .A2(b[1]), .A3(n329), .B(n323), .ZN(n388) );
  VHSR_CLKNAND2_2 U357 ( .A1(b[1]), .A2(a[1]), .ZN(n412) );
  VHSR_MAOI222_2 U358 ( .A(n414), .B(n413), .C(n412), .ZN(n411) );
  VHSR_NAND3_2 U359 ( .A1(n414), .A2(a[3]), .A3(b[1]), .ZN(n326) );
  VHSR_NAND3_2 U360 ( .A1(a[1]), .A2(b[3]), .A3(n413), .ZN(n324) );
  VHSR_IN_2 U361 ( .I(n342), .ZN(n335) );
  VHSR_AND2_2 U362 ( .A1(n324), .A2(n335), .Z(n325) );
  VHSR_MAOI222_2 U363 ( .A(n326), .B(n335), .C(n324), .ZN(n327) );
  VHSR_AOI21_2 U364 ( .A1(n326), .A2(n325), .B(n327), .ZN(n344) );
  VHSR_AOI21_2 U365 ( .A1(n345), .A2(n344), .B(n327), .ZN(n351) );
  VHSR_AOI31_2 U366 ( .A1(a[3]), .A2(b[1]), .A3(n329), .B(n328), .ZN(n350) );
  VHSR_CLKNAND2_2 U367 ( .A1(b[3]), .A2(a[3]), .ZN(n343) );
  VHSR_OAI22_2 U368 ( .A1(n333), .A2(n332), .B1(n331), .B2(n330), .ZN(n334) );
  VHSR_OAI21_2 U369 ( .A1(n343), .A2(n335), .B(n334), .ZN(n349) );
  VHSR_AOI21_2 U370 ( .A1(n339), .A2(n335), .B(n343), .ZN(n384) );
  VHSR_AD1_1 U371 ( .A(n338), .B(n337), .CI(n336), .CO(n319), .S(n392) );
  VHSR_IN_2 U372 ( .I(n339), .ZN(n341) );
  VHSR_OAI21_2 U373 ( .A1(n343), .A2(n342), .B(n341), .ZN(n340) );
  VHSR_OAI31_2 U374 ( .A1(n343), .A2(n342), .A3(n341), .B(n340), .ZN(n391) );
  VHSR_XNOR2_2 U375 ( .A1(n345), .A2(n344), .ZN(n406) );
  VHSR_NOR2_1 U376 ( .A1(n346), .A2(n409), .ZN(n347) );
  VHSR_AOI32_2 U377 ( .A1(b[0]), .A2(n348), .A3(a[4]), .B1(n347), .B2(n348), 
        .ZN(n405) );
  VHSR_AD1_1 U378 ( .A(n351), .B(n350), .CI(n349), .CO(n339), .S(n352) );
  VHSR_IN_2 U379 ( .I(n352), .ZN(n387) );
  VHSR_AD1_1 U380 ( .A(n355), .B(n354), .CI(n353), .CO(n336), .S(n386) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[7]), .A2(b[6]), .ZN(n359) );
  VHSR_AOI21_2 U382 ( .A1(a[6]), .A2(b[7]), .B(n359), .ZN(n358) );
  VHSR_AOI31_2 U383 ( .A1(a[6]), .A2(n359), .A3(b[7]), .B(n358), .ZN(n360) );
  VHSR_IN_2 U384 ( .I(n360), .ZN(n361) );
  VHSR_MAOI222_2 U385 ( .A(n364), .B(n362), .C(n361), .ZN(n371) );
  VHSR_OAI21_2 U386 ( .A1(n364), .A2(n363), .B(n371), .ZN(n368) );
  VHSR_CLKXOR2_2 U387 ( .A1(n369), .A2(n368), .Z(n365) );
  VHSR_CLKNAND2_2 U388 ( .A1(n366), .A2(n365), .ZN(n401) );
  VHSR_OAI21_2 U389 ( .A1(n366), .A2(n365), .B(n401), .ZN(n367) );
  VHSR_CLKNAND2_2 U390 ( .A1(a[7]), .A2(b[7]), .ZN(n400) );
  VHSR_NOR2_1 U391 ( .A1(n369), .A2(n368), .ZN(n370) );
  VHSR_AND3_2 U392 ( .A1(n402), .A2(n372), .A3(n401), .Z(n373) );
  VHSR_NOR2_1 U393 ( .A1(n400), .A2(n373), .ZN(product[15]) );
  VHSR_AD1_1 U394 ( .A(n392), .B(n391), .CI(n390), .CO(n383), .S(product[6])
         );
  VHSR_AD1_1 U395 ( .A(n395), .B(n394), .CI(n393), .CO(n396), .S(product[11])
         );
  VHSR_AD1_1 U396 ( .A(n398), .B(n397), .CI(n396), .CO(n366), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U397 ( .A1(n400), .A2(n399), .ZN(n403) );
  VHSR_XOR3_2 U398 ( .A1(n403), .A2(n402), .A3(n401), .Z(product[14]) );
  VHSR_AOI21_2 U399 ( .A1(n406), .A2(n405), .B(n404), .ZN(product[4]) );
  VHSR_OAI22_2 U400 ( .A1(n410), .A2(n409), .B1(n408), .B2(n407), .ZN(
        product[1]) );
  VHSR_AOI31_2 U401 ( .A1(n414), .A2(n413), .A3(n412), .B(n411), .ZN(
        product[2]) );
endmodule


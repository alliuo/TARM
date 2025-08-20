
module mul8_121 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n221, n222,
         n223, n224, n225, n226, n227, n228, n229, n230, n231, n232, n233,
         n234, n235, n236, n237, n238, n239, n240, n241, n242, n243, n244,
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
         n410, n411, n412, n413, n414, n415, n416, n417;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U211 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_NOR2_1 U212 ( .A1(n256), .A2(n252), .ZN(n245) );
  VHSR_INOR3_2 U213 ( .A1(n245), .B1(n333), .B2(n287), .ZN(n305) );
  VHSR_NOR2_1 U214 ( .A1(n409), .A2(n408), .ZN(n407) );
  VHSR_INOR2_2 U215 ( .A1(n374), .B1(n373), .ZN(n405) );
  VHSR_IN_2 U216 ( .I(n370), .ZN(product[13]) );
  VHSR_NOR2_2 U217 ( .A1(n360), .A2(n359), .ZN(n372) );
  VHSR_INAND2_1 U218 ( .A1(n365), .B1(n363), .ZN(n366) );
  VHSR_AD1_1 U219 ( .A(n382), .B(n381), .CI(n380), .CO(n377), .S(product[9])
         );
  VHSR_AD1_1 U220 ( .A(n392), .B(n391), .CI(n414), .CO(n348), .S(product[3])
         );
  VHSR_AD1_1 U221 ( .A(n407), .B(n390), .CI(n389), .CO(n393), .S(product[5])
         );
  VHSR_AD1_1 U222 ( .A(n388), .B(n387), .CI(n386), .CO(n383), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U223 ( .A(n385), .B(n384), .CI(n383), .CO(n380), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n379), .B(n378), .CI(n377), .CO(n396), .S(product[10])
         );
  VHSR_AOI22_2 U225 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n256) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n333) );
  VHSR_IN_2 U227 ( .I(b[2]), .ZN(n331) );
  VHSR_IN_2 U228 ( .I(a[5]), .ZN(n292) );
  VHSR_IN_2 U229 ( .I(a[4]), .ZN(n291) );
  VHSR_NOR4_2 U230 ( .A1(n333), .A2(n331), .A3(n292), .A4(n291), .ZN(n254) );
  VHSR_IN_2 U231 ( .I(a[7]), .ZN(n287) );
  VHSR_IN_2 U232 ( .I(b[1]), .ZN(n413) );
  VHSR_NOR2_1 U233 ( .A1(n287), .A2(n413), .ZN(n222) );
  VHSR_AOI211_2 U234 ( .A1(b[2]), .A2(a[4]), .B(n333), .C(n292), .ZN(n223) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[6]), .A2(b[2]), .ZN(n225) );
  VHSR_IN_2 U236 ( .I(n225), .ZN(n221) );
  VHSR_MAOI222_2 U237 ( .A(n222), .B(n223), .C(n221), .ZN(n235) );
  VHSR_AOI21_2 U238 ( .A1(b[1]), .A2(a[7]), .B(n223), .ZN(n226) );
  VHSR_IN_2 U239 ( .I(n235), .ZN(n224) );
  VHSR_AOI21_2 U240 ( .A1(n226), .A2(n225), .B(n224), .ZN(n263) );
  VHSR_CLKNAND2_2 U241 ( .A1(a[6]), .A2(b[1]), .ZN(n232) );
  VHSR_IN_2 U242 ( .I(n232), .ZN(n229) );
  VHSR_IN_2 U243 ( .I(b[0]), .ZN(n411) );
  VHSR_NOR4_2 U244 ( .A1(n292), .A2(n291), .A3(n413), .A4(n411), .ZN(n281) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[2]), .A2(a[5]), .ZN(n228) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[3]), .A2(a[4]), .ZN(n227) );
  VHSR_AOI21_2 U247 ( .A1(n228), .A2(n227), .B(n254), .ZN(n230) );
  VHSR_MAOI222_2 U248 ( .A(n229), .B(n281), .C(n230), .ZN(n234) );
  VHSR_CLKNAND2_2 U249 ( .A1(b[2]), .A2(a[4]), .ZN(n277) );
  VHSR_OAI21_2 U250 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n276) );
  VHSR_OAI211_2 U251 ( .A1(n291), .A2(n411), .B(a[5]), .C(b[1]), .ZN(n275) );
  VHSR_MAOI222_2 U252 ( .A(n277), .B(n276), .C(n275), .ZN(n274) );
  VHSR_NOR2_1 U253 ( .A1(n281), .A2(n230), .ZN(n233) );
  VHSR_IN_2 U254 ( .I(n234), .ZN(n231) );
  VHSR_AOI21_2 U255 ( .A1(n233), .A2(n232), .B(n231), .ZN(n266) );
  VHSR_CLKNAND2_2 U256 ( .A1(n274), .A2(n266), .ZN(n265) );
  VHSR_CLKNAND2_2 U257 ( .A1(n234), .A2(n265), .ZN(n262) );
  VHSR_CLKNAND2_2 U258 ( .A1(n263), .A2(n262), .ZN(n261) );
  VHSR_CLKNAND2_2 U259 ( .A1(n235), .A2(n261), .ZN(n253) );
  VHSR_IN_2 U260 ( .I(b[7]), .ZN(n289) );
  VHSR_IN_2 U261 ( .I(a[3]), .ZN(n330) );
  VHSR_IN_2 U262 ( .I(b[6]), .ZN(n290) );
  VHSR_IN_2 U263 ( .I(a[2]), .ZN(n332) );
  VHSR_OAI22_2 U264 ( .A1(n290), .A2(n330), .B1(n289), .B2(n332), .ZN(n251) );
  VHSR_AOI22_2 U265 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n242) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[4]), .A2(a[2]), .ZN(n273) );
  VHSR_NAND3_2 U267 ( .A1(a[3]), .A2(b[5]), .A3(n273), .ZN(n241) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[7]), .A2(a[2]), .ZN(n236) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[1]), .ZN(n238) );
  VHSR_OAI22_2 U270 ( .A1(n242), .A2(n241), .B1(n236), .B2(n238), .ZN(n243) );
  VHSR_IN_2 U271 ( .I(b[4]), .ZN(n349) );
  VHSR_IN_2 U272 ( .I(a[0]), .ZN(n412) );
  VHSR_OAI211_2 U273 ( .A1(n349), .A2(n412), .B(b[5]), .C(a[1]), .ZN(n272) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[6]), .A2(a[0]), .ZN(n271) );
  VHSR_MAOI222_2 U275 ( .A(n273), .B(n272), .C(n271), .ZN(n270) );
  VHSR_NAND4_2 U276 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n248) );
  VHSR_IN_2 U277 ( .I(b[5]), .ZN(n286) );
  VHSR_OAI22_2 U278 ( .A1(n349), .A2(n330), .B1(n286), .B2(n332), .ZN(n237) );
  VHSR_AND2_2 U279 ( .A1(n248), .A2(n237), .Z(n240) );
  VHSR_OAI21_2 U280 ( .A1(n289), .A2(n412), .B(n238), .ZN(n239) );
  VHSR_IN_2 U281 ( .I(a[1]), .ZN(n410) );
  VHSR_NOR4_2 U282 ( .A1(n349), .A2(n286), .A3(n410), .A4(n412), .ZN(n279) );
  VHSR_AND2_2 U283 ( .A1(n270), .A2(n269), .Z(n268) );
  VHSR_AD1_1 U284 ( .A(n240), .B(n239), .CI(n279), .CO(n257), .S(n269) );
  VHSR_AOI21_2 U285 ( .A1(n242), .A2(n241), .B(n243), .ZN(n260) );
  VHSR_OAI32_2 U286 ( .A1(n243), .A2(n268), .A3(n257), .B1(n260), .B2(n243), 
        .ZN(n249) );
  VHSR_CLKNAND2_2 U287 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U288 ( .A1(n251), .A2(n247), .ZN(n246) );
  VHSR_NOR3_2 U289 ( .A1(n289), .A2(n330), .A3(n246), .ZN(n304) );
  VHSR_NOR2_1 U290 ( .A1(n333), .A2(n287), .ZN(n244) );
  VHSR_IAO21_2 U291 ( .A1(n245), .A2(n244), .B(n305), .ZN(n308) );
  VHSR_OAI32_2 U292 ( .A1(n304), .A2(n330), .A3(n289), .B1(n246), .B2(n304), 
        .ZN(n307) );
  VHSR_OAI21_2 U293 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_XNOR2_2 U294 ( .A1(n251), .A2(n250), .ZN(n315) );
  VHSR_AOI21_2 U295 ( .A1(n254), .A2(n253), .B(n252), .ZN(n255) );
  VHSR_XNOR2_2 U296 ( .A1(n256), .A2(n255), .ZN(n314) );
  VHSR_NOR2_1 U297 ( .A1(n268), .A2(n257), .ZN(n259) );
  VHSR_AOI22_2 U298 ( .A1(n268), .A2(n257), .B1(n260), .B2(n259), .ZN(n258) );
  VHSR_OAI21_2 U299 ( .A1(n260), .A2(n259), .B(n258), .ZN(n320) );
  VHSR_OAI21_2 U300 ( .A1(n263), .A2(n262), .B(n261), .ZN(n264) );
  VHSR_IN_2 U301 ( .I(n264), .ZN(n319) );
  VHSR_OAI21_2 U302 ( .A1(n274), .A2(n266), .B(n265), .ZN(n267) );
  VHSR_IN_2 U303 ( .I(n267), .ZN(n337) );
  VHSR_IAO21_2 U304 ( .A1(n270), .A2(n269), .B(n268), .ZN(n336) );
  VHSR_AOI31_2 U305 ( .A1(n273), .A2(n272), .A3(n271), .B(n270), .ZN(n346) );
  VHSR_AOI31_2 U306 ( .A1(n277), .A2(n276), .A3(n275), .B(n274), .ZN(n345) );
  VHSR_CLKNAND2_2 U307 ( .A1(b[5]), .A2(a[0]), .ZN(n278) );
  VHSR_OAI32_2 U308 ( .A1(n279), .A2(n410), .A3(n349), .B1(n278), .B2(n279), 
        .ZN(n354) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[4]), .A2(b[4]), .ZN(n294) );
  VHSR_IN_2 U310 ( .I(n294), .ZN(n384) );
  VHSR_NOR2_1 U311 ( .A1(n411), .A2(n412), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U312 ( .A1(n384), .A2(product[0]), .ZN(n351) );
  VHSR_IN_2 U313 ( .I(n351), .ZN(n353) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[1]), .ZN(n280) );
  VHSR_OAI32_2 U315 ( .A1(n281), .A2(n292), .A3(n411), .B1(n280), .B2(n281), 
        .ZN(n352) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[6]), .A2(b[6]), .ZN(n375) );
  VHSR_IN_2 U317 ( .I(n375), .ZN(n402) );
  VHSR_CLKNAND2_2 U318 ( .A1(a[6]), .A2(b[4]), .ZN(n312) );
  VHSR_NAND3_2 U319 ( .A1(a[7]), .A2(b[5]), .A3(n312), .ZN(n283) );
  VHSR_CLKNAND2_2 U320 ( .A1(a[4]), .A2(b[6]), .ZN(n311) );
  VHSR_NAND3_2 U321 ( .A1(b[7]), .A2(a[5]), .A3(n311), .ZN(n282) );
  VHSR_CLKNAND2_2 U322 ( .A1(n283), .A2(n282), .ZN(n285) );
  VHSR_MAOI222_2 U323 ( .A(n375), .B(n283), .C(n282), .ZN(n359) );
  VHSR_IN_2 U324 ( .I(n359), .ZN(n284) );
  VHSR_OAI21_2 U325 ( .A1(n402), .A2(n285), .B(n284), .ZN(n300) );
  VHSR_NOR3_2 U326 ( .A1(n292), .A2(n286), .A3(n294), .ZN(n316) );
  VHSR_NOR3_2 U327 ( .A1(n287), .A2(n312), .A3(n286), .ZN(n367) );
  VHSR_AOI22_2 U328 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n288) );
  VHSR_NOR2_1 U329 ( .A1(n367), .A2(n288), .ZN(n296) );
  VHSR_NOR4_2 U330 ( .A1(n292), .A2(n291), .A3(n290), .A4(n289), .ZN(n365) );
  VHSR_AOI22_2 U331 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n293) );
  VHSR_NOR2_1 U332 ( .A1(n365), .A2(n293), .ZN(n295) );
  VHSR_NAND3_2 U333 ( .A1(b[5]), .A2(a[5]), .A3(n294), .ZN(n310) );
  VHSR_MAOI222_2 U334 ( .A(n312), .B(n311), .C(n310), .ZN(n309) );
  VHSR_AND2_2 U335 ( .A1(n302), .A2(n309), .Z(n301) );
  VHSR_AD1_1 U336 ( .A(n316), .B(n296), .CI(n295), .CO(n297), .S(n302) );
  VHSR_NOR2_1 U337 ( .A1(n301), .A2(n297), .ZN(n299) );
  VHSR_CLKNAND2_2 U338 ( .A1(n301), .A2(n297), .ZN(n298) );
  VHSR_NOR2_1 U339 ( .A1(n299), .A2(n300), .ZN(n360) );
  VHSR_AOI22_2 U340 ( .A1(n300), .A2(n299), .B1(n298), .B2(n360), .ZN(n400) );
  VHSR_IAO21_2 U341 ( .A1(n302), .A2(n309), .B(n301), .ZN(n398) );
  VHSR_AD1_1 U342 ( .A(n305), .B(n304), .CI(n303), .CO(n401), .S(n397) );
  VHSR_AD1_1 U343 ( .A(n308), .B(n307), .CI(n306), .CO(n303), .S(n379) );
  VHSR_AOI31_2 U344 ( .A1(n312), .A2(n311), .A3(n310), .B(n309), .ZN(n378) );
  VHSR_AD1_1 U345 ( .A(n315), .B(n314), .CI(n313), .CO(n306), .S(n382) );
  VHSR_AOI22_2 U346 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n317) );
  VHSR_NOR2_1 U347 ( .A1(n317), .A2(n316), .ZN(n381) );
  VHSR_AD1_1 U348 ( .A(n320), .B(n319), .CI(n318), .CO(n313), .S(n385) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[2]), .A2(a[0]), .ZN(n417) );
  VHSR_NOR3_2 U350 ( .A1(n333), .A2(n410), .A3(n417), .ZN(n328) );
  VHSR_AOI22_2 U351 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n321) );
  VHSR_NOR2_1 U352 ( .A1(n328), .A2(n321), .ZN(n392) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[0]), .A2(a[2]), .ZN(n416) );
  VHSR_IN_2 U354 ( .I(n416), .ZN(n329) );
  VHSR_AOI22_2 U355 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n322) );
  VHSR_AOI31_2 U356 ( .A1(a[3]), .A2(b[1]), .A3(n329), .B(n322), .ZN(n391) );
  VHSR_CLKNAND2_2 U357 ( .A1(b[1]), .A2(a[1]), .ZN(n415) );
  VHSR_MAOI222_2 U358 ( .A(n417), .B(n416), .C(n415), .ZN(n414) );
  VHSR_NAND3_2 U359 ( .A1(n416), .A2(a[3]), .A3(b[1]), .ZN(n324) );
  VHSR_IN_2 U360 ( .I(n324), .ZN(n323) );
  VHSR_AOI31_2 U361 ( .A1(a[1]), .A2(b[3]), .A3(n417), .B(n323), .ZN(n326) );
  VHSR_CLKNAND2_2 U362 ( .A1(b[2]), .A2(a[2]), .ZN(n343) );
  VHSR_NAND3_2 U363 ( .A1(a[1]), .A2(b[3]), .A3(n417), .ZN(n325) );
  VHSR_MAOI222_2 U364 ( .A(n343), .B(n325), .C(n324), .ZN(n327) );
  VHSR_AOI21_2 U365 ( .A1(n326), .A2(n343), .B(n327), .ZN(n347) );
  VHSR_AOI21_2 U366 ( .A1(n348), .A2(n347), .B(n327), .ZN(n357) );
  VHSR_AOI31_2 U367 ( .A1(a[3]), .A2(b[1]), .A3(n329), .B(n328), .ZN(n356) );
  VHSR_CLKNAND2_2 U368 ( .A1(b[3]), .A2(a[3]), .ZN(n339) );
  VHSR_OAI22_2 U369 ( .A1(n333), .A2(n332), .B1(n331), .B2(n330), .ZN(n334) );
  VHSR_OAI21_2 U370 ( .A1(n339), .A2(n343), .B(n334), .ZN(n355) );
  VHSR_AOI21_2 U371 ( .A1(n338), .A2(n343), .B(n339), .ZN(n388) );
  VHSR_AD1_1 U372 ( .A(n337), .B(n336), .CI(n335), .CO(n318), .S(n387) );
  VHSR_IN_2 U373 ( .I(n338), .ZN(n342) );
  VHSR_IN_2 U374 ( .I(n339), .ZN(n341) );
  VHSR_AOI21_2 U375 ( .A1(n343), .A2(n341), .B(n342), .ZN(n340) );
  VHSR_AOI31_2 U376 ( .A1(n343), .A2(n342), .A3(n341), .B(n340), .ZN(n395) );
  VHSR_AD1_1 U377 ( .A(n346), .B(n345), .CI(n344), .CO(n335), .S(n394) );
  VHSR_XNOR2_2 U378 ( .A1(n348), .A2(n347), .ZN(n409) );
  VHSR_NOR2_1 U379 ( .A1(n349), .A2(n412), .ZN(n350) );
  VHSR_AOI32_2 U380 ( .A1(b[0]), .A2(n351), .A3(a[4]), .B1(n350), .B2(n351), 
        .ZN(n408) );
  VHSR_AD1_1 U381 ( .A(n354), .B(n353), .CI(n352), .CO(n344), .S(n390) );
  VHSR_AD1_1 U382 ( .A(n357), .B(n356), .CI(n355), .CO(n338), .S(n358) );
  VHSR_IN_2 U383 ( .I(n358), .ZN(n389) );
  VHSR_CLKNAND2_2 U384 ( .A1(a[7]), .A2(b[6]), .ZN(n362) );
  VHSR_AOI21_2 U385 ( .A1(a[6]), .A2(b[7]), .B(n362), .ZN(n361) );
  VHSR_AOI31_2 U386 ( .A1(a[6]), .A2(n362), .A3(b[7]), .B(n361), .ZN(n363) );
  VHSR_IN_2 U387 ( .I(n363), .ZN(n364) );
  VHSR_MAOI222_2 U388 ( .A(n367), .B(n365), .C(n364), .ZN(n374) );
  VHSR_OAI21_2 U389 ( .A1(n367), .A2(n366), .B(n374), .ZN(n371) );
  VHSR_CLKXOR2_2 U390 ( .A1(n372), .A2(n371), .Z(n368) );
  VHSR_CLKNAND2_2 U391 ( .A1(n369), .A2(n368), .ZN(n404) );
  VHSR_OAI21_2 U392 ( .A1(n369), .A2(n368), .B(n404), .ZN(n370) );
  VHSR_CLKNAND2_2 U393 ( .A1(a[7]), .A2(b[7]), .ZN(n403) );
  VHSR_NOR2_1 U394 ( .A1(n372), .A2(n371), .ZN(n373) );
  VHSR_AND3_2 U395 ( .A1(n405), .A2(n375), .A3(n404), .Z(n376) );
  VHSR_NOR2_1 U396 ( .A1(n403), .A2(n376), .ZN(product[15]) );
  VHSR_AD1_1 U397 ( .A(n395), .B(n394), .CI(n393), .CO(n386), .S(product[6])
         );
  VHSR_AD1_1 U398 ( .A(n398), .B(n397), .CI(n396), .CO(n399), .S(product[11])
         );
  VHSR_AD1_1 U399 ( .A(n401), .B(n400), .CI(n399), .CO(n369), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U400 ( .A1(n403), .A2(n402), .ZN(n406) );
  VHSR_XOR3_2 U401 ( .A1(n406), .A2(n405), .A3(n404), .Z(product[14]) );
  VHSR_AOI21_2 U402 ( .A1(n409), .A2(n408), .B(n407), .ZN(product[4]) );
  VHSR_OAI22_2 U403 ( .A1(n413), .A2(n412), .B1(n411), .B2(n410), .ZN(
        product[1]) );
  VHSR_AOI31_2 U404 ( .A1(n417), .A2(n416), .A3(n415), .B(n414), .ZN(
        product[2]) );
endmodule


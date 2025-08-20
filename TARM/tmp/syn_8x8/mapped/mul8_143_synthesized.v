
module mul8_143 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n214, n215, n216, n217, n218,
         n219, n220, n221, n222, n223, n224, n225, n226, n227, n228, n229,
         n230, n231, n232, n233, n234, n235, n236, n237, n238, n239, n240,
         n241, n242, n243, n244, n245, n246, n247, n248, n249, n250, n251,
         n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
         n263, n264, n265, n266, n267, n268, n269, n270, n271, n272, n273,
         n274, n275, n276, n277, n278, n279, n280, n281, n282, n283, n284,
         n285, n286, n287, n288, n289, n290, n291, n292, n293, n294, n295,
         n296, n297, n298, n299, n300, n301, n302, n303, n304, n305, n306,
         n307, n308, n309, n310, n311, n312, n313, n314, n315, n316, n317,
         n318, n319, n320, n321, n322, n323, n324, n325, n326, n327, n328,
         n329, n330, n331, n332, n333, n334, n335, n336, n337, n338, n339,
         n340, n341, n342, n343, n344, n345, n346, n347, n348, n349, n350,
         n351, n352, n353, n354, n355, n356, n357, n358, n359, n360, n361,
         n362, n363, n364, n365, n366, n367, n368, n369, n370, n371, n372,
         n373, n374, n375, n376, n377, n378, n379, n380, n381, n382, n383,
         n384, n385, n386, n387, n388, n389, n390, n391, n392, n393, n394,
         n395, n396, n397, n398, n399, n400, n401, n402;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U207 ( .A1(n269), .B1(n218), .ZN(n220) );
  VHSR_INOR2_2 U208 ( .A1(n231), .B1(n249), .ZN(n237) );
  VHSR_INOR2_2 U209 ( .A1(n221), .B1(n255), .ZN(n247) );
  VHSR_INOR2_2 U210 ( .A1(n344), .B1(n343), .ZN(n354) );
  VHSR_NOR2_1 U211 ( .A1(n233), .A2(n232), .ZN(n294) );
  VHSR_IOA21_2 U212 ( .A1(n392), .A2(n391), .B(n390), .ZN(n394) );
  VHSR_NOR2_1 U213 ( .A1(n306), .A2(n307), .ZN(n367) );
  VHSR_IN_2 U214 ( .I(n353), .ZN(product[13]) );
  VHSR_CLKN_1 U215 ( .I(n358), .ZN(n359) );
  VHSR_INAND3_1 U216 ( .A1(n385), .B1(n388), .B2(n387), .ZN(n358) );
  VHSR_INOR2_1 U217 ( .A1(n355), .B1(n354), .ZN(n357) );
  VHSR_INOR2_1 U218 ( .A1(n223), .B1(n245), .ZN(n242) );
  VHSR_INAND2_1 U219 ( .A1(n270), .B1(n350), .ZN(n273) );
  VHSR_AD1_1 U220 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(product[9])
         );
  VHSR_AD1_1 U221 ( .A(n375), .B(n402), .CI(n374), .CO(n336), .S(product[3])
         );
  VHSR_AD1_1 U222 ( .A(n393), .B(n373), .CI(n372), .CO(n376), .S(product[5])
         );
  VHSR_AD1_1 U223 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U224 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n362), .B(n361), .CI(n360), .CO(n379), .S(product[10])
         );
  VHSR_CLKNAND2_2 U226 ( .A1(a[7]), .A2(b[3]), .ZN(n233) );
  VHSR_IN_2 U227 ( .I(a[6]), .ZN(n279) );
  VHSR_IN_2 U228 ( .I(b[3]), .ZN(n321) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n282) );
  VHSR_IN_2 U230 ( .I(b[2]), .ZN(n400) );
  VHSR_OAI22_2 U231 ( .A1(n279), .A2(n321), .B1(n282), .B2(n400), .ZN(n244) );
  VHSR_IN_2 U232 ( .I(b[1]), .ZN(n398) );
  VHSR_IN_2 U233 ( .I(a[4]), .ZN(n307) );
  VHSR_NOR2_1 U234 ( .A1(n307), .A2(n400), .ZN(n264) );
  VHSR_IN_2 U235 ( .I(a[5]), .ZN(n305) );
  VHSR_OR3_2 U236 ( .A1(n264), .A2(n321), .A3(n305), .Z(n214) );
  VHSR_OAI21_2 U237 ( .A1(n398), .A2(n282), .B(n214), .ZN(n222) );
  VHSR_NOR4_2 U238 ( .A1(n264), .A2(n233), .A3(n305), .A4(n398), .ZN(n215) );
  VHSR_AOI31_2 U239 ( .A1(b[2]), .A2(a[6]), .A3(n222), .B(n215), .ZN(n223) );
  VHSR_IN_2 U240 ( .I(b[0]), .ZN(n397) );
  VHSR_NOR4_2 U241 ( .A1(n307), .A2(n305), .A3(n398), .A4(n397), .ZN(n269) );
  VHSR_NAND4_2 U242 ( .A1(a[4]), .A2(a[5]), .A3(b[3]), .A4(b[2]), .ZN(n241) );
  VHSR_NOR2_1 U243 ( .A1(n307), .A2(n321), .ZN(n216) );
  VHSR_AOI32_2 U244 ( .A1(b[2]), .A2(n241), .A3(a[5]), .B1(n216), .B2(n241), 
        .ZN(n218) );
  VHSR_IN_2 U245 ( .I(n218), .ZN(n217) );
  VHSR_OAI22_2 U246 ( .A1(n279), .A2(n398), .B1(n282), .B2(n397), .ZN(n219) );
  VHSR_MAOI222_2 U247 ( .A(n269), .B(n217), .C(n219), .ZN(n221) );
  VHSR_AOI211_2 U248 ( .A1(a[4]), .A2(b[0]), .B(n305), .C(n398), .ZN(n263) );
  VHSR_NOR2_1 U249 ( .A1(n279), .A2(n397), .ZN(n262) );
  VHSR_MAOI222_2 U250 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_OAI21_2 U251 ( .A1(n220), .A2(n219), .B(n221), .ZN(n256) );
  VHSR_NOR2_1 U252 ( .A1(n261), .A2(n256), .ZN(n255) );
  VHSR_AOI32_2 U253 ( .A1(b[2]), .A2(n223), .A3(a[6]), .B1(n222), .B2(n223), 
        .ZN(n246) );
  VHSR_NOR2_1 U254 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_CLKNAND2_2 U255 ( .A1(n242), .A2(n241), .ZN(n240) );
  VHSR_CLKNAND2_2 U256 ( .A1(n244), .A2(n240), .ZN(n232) );
  VHSR_IN_2 U257 ( .I(b[7]), .ZN(n271) );
  VHSR_IN_2 U258 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U259 ( .I(b[6]), .ZN(n278) );
  VHSR_IN_2 U260 ( .I(a[2]), .ZN(n322) );
  VHSR_OAI22_2 U261 ( .A1(n278), .A2(n324), .B1(n271), .B2(n322), .ZN(n239) );
  VHSR_NOR2_1 U262 ( .A1(n271), .A2(n322), .ZN(n225) );
  VHSR_IN_2 U263 ( .I(a[1]), .ZN(n396) );
  VHSR_NOR2_1 U264 ( .A1(n278), .A2(n396), .ZN(n224) );
  VHSR_IN_2 U265 ( .I(b[5]), .ZN(n308) );
  VHSR_AOI211_2 U266 ( .A1(a[2]), .A2(b[4]), .B(n308), .C(n324), .ZN(n230) );
  VHSR_OAI22_2 U267 ( .A1(n278), .A2(n322), .B1(n271), .B2(n396), .ZN(n229) );
  VHSR_AOI22_2 U268 ( .A1(n225), .A2(n224), .B1(n230), .B2(n229), .ZN(n231) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[4]), .A2(a[2]), .ZN(n260) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[4]), .A2(a[0]), .ZN(n391) );
  VHSR_NAND3_2 U271 ( .A1(a[1]), .A2(b[5]), .A3(n391), .ZN(n259) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[0]), .ZN(n258) );
  VHSR_MAOI222_2 U273 ( .A(n260), .B(n259), .C(n258), .ZN(n257) );
  VHSR_NOR3_2 U274 ( .A1(n308), .A2(n396), .A3(n391), .ZN(n266) );
  VHSR_NAND4_2 U275 ( .A1(b[5]), .A2(b[4]), .A3(a[2]), .A4(a[3]), .ZN(n236) );
  VHSR_IN_2 U276 ( .I(b[4]), .ZN(n306) );
  VHSR_OAI22_2 U277 ( .A1(n308), .A2(n322), .B1(n306), .B2(n324), .ZN(n226) );
  VHSR_AND2_2 U278 ( .A1(n236), .A2(n226), .Z(n228) );
  VHSR_IN_2 U279 ( .I(a[0]), .ZN(n401) );
  VHSR_OAI22_2 U280 ( .A1(n278), .A2(n396), .B1(n271), .B2(n401), .ZN(n227) );
  VHSR_AND2_2 U281 ( .A1(n257), .A2(n254), .Z(n253) );
  VHSR_AD1_1 U282 ( .A(n266), .B(n228), .CI(n227), .CO(n248), .S(n254) );
  VHSR_NOR2_1 U283 ( .A1(n253), .A2(n248), .ZN(n251) );
  VHSR_OAI21_2 U284 ( .A1(n230), .A2(n229), .B(n231), .ZN(n252) );
  VHSR_NOR2_1 U285 ( .A1(n251), .A2(n252), .ZN(n249) );
  VHSR_CLKNAND2_2 U286 ( .A1(n237), .A2(n236), .ZN(n235) );
  VHSR_CLKNAND2_2 U287 ( .A1(n239), .A2(n235), .ZN(n234) );
  VHSR_NOR3_2 U288 ( .A1(n271), .A2(n324), .A3(n234), .ZN(n293) );
  VHSR_AOI21_2 U289 ( .A1(n233), .A2(n232), .B(n294), .ZN(n297) );
  VHSR_OAI32_2 U290 ( .A1(n293), .A2(n324), .A3(n271), .B1(n234), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U291 ( .A1(n237), .A2(n236), .B(n235), .ZN(n238) );
  VHSR_XNOR2_2 U292 ( .A1(n239), .A2(n238), .ZN(n304) );
  VHSR_OAI21_2 U293 ( .A1(n242), .A2(n241), .B(n240), .ZN(n243) );
  VHSR_XNOR2_2 U294 ( .A1(n244), .A2(n243), .ZN(n303) );
  VHSR_AOI21_2 U295 ( .A1(n247), .A2(n246), .B(n245), .ZN(n313) );
  VHSR_CLKNAND2_2 U296 ( .A1(n253), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U297 ( .A1(n252), .A2(n251), .B1(n250), .B2(n249), .ZN(n312) );
  VHSR_IAO21_2 U298 ( .A1(n257), .A2(n254), .B(n253), .ZN(n316) );
  VHSR_AOI21_2 U299 ( .A1(n261), .A2(n256), .B(n255), .ZN(n315) );
  VHSR_AOI31_2 U300 ( .A1(n260), .A2(n259), .A3(n258), .B(n257), .ZN(n330) );
  VHSR_OAI31_2 U301 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n265) );
  VHSR_IN_2 U302 ( .I(n265), .ZN(n329) );
  VHSR_AOI22_2 U303 ( .A1(b[5]), .A2(a[0]), .B1(b[4]), .B2(a[1]), .ZN(n267) );
  VHSR_NOR2_1 U304 ( .A1(n267), .A2(n266), .ZN(n342) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[5]), .A2(b[0]), .ZN(n268) );
  VHSR_OAI32_2 U306 ( .A1(n269), .A2(n398), .A3(n307), .B1(n268), .B2(n269), 
        .ZN(n341) );
  VHSR_NOR2_1 U307 ( .A1(n397), .A2(n401), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U308 ( .A1(n367), .A2(product[0]), .ZN(n390) );
  VHSR_IN_2 U309 ( .I(n390), .ZN(n340) );
  VHSR_AOI22_2 U310 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n270) );
  VHSR_NAND4_2 U311 ( .A1(a[6]), .A2(a[7]), .A3(b[5]), .A4(b[4]), .ZN(n350) );
  VHSR_NAND3_2 U312 ( .A1(b[5]), .A2(a[5]), .A3(n367), .ZN(n310) );
  VHSR_NAND4_2 U313 ( .A1(b[6]), .A2(b[7]), .A3(a[4]), .A4(a[5]), .ZN(n348) );
  VHSR_NOR2_1 U314 ( .A1(n271), .A2(n307), .ZN(n272) );
  VHSR_AOI32_2 U315 ( .A1(b[6]), .A2(n348), .A3(a[5]), .B1(n272), .B2(n348), 
        .ZN(n275) );
  VHSR_MAOI222_2 U316 ( .A(n273), .B(n310), .C(n275), .ZN(n277) );
  VHSR_AND2_2 U317 ( .A1(n273), .A2(n310), .Z(n274) );
  VHSR_AOI21_2 U318 ( .A1(n275), .A2(n274), .B(n277), .ZN(n276) );
  VHSR_IN_2 U319 ( .I(n276), .ZN(n291) );
  VHSR_NOR2_1 U320 ( .A1(n279), .A2(n306), .ZN(n284) );
  VHSR_NOR2_1 U321 ( .A1(n278), .A2(n307), .ZN(n283) );
  VHSR_CLKNAND2_2 U322 ( .A1(b[5]), .A2(a[5]), .ZN(n285) );
  VHSR_NOR2_1 U323 ( .A1(n367), .A2(n285), .ZN(n300) );
  VHSR_MAOI222_2 U324 ( .A(n284), .B(n283), .C(n300), .ZN(n298) );
  VHSR_NOR2_1 U325 ( .A1(n291), .A2(n298), .ZN(n290) );
  VHSR_NOR2_1 U326 ( .A1(n277), .A2(n290), .ZN(n289) );
  VHSR_NOR2_1 U327 ( .A1(n279), .A2(n278), .ZN(n385) );
  VHSR_IN_2 U328 ( .I(n283), .ZN(n280) );
  VHSR_NAND3_2 U329 ( .A1(a[5]), .A2(b[7]), .A3(n280), .ZN(n281) );
  VHSR_OAI31_2 U330 ( .A1(n284), .A2(n308), .A3(n282), .B(n281), .ZN(n287) );
  VHSR_CLKNAND2_2 U331 ( .A1(a[7]), .A2(b[7]), .ZN(n386) );
  VHSR_OR2_2 U332 ( .A1(n284), .A2(n283), .Z(n299) );
  VHSR_NOR3_2 U333 ( .A1(n386), .A2(n299), .A3(n285), .ZN(n286) );
  VHSR_AOI31_2 U334 ( .A1(b[6]), .A2(a[6]), .A3(n287), .B(n286), .ZN(n344) );
  VHSR_OAI21_2 U335 ( .A1(n385), .A2(n287), .B(n344), .ZN(n288) );
  VHSR_NOR2_1 U336 ( .A1(n289), .A2(n288), .ZN(n343) );
  VHSR_AOI21_2 U337 ( .A1(n289), .A2(n288), .B(n343), .ZN(n383) );
  VHSR_AOI21_2 U338 ( .A1(n291), .A2(n298), .B(n290), .ZN(n381) );
  VHSR_AD1_1 U339 ( .A(n294), .B(n293), .CI(n292), .CO(n384), .S(n380) );
  VHSR_AD1_1 U340 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n362) );
  VHSR_OAI21_2 U341 ( .A1(n300), .A2(n299), .B(n298), .ZN(n301) );
  VHSR_IN_2 U342 ( .I(n301), .ZN(n361) );
  VHSR_AD1_1 U343 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n365) );
  VHSR_OAI22_2 U344 ( .A1(n308), .A2(n307), .B1(n306), .B2(n305), .ZN(n309) );
  VHSR_AND2_2 U345 ( .A1(n310), .A2(n309), .Z(n364) );
  VHSR_AD1_1 U346 ( .A(n313), .B(n312), .CI(n311), .CO(n302), .S(n368) );
  VHSR_AD1_1 U347 ( .A(n316), .B(n315), .CI(n314), .CO(n311), .S(n371) );
  VHSR_NOR2_1 U348 ( .A1(n400), .A2(n322), .ZN(n333) );
  VHSR_NOR2_1 U349 ( .A1(n400), .A2(n324), .ZN(n318) );
  VHSR_OAI21_2 U350 ( .A1(n321), .A2(n322), .B(n318), .ZN(n317) );
  VHSR_OAI31_2 U351 ( .A1(n321), .A2(n318), .A3(n322), .B(n317), .ZN(n339) );
  VHSR_NOR2_1 U352 ( .A1(n321), .A2(n396), .ZN(n320) );
  VHSR_NOR2_1 U353 ( .A1(n398), .A2(n324), .ZN(n319) );
  VHSR_MAOI222_2 U354 ( .A(n333), .B(n320), .C(n319), .ZN(n326) );
  VHSR_OAI22_2 U355 ( .A1(n321), .A2(n401), .B1(n400), .B2(n396), .ZN(n375) );
  VHSR_AOI22_2 U356 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n399) );
  VHSR_NOR3_2 U357 ( .A1(n399), .A2(n401), .A3(n400), .ZN(n402) );
  VHSR_OAI22_2 U358 ( .A1(n398), .A2(n322), .B1(n397), .B2(n324), .ZN(n374) );
  VHSR_IN_2 U359 ( .I(n326), .ZN(n325) );
  VHSR_AOI21_2 U360 ( .A1(a[1]), .A2(b[3]), .B(n333), .ZN(n323) );
  VHSR_OAI32_2 U361 ( .A1(n325), .A2(n324), .A3(n398), .B1(n323), .B2(n325), 
        .ZN(n335) );
  VHSR_CLKNAND2_2 U362 ( .A1(n336), .A2(n335), .ZN(n334) );
  VHSR_CLKNAND2_2 U363 ( .A1(n326), .A2(n334), .ZN(n338) );
  VHSR_AND2_2 U364 ( .A1(n339), .A2(n338), .Z(n337) );
  VHSR_OAI211_2 U365 ( .A1(n333), .A2(n337), .B(a[3]), .C(b[3]), .ZN(n327) );
  VHSR_IN_2 U366 ( .I(n327), .ZN(n370) );
  VHSR_AD1_1 U367 ( .A(n330), .B(n329), .CI(n328), .CO(n314), .S(n378) );
  VHSR_CLKNAND2_2 U368 ( .A1(b[3]), .A2(a[3]), .ZN(n332) );
  VHSR_CLKNAND2_2 U369 ( .A1(n337), .A2(n332), .ZN(n331) );
  VHSR_OAI31_2 U370 ( .A1(n333), .A2(n337), .A3(n332), .B(n331), .ZN(n377) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[4]), .A2(b[0]), .ZN(n392) );
  VHSR_OAI21_2 U372 ( .A1(n336), .A2(n335), .B(n334), .ZN(n395) );
  VHSR_AOI211_2 U373 ( .A1(n392), .A2(n391), .B(n340), .C(n395), .ZN(n393) );
  VHSR_IAO21_2 U374 ( .A1(n339), .A2(n338), .B(n337), .ZN(n373) );
  VHSR_AD1_1 U375 ( .A(n342), .B(n341), .CI(n340), .CO(n328), .S(n372) );
  VHSR_CLKNAND2_2 U376 ( .A1(b[6]), .A2(a[7]), .ZN(n346) );
  VHSR_AOI21_2 U377 ( .A1(a[6]), .A2(b[7]), .B(n346), .ZN(n345) );
  VHSR_AOI31_2 U378 ( .A1(a[6]), .A2(n346), .A3(b[7]), .B(n345), .ZN(n347) );
  VHSR_AND2_2 U379 ( .A1(n348), .A2(n347), .Z(n349) );
  VHSR_MAOI222_2 U380 ( .A(n350), .B(n348), .C(n347), .ZN(n356) );
  VHSR_AOI21_2 U381 ( .A1(n350), .A2(n349), .B(n356), .ZN(n355) );
  VHSR_XNOR2_2 U382 ( .A1(n354), .A2(n355), .ZN(n351) );
  VHSR_CLKNAND2_2 U383 ( .A1(n352), .A2(n351), .ZN(n387) );
  VHSR_OAI21_2 U384 ( .A1(n352), .A2(n351), .B(n387), .ZN(n353) );
  VHSR_NOR2_1 U385 ( .A1(n357), .A2(n356), .ZN(n388) );
  VHSR_NOR2_1 U386 ( .A1(n386), .A2(n359), .ZN(product[15]) );
  VHSR_AD1_1 U387 ( .A(n378), .B(n377), .CI(n376), .CO(n369), .S(product[6])
         );
  VHSR_AD1_1 U388 ( .A(n381), .B(n380), .CI(n379), .CO(n382), .S(product[11])
         );
  VHSR_AD1_1 U389 ( .A(n384), .B(n383), .CI(n382), .CO(n352), .S(product[12])
         );
  VHSR_NOR2_1 U390 ( .A1(n386), .A2(n385), .ZN(n389) );
  VHSR_XOR3_2 U391 ( .A1(n389), .A2(n388), .A3(n387), .Z(product[14]) );
  VHSR_AOI21_2 U392 ( .A1(n395), .A2(n394), .B(n393), .ZN(product[4]) );
  VHSR_OAI22_2 U393 ( .A1(n398), .A2(n401), .B1(n397), .B2(n396), .ZN(
        product[1]) );
  VHSR_OAI32_2 U394 ( .A1(n402), .A2(n401), .A3(n400), .B1(n399), .B2(n402), 
        .ZN(product[2]) );
endmodule


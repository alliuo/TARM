
module mul8_136 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n215, n216, n217, n218, n219,
         n220, n221, n222, n223, n224, n225, n226, n227, n228, n229, n230,
         n231, n232, n233, n234, n235, n236, n237, n238, n239, n240, n241,
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
         n396, n397, n398, n399, n400, n401, n402, n403, n404, n405, n406;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U208 ( .A1(n224), .B1(n247), .ZN(n244) );
  VHSR_INOR2_2 U209 ( .A1(n233), .B1(n251), .ZN(n239) );
  VHSR_INOR2_2 U210 ( .A1(n222), .B1(n255), .ZN(n249) );
  VHSR_INOR2_2 U211 ( .A1(n348), .B1(n347), .ZN(n358) );
  VHSR_INAND2_2 U212 ( .A1(n324), .B1(n338), .ZN(n342) );
  VHSR_NOR2_1 U213 ( .A1(n235), .A2(n234), .ZN(n296) );
  VHSR_IOA21_2 U214 ( .A1(n321), .A2(n320), .B(n319), .ZN(n404) );
  VHSR_NOR2_1 U215 ( .A1(n308), .A2(n309), .ZN(n381) );
  VHSR_IN_2 U216 ( .I(n357), .ZN(product[13]) );
  VHSR_CLKN_1 U217 ( .I(n362), .ZN(n363) );
  VHSR_INAND3_1 U218 ( .A1(n389), .B1(n392), .B2(n391), .ZN(n362) );
  VHSR_INOR2_1 U219 ( .A1(n359), .B1(n358), .ZN(n361) );
  VHSR_IOA21_1 U220 ( .A1(n396), .A2(n395), .B(n394), .ZN(n398) );
  VHSR_INAND2_1 U221 ( .A1(n272), .B1(n354), .ZN(n275) );
  VHSR_AD1_1 U222 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(product[6])
         );
  VHSR_AD1_1 U223 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(product[9])
         );
  VHSR_AD1_1 U224 ( .A(n379), .B(n404), .CI(n378), .CO(n340), .S(product[3])
         );
  VHSR_AD1_1 U225 ( .A(n397), .B(n377), .CI(n376), .CO(n373), .S(product[5])
         );
  VHSR_AD1_1 U226 ( .A(n372), .B(n371), .CI(n370), .CO(n380), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U227 ( .A(n366), .B(n365), .CI(n364), .CO(n383), .S(product[10])
         );
  VHSR_CLKNAND2_2 U228 ( .A1(a[7]), .A2(b[3]), .ZN(n235) );
  VHSR_IN_2 U229 ( .I(a[6]), .ZN(n281) );
  VHSR_IN_2 U230 ( .I(b[3]), .ZN(n325) );
  VHSR_IN_2 U231 ( .I(a[7]), .ZN(n284) );
  VHSR_IN_2 U232 ( .I(b[2]), .ZN(n318) );
  VHSR_OAI22_2 U233 ( .A1(n281), .A2(n325), .B1(n284), .B2(n318), .ZN(n246) );
  VHSR_IN_2 U234 ( .I(b[1]), .ZN(n403) );
  VHSR_IN_2 U235 ( .I(a[4]), .ZN(n309) );
  VHSR_NOR2_1 U236 ( .A1(n309), .A2(n318), .ZN(n262) );
  VHSR_IN_2 U237 ( .I(a[5]), .ZN(n307) );
  VHSR_OR3_2 U238 ( .A1(n262), .A2(n325), .A3(n307), .Z(n215) );
  VHSR_OAI21_2 U239 ( .A1(n403), .A2(n284), .B(n215), .ZN(n223) );
  VHSR_NOR4_2 U240 ( .A1(n262), .A2(n235), .A3(n307), .A4(n403), .ZN(n216) );
  VHSR_AOI31_2 U241 ( .A1(b[2]), .A2(a[6]), .A3(n223), .B(n216), .ZN(n224) );
  VHSR_NOR2_1 U242 ( .A1(n281), .A2(n403), .ZN(n219) );
  VHSR_IN_2 U243 ( .I(b[0]), .ZN(n401) );
  VHSR_NOR4_2 U244 ( .A1(n309), .A2(n307), .A3(n403), .A4(n401), .ZN(n271) );
  VHSR_CLKNAND2_2 U245 ( .A1(a[5]), .A2(b[2]), .ZN(n218) );
  VHSR_CLKNAND2_2 U246 ( .A1(a[4]), .A2(b[3]), .ZN(n217) );
  VHSR_NOR4_2 U247 ( .A1(n309), .A2(n307), .A3(n325), .A4(n318), .ZN(n225) );
  VHSR_AOI21_2 U248 ( .A1(n218), .A2(n217), .B(n225), .ZN(n220) );
  VHSR_MAOI222_2 U249 ( .A(n219), .B(n271), .C(n220), .ZN(n222) );
  VHSR_AOI211_2 U250 ( .A1(a[4]), .A2(b[0]), .B(n307), .C(n403), .ZN(n261) );
  VHSR_AOI21_2 U251 ( .A1(n281), .A2(n284), .B(n401), .ZN(n260) );
  VHSR_MAOI222_2 U252 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_OR2_2 U253 ( .A1(n271), .A2(n220), .Z(n221) );
  VHSR_AOI32_2 U254 ( .A1(b[1]), .A2(n222), .A3(a[6]), .B1(n221), .B2(n222), 
        .ZN(n256) );
  VHSR_NOR2_1 U255 ( .A1(n259), .A2(n256), .ZN(n255) );
  VHSR_AOI32_2 U256 ( .A1(b[2]), .A2(n224), .A3(a[6]), .B1(n223), .B2(n224), 
        .ZN(n248) );
  VHSR_NOR2_1 U257 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_IN_2 U258 ( .I(n225), .ZN(n243) );
  VHSR_CLKNAND2_2 U259 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_CLKNAND2_2 U260 ( .A1(n246), .A2(n242), .ZN(n234) );
  VHSR_IN_2 U261 ( .I(b[7]), .ZN(n273) );
  VHSR_IN_2 U262 ( .I(a[3]), .ZN(n326) );
  VHSR_IN_2 U263 ( .I(b[6]), .ZN(n280) );
  VHSR_IN_2 U264 ( .I(a[2]), .ZN(n322) );
  VHSR_OAI22_2 U265 ( .A1(n280), .A2(n326), .B1(n273), .B2(n322), .ZN(n241) );
  VHSR_NOR2_1 U266 ( .A1(n273), .A2(n322), .ZN(n227) );
  VHSR_IN_2 U267 ( .I(a[1]), .ZN(n400) );
  VHSR_NOR2_1 U268 ( .A1(n280), .A2(n400), .ZN(n226) );
  VHSR_IN_2 U269 ( .I(b[5]), .ZN(n310) );
  VHSR_AOI211_2 U270 ( .A1(a[2]), .A2(b[4]), .B(n310), .C(n326), .ZN(n232) );
  VHSR_OAI22_2 U271 ( .A1(n280), .A2(n322), .B1(n273), .B2(n400), .ZN(n231) );
  VHSR_AOI22_2 U272 ( .A1(n227), .A2(n226), .B1(n232), .B2(n231), .ZN(n233) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[4]), .A2(a[2]), .ZN(n267) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[6]), .A2(a[0]), .ZN(n266) );
  VHSR_CLKNAND2_2 U275 ( .A1(b[4]), .A2(a[0]), .ZN(n395) );
  VHSR_NAND3_2 U276 ( .A1(a[1]), .A2(b[5]), .A3(n395), .ZN(n265) );
  VHSR_MAOI222_2 U277 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NAND4_2 U278 ( .A1(b[5]), .A2(b[4]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_IN_2 U279 ( .I(b[4]), .ZN(n308) );
  VHSR_OAI22_2 U280 ( .A1(n310), .A2(n322), .B1(n308), .B2(n326), .ZN(n228) );
  VHSR_AND2_2 U281 ( .A1(n238), .A2(n228), .Z(n230) );
  VHSR_IN_2 U282 ( .I(a[0]), .ZN(n402) );
  VHSR_OAI22_2 U283 ( .A1(n280), .A2(n400), .B1(n273), .B2(n402), .ZN(n229) );
  VHSR_NOR3_2 U284 ( .A1(n310), .A2(n400), .A3(n395), .ZN(n269) );
  VHSR_AND2_2 U285 ( .A1(n264), .A2(n258), .Z(n257) );
  VHSR_AD1_1 U286 ( .A(n230), .B(n229), .CI(n269), .CO(n250), .S(n258) );
  VHSR_NOR2_1 U287 ( .A1(n257), .A2(n250), .ZN(n253) );
  VHSR_OAI21_2 U288 ( .A1(n232), .A2(n231), .B(n233), .ZN(n254) );
  VHSR_NOR2_1 U289 ( .A1(n253), .A2(n254), .ZN(n251) );
  VHSR_CLKNAND2_2 U290 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U291 ( .A1(n241), .A2(n237), .ZN(n236) );
  VHSR_NOR3_2 U292 ( .A1(n273), .A2(n326), .A3(n236), .ZN(n295) );
  VHSR_AOI21_2 U293 ( .A1(n235), .A2(n234), .B(n296), .ZN(n299) );
  VHSR_OAI32_2 U294 ( .A1(n295), .A2(n326), .A3(n273), .B1(n236), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U295 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U296 ( .A1(n241), .A2(n240), .ZN(n306) );
  VHSR_OAI21_2 U297 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U298 ( .A1(n246), .A2(n245), .ZN(n305) );
  VHSR_AOI21_2 U299 ( .A1(n249), .A2(n248), .B(n247), .ZN(n315) );
  VHSR_CLKNAND2_2 U300 ( .A1(n257), .A2(n250), .ZN(n252) );
  VHSR_AOI22_2 U301 ( .A1(n254), .A2(n253), .B1(n252), .B2(n251), .ZN(n314) );
  VHSR_AOI21_2 U302 ( .A1(n259), .A2(n256), .B(n255), .ZN(n330) );
  VHSR_IAO21_2 U303 ( .A1(n264), .A2(n258), .B(n257), .ZN(n329) );
  VHSR_OAI31_2 U304 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n263) );
  VHSR_IN_2 U305 ( .I(n263), .ZN(n333) );
  VHSR_AOI31_2 U306 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n332) );
  VHSR_AOI22_2 U307 ( .A1(b[5]), .A2(a[0]), .B1(b[4]), .B2(a[1]), .ZN(n268) );
  VHSR_NOR2_1 U308 ( .A1(n269), .A2(n268), .ZN(n346) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[5]), .A2(b[0]), .ZN(n270) );
  VHSR_OAI32_2 U310 ( .A1(n271), .A2(n403), .A3(n309), .B1(n270), .B2(n271), 
        .ZN(n345) );
  VHSR_NOR2_1 U311 ( .A1(n401), .A2(n402), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U312 ( .A1(n381), .A2(product[0]), .ZN(n394) );
  VHSR_IN_2 U313 ( .I(n394), .ZN(n344) );
  VHSR_AOI22_2 U314 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n272) );
  VHSR_NAND4_2 U315 ( .A1(a[6]), .A2(a[7]), .A3(b[5]), .A4(b[4]), .ZN(n354) );
  VHSR_NAND3_2 U316 ( .A1(b[5]), .A2(a[5]), .A3(n381), .ZN(n312) );
  VHSR_NAND4_2 U317 ( .A1(b[6]), .A2(b[7]), .A3(a[4]), .A4(a[5]), .ZN(n352) );
  VHSR_NOR2_1 U318 ( .A1(n273), .A2(n309), .ZN(n274) );
  VHSR_AOI32_2 U319 ( .A1(b[6]), .A2(n352), .A3(a[5]), .B1(n274), .B2(n352), 
        .ZN(n277) );
  VHSR_MAOI222_2 U320 ( .A(n275), .B(n312), .C(n277), .ZN(n279) );
  VHSR_AND2_2 U321 ( .A1(n275), .A2(n312), .Z(n276) );
  VHSR_AOI21_2 U322 ( .A1(n277), .A2(n276), .B(n279), .ZN(n278) );
  VHSR_IN_2 U323 ( .I(n278), .ZN(n293) );
  VHSR_NOR2_1 U324 ( .A1(n281), .A2(n308), .ZN(n286) );
  VHSR_NOR2_1 U325 ( .A1(n280), .A2(n309), .ZN(n285) );
  VHSR_CLKNAND2_2 U326 ( .A1(b[5]), .A2(a[5]), .ZN(n287) );
  VHSR_NOR2_1 U327 ( .A1(n381), .A2(n287), .ZN(n302) );
  VHSR_MAOI222_2 U328 ( .A(n286), .B(n285), .C(n302), .ZN(n300) );
  VHSR_NOR2_1 U329 ( .A1(n293), .A2(n300), .ZN(n292) );
  VHSR_NOR2_1 U330 ( .A1(n279), .A2(n292), .ZN(n291) );
  VHSR_NOR2_1 U331 ( .A1(n281), .A2(n280), .ZN(n389) );
  VHSR_IN_2 U332 ( .I(n285), .ZN(n282) );
  VHSR_NAND3_2 U333 ( .A1(a[5]), .A2(b[7]), .A3(n282), .ZN(n283) );
  VHSR_OAI31_2 U334 ( .A1(n286), .A2(n310), .A3(n284), .B(n283), .ZN(n289) );
  VHSR_CLKNAND2_2 U335 ( .A1(a[7]), .A2(b[7]), .ZN(n390) );
  VHSR_OR2_2 U336 ( .A1(n286), .A2(n285), .Z(n301) );
  VHSR_NOR3_2 U337 ( .A1(n390), .A2(n301), .A3(n287), .ZN(n288) );
  VHSR_AOI31_2 U338 ( .A1(b[6]), .A2(a[6]), .A3(n289), .B(n288), .ZN(n348) );
  VHSR_OAI21_2 U339 ( .A1(n389), .A2(n289), .B(n348), .ZN(n290) );
  VHSR_NOR2_1 U340 ( .A1(n291), .A2(n290), .ZN(n347) );
  VHSR_AOI21_2 U341 ( .A1(n291), .A2(n290), .B(n347), .ZN(n387) );
  VHSR_AOI21_2 U342 ( .A1(n293), .A2(n300), .B(n292), .ZN(n385) );
  VHSR_AD1_1 U343 ( .A(n296), .B(n295), .CI(n294), .CO(n388), .S(n384) );
  VHSR_AD1_1 U344 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n366) );
  VHSR_OAI21_2 U345 ( .A1(n302), .A2(n301), .B(n300), .ZN(n303) );
  VHSR_IN_2 U346 ( .I(n303), .ZN(n365) );
  VHSR_AD1_1 U347 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n369) );
  VHSR_OAI22_2 U348 ( .A1(n310), .A2(n309), .B1(n308), .B2(n307), .ZN(n311) );
  VHSR_AND2_2 U349 ( .A1(n312), .A2(n311), .Z(n368) );
  VHSR_AD1_1 U350 ( .A(n315), .B(n314), .CI(n313), .CO(n304), .S(n382) );
  VHSR_NOR2_1 U351 ( .A1(n318), .A2(n322), .ZN(n337) );
  VHSR_IN_2 U352 ( .I(n337), .ZN(n327) );
  VHSR_NOR2_1 U353 ( .A1(n318), .A2(n326), .ZN(n317) );
  VHSR_OAI21_2 U354 ( .A1(n325), .A2(n322), .B(n317), .ZN(n316) );
  VHSR_OAI31_2 U355 ( .A1(n325), .A2(n317), .A3(n322), .B(n316), .ZN(n343) );
  VHSR_AOI22_2 U356 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n323) );
  VHSR_CLKNAND2_2 U357 ( .A1(b[3]), .A2(a[3]), .ZN(n336) );
  VHSR_CLKNAND2_2 U358 ( .A1(b[1]), .A2(a[1]), .ZN(n405) );
  VHSR_OAI22_2 U359 ( .A1(n327), .A2(n323), .B1(n336), .B2(n405), .ZN(n324) );
  VHSR_OAI22_2 U360 ( .A1(n325), .A2(n402), .B1(n318), .B2(n400), .ZN(n379) );
  VHSR_IN_2 U361 ( .I(n405), .ZN(n321) );
  VHSR_NOR2_1 U362 ( .A1(n401), .A2(n322), .ZN(n320) );
  VHSR_OAI211_2 U363 ( .A1(n320), .A2(n321), .B(b[2]), .C(a[0]), .ZN(n319) );
  VHSR_OAI22_2 U364 ( .A1(n403), .A2(n322), .B1(n401), .B2(n326), .ZN(n378) );
  VHSR_AOI21_2 U365 ( .A1(n323), .A2(n327), .B(n324), .ZN(n339) );
  VHSR_CLKNAND2_2 U366 ( .A1(n340), .A2(n339), .ZN(n338) );
  VHSR_CLKNAND2_2 U367 ( .A1(n343), .A2(n342), .ZN(n334) );
  VHSR_AOI211_2 U368 ( .A1(n327), .A2(n334), .B(n326), .C(n325), .ZN(n372) );
  VHSR_AD1_1 U369 ( .A(n330), .B(n329), .CI(n328), .CO(n313), .S(n371) );
  VHSR_AD1_1 U370 ( .A(n333), .B(n332), .CI(n331), .CO(n328), .S(n375) );
  VHSR_IN_2 U371 ( .I(n334), .ZN(n341) );
  VHSR_CLKNAND2_2 U372 ( .A1(n341), .A2(n336), .ZN(n335) );
  VHSR_OAI31_2 U373 ( .A1(n337), .A2(n341), .A3(n336), .B(n335), .ZN(n374) );
  VHSR_CLKNAND2_2 U374 ( .A1(a[4]), .A2(b[0]), .ZN(n396) );
  VHSR_OAI21_2 U375 ( .A1(n340), .A2(n339), .B(n338), .ZN(n399) );
  VHSR_AOI211_2 U376 ( .A1(n396), .A2(n395), .B(n344), .C(n399), .ZN(n397) );
  VHSR_IAO21_2 U377 ( .A1(n343), .A2(n342), .B(n341), .ZN(n377) );
  VHSR_AD1_1 U378 ( .A(n346), .B(n345), .CI(n344), .CO(n331), .S(n376) );
  VHSR_CLKNAND2_2 U379 ( .A1(b[6]), .A2(a[7]), .ZN(n350) );
  VHSR_AOI21_2 U380 ( .A1(a[6]), .A2(b[7]), .B(n350), .ZN(n349) );
  VHSR_AOI31_2 U381 ( .A1(a[6]), .A2(n350), .A3(b[7]), .B(n349), .ZN(n351) );
  VHSR_AND2_2 U382 ( .A1(n352), .A2(n351), .Z(n353) );
  VHSR_MAOI222_2 U383 ( .A(n354), .B(n352), .C(n351), .ZN(n360) );
  VHSR_AOI21_2 U384 ( .A1(n354), .A2(n353), .B(n360), .ZN(n359) );
  VHSR_XNOR2_2 U385 ( .A1(n358), .A2(n359), .ZN(n355) );
  VHSR_CLKNAND2_2 U386 ( .A1(n356), .A2(n355), .ZN(n391) );
  VHSR_OAI21_2 U387 ( .A1(n356), .A2(n355), .B(n391), .ZN(n357) );
  VHSR_NOR2_1 U388 ( .A1(n361), .A2(n360), .ZN(n392) );
  VHSR_NOR2_1 U389 ( .A1(n390), .A2(n363), .ZN(product[15]) );
  VHSR_AD1_1 U390 ( .A(n382), .B(n381), .CI(n380), .CO(n367), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U391 ( .A(n385), .B(n384), .CI(n383), .CO(n386), .S(product[11])
         );
  VHSR_AD1_1 U392 ( .A(n388), .B(n387), .CI(n386), .CO(n356), .S(product[12])
         );
  VHSR_NOR2_1 U393 ( .A1(n390), .A2(n389), .ZN(n393) );
  VHSR_XOR3_2 U394 ( .A1(n393), .A2(n392), .A3(n391), .Z(product[14]) );
  VHSR_AOI21_2 U395 ( .A1(n399), .A2(n398), .B(n397), .ZN(product[4]) );
  VHSR_OAI22_2 U396 ( .A1(n403), .A2(n402), .B1(n401), .B2(n400), .ZN(
        product[1]) );
  VHSR_AOI22_2 U397 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n406) );
  VHSR_AOI21_2 U398 ( .A1(n406), .A2(n405), .B(n404), .ZN(product[2]) );
endmodule


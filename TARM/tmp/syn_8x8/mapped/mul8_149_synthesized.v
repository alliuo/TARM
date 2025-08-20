
module mul8_149 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n213, n214, n215, n216, n217,
         n218, n219, n220, n221, n222, n223, n224, n225, n226, n227, n228,
         n229, n230, n231, n232, n233, n234, n235, n236, n237, n238, n239,
         n240, n241, n242, n243, n244, n245, n246, n247, n248, n249, n250,
         n251, n252, n253, n254, n255, n256, n257, n258, n259, n260, n261,
         n262, n263, n264, n265, n266, n267, n268, n269, n270, n271, n272,
         n273, n274, n275, n276, n277, n278, n279, n280, n281, n282, n283,
         n284, n285, n286, n287, n288, n289, n290, n291, n292, n293, n294,
         n295, n296, n297, n298, n299, n300, n301, n302, n303, n304, n305,
         n306, n307, n308, n309, n310, n311, n312, n313, n314, n315, n316,
         n317, n318, n319, n320, n321, n322, n323, n324, n325, n326, n327,
         n328, n329, n330, n331, n332, n333, n334, n335, n336, n337, n338,
         n339, n340, n341, n342, n343, n344, n345, n346, n347, n348, n349,
         n350, n351, n352, n353, n354, n355, n356, n357, n358, n359, n360,
         n361, n362, n363, n364, n365, n366, n367, n368, n369, n370, n371,
         n372, n373, n374, n375, n376, n377, n378, n379, n380, n381, n382,
         n383, n384, n385, n386, n387, n388, n389, n390, n391, n392, n393,
         n394, n395, n396, n397, n398, n399, n400, n401, n402, n403, n404;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U206 ( .A1(n239), .B1(n223), .ZN(n224) );
  VHSR_NOR2_1 U207 ( .A1(n343), .A2(n317), .ZN(n262) );
  VHSR_INOR2_2 U208 ( .A1(n227), .B1(n251), .ZN(n245) );
  VHSR_INOR2_2 U209 ( .A1(n347), .B1(n346), .ZN(n359) );
  VHSR_INOR2_2 U210 ( .A1(n280), .B1(n290), .ZN(n289) );
  VHSR_NOR2_1 U211 ( .A1(n309), .A2(n343), .ZN(n374) );
  VHSR_IN_2 U212 ( .I(n357), .ZN(product[13]) );
  VHSR_NOR2_2 U213 ( .A1(n232), .A2(n231), .ZN(n293) );
  VHSR_INOR2_1 U214 ( .A1(n361), .B1(n360), .ZN(n392) );
  VHSR_INOR2_1 U215 ( .A1(n229), .B1(n243), .ZN(n240) );
  VHSR_INOR2_1 U216 ( .A1(n220), .B1(n247), .ZN(n235) );
  VHSR_AD1_1 U217 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(product[6])
         );
  VHSR_AD1_1 U218 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U219 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(product[10])
         );
  VHSR_AD1_1 U220 ( .A(n385), .B(n401), .CI(n384), .CO(n342), .S(product[3])
         );
  VHSR_AD1_1 U221 ( .A(n383), .B(n382), .CI(n394), .CO(n379), .S(product[5])
         );
  VHSR_AD1_1 U222 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U223 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(product[9])
         );
  VHSR_AD1_1 U224 ( .A(n366), .B(n365), .CI(n364), .CO(n386), .S(product[11])
         );
  VHSR_IN_2 U225 ( .I(b[7]), .ZN(n270) );
  VHSR_IN_2 U226 ( .I(a[3]), .ZN(n321) );
  VHSR_IN_2 U227 ( .I(b[6]), .ZN(n274) );
  VHSR_IN_2 U228 ( .I(a[2]), .ZN(n319) );
  VHSR_OAI22_2 U229 ( .A1(n274), .A2(n321), .B1(n270), .B2(n319), .ZN(n237) );
  VHSR_NOR2_1 U230 ( .A1(n270), .A2(n319), .ZN(n214) );
  VHSR_IN_2 U231 ( .I(a[1]), .ZN(n397) );
  VHSR_NOR2_1 U232 ( .A1(n274), .A2(n397), .ZN(n213) );
  VHSR_IN_2 U233 ( .I(b[5]), .ZN(n305) );
  VHSR_AOI211_2 U234 ( .A1(b[4]), .A2(a[2]), .B(n305), .C(n321), .ZN(n219) );
  VHSR_OAI22_2 U235 ( .A1(n274), .A2(n319), .B1(n270), .B2(n397), .ZN(n218) );
  VHSR_AOI22_2 U236 ( .A1(n214), .A2(n213), .B1(n219), .B2(n218), .ZN(n220) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[4]), .A2(a[2]), .ZN(n258) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[6]), .A2(a[0]), .ZN(n257) );
  VHSR_IN_2 U239 ( .I(b[4]), .ZN(n309) );
  VHSR_IN_2 U240 ( .I(a[0]), .ZN(n399) );
  VHSR_OAI211_2 U241 ( .A1(n309), .A2(n399), .B(b[5]), .C(a[1]), .ZN(n256) );
  VHSR_MAOI222_2 U242 ( .A(n258), .B(n257), .C(n256), .ZN(n255) );
  VHSR_NAND4_2 U243 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n234) );
  VHSR_OAI22_2 U244 ( .A1(n309), .A2(n321), .B1(n305), .B2(n319), .ZN(n215) );
  VHSR_AND2_2 U245 ( .A1(n234), .A2(n215), .Z(n217) );
  VHSR_OAI22_2 U246 ( .A1(n274), .A2(n397), .B1(n270), .B2(n399), .ZN(n216) );
  VHSR_NOR4_2 U247 ( .A1(n309), .A2(n305), .A3(n397), .A4(n399), .ZN(n267) );
  VHSR_AND2_2 U248 ( .A1(n255), .A2(n254), .Z(n253) );
  VHSR_AD1_1 U249 ( .A(n217), .B(n216), .CI(n267), .CO(n246), .S(n254) );
  VHSR_NOR2_1 U250 ( .A1(n253), .A2(n246), .ZN(n249) );
  VHSR_OAI21_2 U251 ( .A1(n219), .A2(n218), .B(n220), .ZN(n250) );
  VHSR_NOR2_1 U252 ( .A1(n249), .A2(n250), .ZN(n247) );
  VHSR_CLKNAND2_2 U253 ( .A1(n235), .A2(n234), .ZN(n233) );
  VHSR_CLKNAND2_2 U254 ( .A1(n237), .A2(n233), .ZN(n230) );
  VHSR_NOR3_2 U255 ( .A1(n270), .A2(n321), .A3(n230), .ZN(n294) );
  VHSR_CLKNAND2_2 U256 ( .A1(a[7]), .A2(b[3]), .ZN(n232) );
  VHSR_IN_2 U257 ( .I(a[6]), .ZN(n273) );
  VHSR_IN_2 U258 ( .I(b[3]), .ZN(n318) );
  VHSR_IN_2 U259 ( .I(a[7]), .ZN(n268) );
  VHSR_IN_2 U260 ( .I(b[2]), .ZN(n317) );
  VHSR_OAI22_2 U261 ( .A1(n273), .A2(n318), .B1(n268), .B2(n317), .ZN(n242) );
  VHSR_IN_2 U262 ( .I(a[4]), .ZN(n343) );
  VHSR_CLKNAND2_2 U263 ( .A1(a[5]), .A2(b[3]), .ZN(n221) );
  VHSR_IN_2 U264 ( .I(b[1]), .ZN(n400) );
  VHSR_OAI22_2 U265 ( .A1(n262), .A2(n221), .B1(n268), .B2(n400), .ZN(n228) );
  VHSR_IN_2 U266 ( .I(a[5]), .ZN(n307) );
  VHSR_NOR4_2 U267 ( .A1(n262), .A2(n232), .A3(n307), .A4(n400), .ZN(n222) );
  VHSR_AOI31_2 U268 ( .A1(b[2]), .A2(a[6]), .A3(n228), .B(n222), .ZN(n229) );
  VHSR_IN_2 U269 ( .I(b[0]), .ZN(n398) );
  VHSR_NOR4_2 U270 ( .A1(n343), .A2(n307), .A3(n400), .A4(n398), .ZN(n265) );
  VHSR_NAND3_2 U271 ( .A1(a[5]), .A2(b[3]), .A3(n262), .ZN(n239) );
  VHSR_AOI22_2 U272 ( .A1(a[4]), .A2(b[3]), .B1(a[5]), .B2(b[2]), .ZN(n223) );
  VHSR_OAI22_2 U273 ( .A1(n273), .A2(n400), .B1(n268), .B2(n398), .ZN(n225) );
  VHSR_MAOI222_2 U274 ( .A(n265), .B(n224), .C(n225), .ZN(n227) );
  VHSR_NOR2_1 U275 ( .A1(n273), .A2(n398), .ZN(n261) );
  VHSR_AOI211_2 U276 ( .A1(a[4]), .A2(b[0]), .B(n307), .C(n400), .ZN(n260) );
  VHSR_MAOI222_2 U277 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_OR2_2 U278 ( .A1(n265), .A2(n224), .Z(n226) );
  VHSR_OAI21_2 U279 ( .A1(n226), .A2(n225), .B(n227), .ZN(n252) );
  VHSR_NOR2_1 U280 ( .A1(n259), .A2(n252), .ZN(n251) );
  VHSR_AOI32_2 U281 ( .A1(b[2]), .A2(n229), .A3(a[6]), .B1(n228), .B2(n229), 
        .ZN(n244) );
  VHSR_NOR2_1 U282 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_CLKNAND2_2 U283 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U284 ( .A1(n242), .A2(n238), .ZN(n231) );
  VHSR_OAI32_2 U285 ( .A1(n294), .A2(n321), .A3(n270), .B1(n230), .B2(n294), 
        .ZN(n301) );
  VHSR_AOI21_2 U286 ( .A1(n232), .A2(n231), .B(n293), .ZN(n300) );
  VHSR_OAI21_2 U287 ( .A1(n235), .A2(n234), .B(n233), .ZN(n236) );
  VHSR_XNOR2_2 U288 ( .A1(n237), .A2(n236), .ZN(n304) );
  VHSR_OAI21_2 U289 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U290 ( .A1(n242), .A2(n241), .ZN(n303) );
  VHSR_AOI21_2 U291 ( .A1(n245), .A2(n244), .B(n243), .ZN(n312) );
  VHSR_CLKNAND2_2 U292 ( .A1(n253), .A2(n246), .ZN(n248) );
  VHSR_AOI22_2 U293 ( .A1(n250), .A2(n249), .B1(n248), .B2(n247), .ZN(n311) );
  VHSR_AOI21_2 U294 ( .A1(n259), .A2(n252), .B(n251), .ZN(n327) );
  VHSR_IAO21_2 U295 ( .A1(n255), .A2(n254), .B(n253), .ZN(n326) );
  VHSR_AOI31_2 U296 ( .A1(n258), .A2(n257), .A3(n256), .B(n255), .ZN(n333) );
  VHSR_OAI31_2 U297 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n263) );
  VHSR_IN_2 U298 ( .I(n263), .ZN(n332) );
  VHSR_CLKNAND2_2 U299 ( .A1(a[5]), .A2(b[0]), .ZN(n264) );
  VHSR_OAI32_2 U300 ( .A1(n265), .A2(n400), .A3(n343), .B1(n264), .B2(n265), 
        .ZN(n336) );
  VHSR_NOR2_1 U301 ( .A1(n398), .A2(n399), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U302 ( .A1(n374), .A2(product[0]), .ZN(n345) );
  VHSR_IN_2 U303 ( .I(n345), .ZN(n335) );
  VHSR_CLKNAND2_2 U304 ( .A1(b[5]), .A2(a[0]), .ZN(n266) );
  VHSR_OAI32_2 U305 ( .A1(n267), .A2(n397), .A3(n309), .B1(n266), .B2(n267), 
        .ZN(n334) );
  VHSR_AOI22_2 U306 ( .A1(b[4]), .A2(a[7]), .B1(a[6]), .B2(b[5]), .ZN(n269) );
  VHSR_NOR4_2 U307 ( .A1(n309), .A2(n273), .A3(n305), .A4(n268), .ZN(n354) );
  VHSR_NOR2_1 U308 ( .A1(n269), .A2(n354), .ZN(n276) );
  VHSR_NAND3_2 U309 ( .A1(b[5]), .A2(a[5]), .A3(n374), .ZN(n279) );
  VHSR_IN_2 U310 ( .I(n279), .ZN(n272) );
  VHSR_AOI22_2 U311 ( .A1(a[4]), .A2(b[7]), .B1(b[6]), .B2(a[5]), .ZN(n271) );
  VHSR_NOR4_2 U312 ( .A1(n343), .A2(n274), .A3(n307), .A4(n270), .ZN(n352) );
  VHSR_NOR2_1 U313 ( .A1(n271), .A2(n352), .ZN(n275) );
  VHSR_MAOI222_2 U314 ( .A(n276), .B(n272), .C(n275), .ZN(n280) );
  VHSR_NOR2_1 U315 ( .A1(n309), .A2(n273), .ZN(n282) );
  VHSR_NOR2_1 U316 ( .A1(n343), .A2(n274), .ZN(n284) );
  VHSR_CLKNAND2_2 U317 ( .A1(b[5]), .A2(a[5]), .ZN(n285) );
  VHSR_NOR2_1 U318 ( .A1(n374), .A2(n285), .ZN(n297) );
  VHSR_OR2_2 U319 ( .A1(n282), .A2(n284), .Z(n296) );
  VHSR_AOI22_2 U320 ( .A1(n282), .A2(n284), .B1(n297), .B2(n296), .ZN(n295) );
  VHSR_NOR2_1 U321 ( .A1(n276), .A2(n275), .ZN(n278) );
  VHSR_AOI22_2 U322 ( .A1(n276), .A2(n275), .B1(n279), .B2(n278), .ZN(n277) );
  VHSR_OAI21_2 U323 ( .A1(n279), .A2(n278), .B(n277), .ZN(n291) );
  VHSR_NOR2_1 U324 ( .A1(n295), .A2(n291), .ZN(n290) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[6]), .A2(b[6]), .ZN(n362) );
  VHSR_IN_2 U326 ( .I(n362), .ZN(n389) );
  VHSR_CLKNAND2_2 U327 ( .A1(a[5]), .A2(b[7]), .ZN(n283) );
  VHSR_CLKNAND2_2 U328 ( .A1(b[5]), .A2(a[7]), .ZN(n281) );
  VHSR_OAI22_2 U329 ( .A1(n284), .A2(n283), .B1(n282), .B2(n281), .ZN(n287) );
  VHSR_CLKNAND2_2 U330 ( .A1(a[7]), .A2(b[7]), .ZN(n390) );
  VHSR_NOR3_2 U331 ( .A1(n296), .A2(n285), .A3(n390), .ZN(n286) );
  VHSR_AOI31_2 U332 ( .A1(b[6]), .A2(a[6]), .A3(n287), .B(n286), .ZN(n347) );
  VHSR_OAI21_2 U333 ( .A1(n389), .A2(n287), .B(n347), .ZN(n288) );
  VHSR_NOR2_1 U334 ( .A1(n289), .A2(n288), .ZN(n346) );
  VHSR_AOI21_2 U335 ( .A1(n289), .A2(n288), .B(n346), .ZN(n387) );
  VHSR_AOI21_2 U336 ( .A1(n295), .A2(n291), .B(n290), .ZN(n366) );
  VHSR_AD1_1 U337 ( .A(n294), .B(n293), .CI(n292), .CO(n388), .S(n365) );
  VHSR_OAI21_2 U338 ( .A1(n297), .A2(n296), .B(n295), .ZN(n298) );
  VHSR_IN_2 U339 ( .I(n298), .ZN(n369) );
  VHSR_AD1_1 U340 ( .A(n301), .B(n300), .CI(n299), .CO(n292), .S(n368) );
  VHSR_AD1_1 U341 ( .A(n304), .B(n303), .CI(n302), .CO(n299), .S(n372) );
  VHSR_NOR2_1 U342 ( .A1(n305), .A2(n343), .ZN(n308) );
  VHSR_OAI21_2 U343 ( .A1(n309), .A2(n307), .B(n308), .ZN(n306) );
  VHSR_OAI31_2 U344 ( .A1(n309), .A2(n308), .A3(n307), .B(n306), .ZN(n371) );
  VHSR_AD1_1 U345 ( .A(n312), .B(n311), .CI(n310), .CO(n302), .S(n375) );
  VHSR_NOR2_1 U346 ( .A1(n317), .A2(n319), .ZN(n330) );
  VHSR_NOR2_1 U347 ( .A1(n317), .A2(n321), .ZN(n314) );
  VHSR_OAI21_2 U348 ( .A1(n318), .A2(n319), .B(n314), .ZN(n313) );
  VHSR_OAI31_2 U349 ( .A1(n318), .A2(n314), .A3(n319), .B(n313), .ZN(n339) );
  VHSR_NOR2_1 U350 ( .A1(n400), .A2(n321), .ZN(n316) );
  VHSR_NOR2_1 U351 ( .A1(n318), .A2(n397), .ZN(n315) );
  VHSR_MAOI222_2 U352 ( .A(n330), .B(n316), .C(n315), .ZN(n323) );
  VHSR_OAI22_2 U353 ( .A1(n318), .A2(n399), .B1(n317), .B2(n397), .ZN(n385) );
  VHSR_CLKNAND2_2 U354 ( .A1(b[0]), .A2(a[2]), .ZN(n404) );
  VHSR_CLKNAND2_2 U355 ( .A1(b[2]), .A2(a[0]), .ZN(n403) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[1]), .A2(a[1]), .ZN(n402) );
  VHSR_MAOI222_2 U357 ( .A(n404), .B(n403), .C(n402), .ZN(n401) );
  VHSR_OAI22_2 U358 ( .A1(n400), .A2(n319), .B1(n398), .B2(n321), .ZN(n384) );
  VHSR_IN_2 U359 ( .I(n323), .ZN(n322) );
  VHSR_AOI21_2 U360 ( .A1(a[1]), .A2(b[3]), .B(n330), .ZN(n320) );
  VHSR_OAI32_2 U361 ( .A1(n322), .A2(n321), .A3(n400), .B1(n320), .B2(n322), 
        .ZN(n341) );
  VHSR_CLKNAND2_2 U362 ( .A1(n342), .A2(n341), .ZN(n340) );
  VHSR_CLKNAND2_2 U363 ( .A1(n323), .A2(n340), .ZN(n338) );
  VHSR_AND2_2 U364 ( .A1(n339), .A2(n338), .Z(n337) );
  VHSR_OAI211_2 U365 ( .A1(n330), .A2(n337), .B(a[3]), .C(b[3]), .ZN(n324) );
  VHSR_IN_2 U366 ( .I(n324), .ZN(n378) );
  VHSR_AD1_1 U367 ( .A(n327), .B(n326), .CI(n325), .CO(n310), .S(n377) );
  VHSR_CLKNAND2_2 U368 ( .A1(b[3]), .A2(a[3]), .ZN(n329) );
  VHSR_CLKNAND2_2 U369 ( .A1(n337), .A2(n329), .ZN(n328) );
  VHSR_OAI31_2 U370 ( .A1(n330), .A2(n337), .A3(n329), .B(n328), .ZN(n381) );
  VHSR_AD1_1 U371 ( .A(n333), .B(n332), .CI(n331), .CO(n325), .S(n380) );
  VHSR_AD1_1 U372 ( .A(n336), .B(n335), .CI(n334), .CO(n331), .S(n383) );
  VHSR_IAO21_2 U373 ( .A1(n339), .A2(n338), .B(n337), .ZN(n382) );
  VHSR_OAI21_2 U374 ( .A1(n342), .A2(n341), .B(n340), .ZN(n396) );
  VHSR_NOR2_1 U375 ( .A1(n343), .A2(n398), .ZN(n344) );
  VHSR_AOI32_2 U376 ( .A1(b[4]), .A2(n345), .A3(a[0]), .B1(n344), .B2(n345), 
        .ZN(n395) );
  VHSR_NOR2_1 U377 ( .A1(n396), .A2(n395), .ZN(n394) );
  VHSR_CLKNAND2_2 U378 ( .A1(a[7]), .A2(b[6]), .ZN(n349) );
  VHSR_AOI21_2 U379 ( .A1(a[6]), .A2(b[7]), .B(n349), .ZN(n348) );
  VHSR_AOI31_2 U380 ( .A1(a[6]), .A2(n349), .A3(b[7]), .B(n348), .ZN(n350) );
  VHSR_IN_2 U381 ( .I(n350), .ZN(n351) );
  VHSR_OR2_2 U382 ( .A1(n352), .A2(n351), .Z(n353) );
  VHSR_MAOI222_2 U383 ( .A(n354), .B(n352), .C(n351), .ZN(n361) );
  VHSR_OAI21_2 U384 ( .A1(n354), .A2(n353), .B(n361), .ZN(n358) );
  VHSR_CLKXOR2_2 U385 ( .A1(n359), .A2(n358), .Z(n355) );
  VHSR_CLKNAND2_2 U386 ( .A1(n356), .A2(n355), .ZN(n391) );
  VHSR_OAI21_2 U387 ( .A1(n356), .A2(n355), .B(n391), .ZN(n357) );
  VHSR_NOR2_1 U388 ( .A1(n359), .A2(n358), .ZN(n360) );
  VHSR_AND3_2 U389 ( .A1(n362), .A2(n392), .A3(n391), .Z(n363) );
  VHSR_NOR2_1 U390 ( .A1(n390), .A2(n363), .ZN(product[15]) );
  VHSR_AD1_1 U391 ( .A(n388), .B(n387), .CI(n386), .CO(n356), .S(product[12])
         );
  VHSR_NOR2_1 U392 ( .A1(n390), .A2(n389), .ZN(n393) );
  VHSR_XOR3_2 U393 ( .A1(n393), .A2(n392), .A3(n391), .Z(product[14]) );
  VHSR_AOI21_2 U394 ( .A1(n396), .A2(n395), .B(n394), .ZN(product[4]) );
  VHSR_OAI22_2 U395 ( .A1(n400), .A2(n399), .B1(n398), .B2(n397), .ZN(
        product[1]) );
  VHSR_AOI31_2 U396 ( .A1(n404), .A2(n403), .A3(n402), .B(n401), .ZN(
        product[2]) );
endmodule


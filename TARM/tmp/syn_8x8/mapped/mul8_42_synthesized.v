
module mul8_42 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n216, n217,
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
         n394, n395, n396, n397, n398, n399, n400, n401, n402;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U206 ( .A1(n266), .B1(a[5]), .B2(b[3]), .ZN(n216) );
  VHSR_NOR2_1 U207 ( .A1(n218), .A2(n282), .ZN(n266) );
  VHSR_INOR2_2 U208 ( .A1(n226), .B1(n252), .ZN(n245) );
  VHSR_NOR2_1 U209 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_NOR2_1 U210 ( .A1(n342), .A2(n341), .ZN(n354) );
  VHSR_NOR2_1 U211 ( .A1(n290), .A2(n291), .ZN(n342) );
  VHSR_IOA21_2 U212 ( .A1(n391), .A2(n390), .B(n389), .ZN(n393) );
  VHSR_INOR2_2 U213 ( .A1(n356), .B1(n355), .ZN(n387) );
  VHSR_IN_2 U214 ( .I(n352), .ZN(product[13]) );
  VHSR_NOR2_2 U215 ( .A1(n236), .A2(n235), .ZN(n296) );
  VHSR_INOR2_1 U216 ( .A1(n224), .B1(n255), .ZN(n254) );
  VHSR_NOR2_2 U217 ( .A1(n292), .A2(n288), .ZN(n290) );
  VHSR_NOR2_2 U218 ( .A1(n272), .A2(n281), .ZN(n384) );
  VHSR_MOAI22_1 U219 ( .A1(n277), .A2(n315), .B1(b[4]), .B2(a[3]), .ZN(n228)
         );
  VHSR_AD1_1 U220 ( .A(n364), .B(n363), .CI(n362), .CO(n359), .S(product[9])
         );
  VHSR_AD1_1 U221 ( .A(n371), .B(n399), .CI(n370), .CO(n334), .S(product[3])
         );
  VHSR_AD1_1 U222 ( .A(n392), .B(n369), .CI(n368), .CO(n372), .S(product[5])
         );
  VHSR_AD1_1 U223 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n361), .B(n360), .CI(n359), .CO(n378), .S(product[10])
         );
  VHSR_CLKNAND2_2 U225 ( .A1(b[3]), .A2(a[7]), .ZN(n236) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n312) );
  VHSR_IN_2 U227 ( .I(a[6]), .ZN(n272) );
  VHSR_IN_2 U228 ( .I(a[7]), .ZN(n278) );
  VHSR_IN_2 U229 ( .I(b[2]), .ZN(n218) );
  VHSR_OAI22_2 U230 ( .A1(n312), .A2(n272), .B1(n278), .B2(n218), .ZN(n247) );
  VHSR_IN_2 U231 ( .I(b[1]), .ZN(n398) );
  VHSR_IN_2 U232 ( .I(a[4]), .ZN(n282) );
  VHSR_OAI21_2 U233 ( .A1(n398), .A2(n278), .B(n216), .ZN(n225) );
  VHSR_IN_2 U234 ( .I(a[5]), .ZN(n283) );
  VHSR_NOR4_2 U235 ( .A1(n266), .A2(n283), .A3(n236), .A4(n398), .ZN(n217) );
  VHSR_AOI31_2 U236 ( .A1(b[2]), .A2(a[6]), .A3(n225), .B(n217), .ZN(n226) );
  VHSR_IN_2 U237 ( .I(b[0]), .ZN(n396) );
  VHSR_NOR4_2 U238 ( .A1(n283), .A2(n282), .A3(n398), .A4(n396), .ZN(n271) );
  VHSR_NAND4_2 U239 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n244) );
  VHSR_NOR2_1 U240 ( .A1(n218), .A2(n283), .ZN(n219) );
  VHSR_AOI32_2 U241 ( .A1(b[3]), .A2(n244), .A3(a[4]), .B1(n219), .B2(n244), 
        .ZN(n220) );
  VHSR_IN_2 U242 ( .I(n220), .ZN(n221) );
  VHSR_OAI22_2 U243 ( .A1(n278), .A2(n396), .B1(n272), .B2(n398), .ZN(n222) );
  VHSR_MAOI222_2 U244 ( .A(n271), .B(n221), .C(n222), .ZN(n224) );
  VHSR_NOR2_1 U245 ( .A1(n272), .A2(n396), .ZN(n265) );
  VHSR_AOI211_2 U246 ( .A1(a[4]), .A2(b[0]), .B(n283), .C(n398), .ZN(n264) );
  VHSR_MAOI222_2 U247 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_OR2_2 U248 ( .A1(n271), .A2(n221), .Z(n223) );
  VHSR_OAI21_2 U249 ( .A1(n223), .A2(n222), .B(n224), .ZN(n256) );
  VHSR_NOR2_1 U250 ( .A1(n263), .A2(n256), .ZN(n255) );
  VHSR_AOI32_2 U251 ( .A1(b[2]), .A2(n226), .A3(a[6]), .B1(n225), .B2(n226), 
        .ZN(n253) );
  VHSR_CLKNAND2_2 U252 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_CLKNAND2_2 U253 ( .A1(n247), .A2(n243), .ZN(n235) );
  VHSR_IN_2 U254 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U255 ( .I(a[3]), .ZN(n320) );
  VHSR_IN_2 U256 ( .I(b[6]), .ZN(n281) );
  VHSR_IN_2 U257 ( .I(a[2]), .ZN(n315) );
  VHSR_OAI22_2 U258 ( .A1(n281), .A2(n320), .B1(n280), .B2(n315), .ZN(n242) );
  VHSR_AOI22_2 U259 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n233) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[4]), .A2(a[2]), .ZN(n262) );
  VHSR_NAND3_2 U261 ( .A1(a[3]), .A2(b[5]), .A3(n262), .ZN(n232) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[7]), .A2(a[2]), .ZN(n227) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[6]), .A2(a[1]), .ZN(n229) );
  VHSR_OAI22_2 U264 ( .A1(n233), .A2(n232), .B1(n227), .B2(n229), .ZN(n234) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[0]), .ZN(n390) );
  VHSR_NAND3_2 U266 ( .A1(a[1]), .A2(b[5]), .A3(n390), .ZN(n261) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[0]), .ZN(n260) );
  VHSR_MAOI222_2 U268 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_NAND4_2 U269 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_IN_2 U270 ( .I(b[5]), .ZN(n277) );
  VHSR_AND2_2 U271 ( .A1(n239), .A2(n228), .Z(n231) );
  VHSR_IN_2 U272 ( .I(a[0]), .ZN(n397) );
  VHSR_OAI21_2 U273 ( .A1(n280), .A2(n397), .B(n229), .ZN(n230) );
  VHSR_IN_2 U274 ( .I(a[1]), .ZN(n395) );
  VHSR_NOR3_2 U275 ( .A1(n277), .A2(n395), .A3(n390), .ZN(n269) );
  VHSR_AND2_2 U276 ( .A1(n259), .A2(n258), .Z(n257) );
  VHSR_AD1_1 U277 ( .A(n231), .B(n230), .CI(n269), .CO(n248), .S(n258) );
  VHSR_AOI21_2 U278 ( .A1(n233), .A2(n232), .B(n234), .ZN(n251) );
  VHSR_OAI32_2 U279 ( .A1(n234), .A2(n257), .A3(n248), .B1(n251), .B2(n234), 
        .ZN(n240) );
  VHSR_CLKNAND2_2 U280 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U281 ( .A1(n242), .A2(n238), .ZN(n237) );
  VHSR_NOR3_2 U282 ( .A1(n280), .A2(n320), .A3(n237), .ZN(n295) );
  VHSR_AOI21_2 U283 ( .A1(n236), .A2(n235), .B(n296), .ZN(n299) );
  VHSR_OAI32_2 U284 ( .A1(n295), .A2(n320), .A3(n280), .B1(n237), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U285 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U286 ( .A1(n242), .A2(n241), .ZN(n306) );
  VHSR_OAI21_2 U287 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U288 ( .A1(n247), .A2(n246), .ZN(n305) );
  VHSR_NOR2_1 U289 ( .A1(n257), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U290 ( .A1(n257), .A2(n248), .B1(n251), .B2(n250), .ZN(n249) );
  VHSR_OAI21_2 U291 ( .A1(n251), .A2(n250), .B(n249), .ZN(n311) );
  VHSR_AOI21_2 U292 ( .A1(n254), .A2(n253), .B(n252), .ZN(n310) );
  VHSR_AOI21_2 U293 ( .A1(n263), .A2(n256), .B(n255), .ZN(n325) );
  VHSR_IAO21_2 U294 ( .A1(n259), .A2(n258), .B(n257), .ZN(n324) );
  VHSR_AOI31_2 U295 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n332) );
  VHSR_OAI31_2 U296 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n267) );
  VHSR_IN_2 U297 ( .I(n267), .ZN(n331) );
  VHSR_AOI22_2 U298 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n268) );
  VHSR_NOR2_1 U299 ( .A1(n269), .A2(n268), .ZN(n337) );
  VHSR_CLKNAND2_2 U300 ( .A1(a[4]), .A2(b[4]), .ZN(n285) );
  VHSR_IN_2 U301 ( .I(n285), .ZN(n366) );
  VHSR_NOR2_1 U302 ( .A1(n396), .A2(n397), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U303 ( .A1(n366), .A2(product[0]), .ZN(n389) );
  VHSR_IN_2 U304 ( .I(n389), .ZN(n336) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[1]), .ZN(n270) );
  VHSR_OAI32_2 U306 ( .A1(n271), .A2(n396), .A3(n283), .B1(n270), .B2(n271), 
        .ZN(n335) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_NAND3_2 U308 ( .A1(a[7]), .A2(b[5]), .A3(n303), .ZN(n274) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[4]), .A2(b[6]), .ZN(n302) );
  VHSR_NAND3_2 U310 ( .A1(b[7]), .A2(a[5]), .A3(n302), .ZN(n273) );
  VHSR_CLKNAND2_2 U311 ( .A1(n274), .A2(n273), .ZN(n276) );
  VHSR_IN_2 U312 ( .I(n384), .ZN(n357) );
  VHSR_MAOI222_2 U313 ( .A(n357), .B(n274), .C(n273), .ZN(n341) );
  VHSR_IN_2 U314 ( .I(n341), .ZN(n275) );
  VHSR_OAI21_2 U315 ( .A1(n384), .A2(n276), .B(n275), .ZN(n291) );
  VHSR_NOR3_2 U316 ( .A1(n283), .A2(n277), .A3(n285), .ZN(n307) );
  VHSR_NOR3_2 U317 ( .A1(n278), .A2(n303), .A3(n277), .ZN(n349) );
  VHSR_AOI22_2 U318 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n279) );
  VHSR_NOR2_1 U319 ( .A1(n349), .A2(n279), .ZN(n287) );
  VHSR_NOR4_2 U320 ( .A1(n283), .A2(n282), .A3(n281), .A4(n280), .ZN(n347) );
  VHSR_AOI22_2 U321 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n284) );
  VHSR_NOR2_1 U322 ( .A1(n347), .A2(n284), .ZN(n286) );
  VHSR_NAND3_2 U323 ( .A1(b[5]), .A2(a[5]), .A3(n285), .ZN(n301) );
  VHSR_MAOI222_2 U324 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_AND2_2 U325 ( .A1(n293), .A2(n300), .Z(n292) );
  VHSR_AD1_1 U326 ( .A(n307), .B(n287), .CI(n286), .CO(n288), .S(n293) );
  VHSR_CLKNAND2_2 U327 ( .A1(n292), .A2(n288), .ZN(n289) );
  VHSR_AOI22_2 U328 ( .A1(n291), .A2(n290), .B1(n289), .B2(n342), .ZN(n382) );
  VHSR_IAO21_2 U329 ( .A1(n293), .A2(n300), .B(n292), .ZN(n380) );
  VHSR_AD1_1 U330 ( .A(n296), .B(n295), .CI(n294), .CO(n383), .S(n379) );
  VHSR_AD1_1 U331 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n361) );
  VHSR_AOI31_2 U332 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n360) );
  VHSR_AD1_1 U333 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n364) );
  VHSR_AOI22_2 U334 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n308) );
  VHSR_NOR2_1 U335 ( .A1(n308), .A2(n307), .ZN(n363) );
  VHSR_AD1_1 U336 ( .A(n311), .B(n310), .CI(n309), .CO(n304), .S(n367) );
  VHSR_CLKNAND2_2 U337 ( .A1(b[2]), .A2(a[2]), .ZN(n317) );
  VHSR_IN_2 U338 ( .I(n317), .ZN(n329) );
  VHSR_CLKNAND2_2 U339 ( .A1(b[2]), .A2(a[0]), .ZN(n402) );
  VHSR_NOR3_2 U340 ( .A1(n312), .A2(n395), .A3(n402), .ZN(n340) );
  VHSR_CLKNAND2_2 U341 ( .A1(b[3]), .A2(a[3]), .ZN(n327) );
  VHSR_AOI22_2 U342 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n313) );
  VHSR_IAO21_2 U343 ( .A1(n327), .A2(n317), .B(n313), .ZN(n339) );
  VHSR_AOI22_2 U344 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n314) );
  VHSR_NOR2_1 U345 ( .A1(n314), .A2(n340), .ZN(n371) );
  VHSR_CLKNAND2_2 U346 ( .A1(b[1]), .A2(a[1]), .ZN(n401) );
  VHSR_CLKNAND2_2 U347 ( .A1(b[0]), .A2(a[2]), .ZN(n400) );
  VHSR_MAOI222_2 U348 ( .A(n402), .B(n401), .C(n400), .ZN(n399) );
  VHSR_OAI22_2 U349 ( .A1(n398), .A2(n315), .B1(n396), .B2(n320), .ZN(n370) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[1]), .A2(a[3]), .ZN(n318) );
  VHSR_NAND3_2 U351 ( .A1(a[1]), .A2(b[3]), .A3(n402), .ZN(n316) );
  VHSR_MAOI222_2 U352 ( .A(n318), .B(n317), .C(n316), .ZN(n321) );
  VHSR_AOI31_2 U353 ( .A1(a[1]), .A2(b[3]), .A3(n402), .B(n329), .ZN(n319) );
  VHSR_OAI32_2 U354 ( .A1(n321), .A2(n320), .A3(n398), .B1(n319), .B2(n321), 
        .ZN(n333) );
  VHSR_AOI21_2 U355 ( .A1(n334), .A2(n333), .B(n321), .ZN(n322) );
  VHSR_IN_2 U356 ( .I(n322), .ZN(n338) );
  VHSR_IAO21_2 U357 ( .A1(n329), .A2(n328), .B(n327), .ZN(n377) );
  VHSR_AD1_1 U358 ( .A(n325), .B(n324), .CI(n323), .CO(n309), .S(n376) );
  VHSR_OAI21_2 U359 ( .A1(n329), .A2(n327), .B(n328), .ZN(n326) );
  VHSR_OAI31_2 U360 ( .A1(n329), .A2(n328), .A3(n327), .B(n326), .ZN(n374) );
  VHSR_AD1_1 U361 ( .A(n332), .B(n331), .CI(n330), .CO(n323), .S(n373) );
  VHSR_CLKNAND2_2 U362 ( .A1(a[4]), .A2(b[0]), .ZN(n391) );
  VHSR_XNOR2_2 U363 ( .A1(n334), .A2(n333), .ZN(n394) );
  VHSR_AOI211_2 U364 ( .A1(n391), .A2(n390), .B(n336), .C(n394), .ZN(n392) );
  VHSR_AD1_1 U365 ( .A(n337), .B(n336), .CI(n335), .CO(n330), .S(n369) );
  VHSR_AD1_1 U366 ( .A(n340), .B(n339), .CI(n338), .CO(n328), .S(n368) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[6]), .A2(b[7]), .ZN(n344) );
  VHSR_AOI21_2 U368 ( .A1(a[7]), .A2(b[6]), .B(n344), .ZN(n343) );
  VHSR_AOI31_2 U369 ( .A1(a[7]), .A2(n344), .A3(b[6]), .B(n343), .ZN(n345) );
  VHSR_IN_2 U370 ( .I(n345), .ZN(n346) );
  VHSR_OR2_2 U371 ( .A1(n347), .A2(n346), .Z(n348) );
  VHSR_MAOI222_2 U372 ( .A(n349), .B(n347), .C(n346), .ZN(n356) );
  VHSR_OAI21_2 U373 ( .A1(n349), .A2(n348), .B(n356), .ZN(n353) );
  VHSR_CLKXOR2_2 U374 ( .A1(n354), .A2(n353), .Z(n350) );
  VHSR_CLKNAND2_2 U375 ( .A1(n351), .A2(n350), .ZN(n386) );
  VHSR_OAI21_2 U376 ( .A1(n351), .A2(n350), .B(n386), .ZN(n352) );
  VHSR_CLKNAND2_2 U377 ( .A1(a[7]), .A2(b[7]), .ZN(n385) );
  VHSR_NOR2_1 U378 ( .A1(n354), .A2(n353), .ZN(n355) );
  VHSR_AND3_2 U379 ( .A1(n387), .A2(n357), .A3(n386), .Z(n358) );
  VHSR_NOR2_1 U380 ( .A1(n385), .A2(n358), .ZN(product[15]) );
  VHSR_AD1_1 U381 ( .A(n374), .B(n373), .CI(n372), .CO(n375), .S(product[6])
         );
  VHSR_AD1_1 U382 ( .A(n377), .B(n376), .CI(n375), .CO(n365), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U383 ( .A(n380), .B(n379), .CI(n378), .CO(n381), .S(product[11])
         );
  VHSR_AD1_1 U384 ( .A(n383), .B(n382), .CI(n381), .CO(n351), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U385 ( .A1(n385), .A2(n384), .ZN(n388) );
  VHSR_XOR3_2 U386 ( .A1(n388), .A2(n387), .A3(n386), .Z(product[14]) );
  VHSR_AOI21_2 U387 ( .A1(n394), .A2(n393), .B(n392), .ZN(product[4]) );
  VHSR_OAI22_2 U388 ( .A1(n398), .A2(n397), .B1(n396), .B2(n395), .ZN(
        product[1]) );
  VHSR_AOI31_2 U389 ( .A1(n402), .A2(n401), .A3(n400), .B(n399), .ZN(
        product[2]) );
endmodule


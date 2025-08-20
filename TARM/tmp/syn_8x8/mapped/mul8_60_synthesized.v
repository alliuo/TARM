
module mul8_60 ( a, b, product );
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
         n394, n395, n396, n397, n398, n399, n400, n401, n402, n403, n404,
         n405, n406;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U206 ( .A1(n251), .A2(n247), .ZN(n240) );
  VHSR_NOR2_1 U207 ( .A1(n294), .A2(n295), .ZN(n349) );
  VHSR_NOR2_1 U208 ( .A1(n398), .A2(n397), .ZN(n396) );
  VHSR_NOR2_1 U209 ( .A1(n286), .A2(n339), .ZN(n373) );
  VHSR_IN_2 U210 ( .I(n359), .ZN(product[13]) );
  VHSR_INOR2_1 U211 ( .A1(n363), .B1(n362), .ZN(n394) );
  VHSR_NOR2_2 U212 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_NOR2_2 U213 ( .A1(n349), .A2(n348), .ZN(n361) );
  VHSR_NOR2_2 U214 ( .A1(n296), .A2(n292), .ZN(n294) );
  VHSR_AND4_1 U215 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .Z(n249) );
  VHSR_AD1_1 U216 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(product[9])
         );
  VHSR_AD1_1 U217 ( .A(n381), .B(n403), .CI(n380), .CO(n338), .S(product[3])
         );
  VHSR_AD1_1 U218 ( .A(n396), .B(n379), .CI(n378), .CO(n382), .S(product[5])
         );
  VHSR_AD1_1 U219 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U220 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U221 ( .A(n368), .B(n367), .CI(n366), .CO(n385), .S(product[10])
         );
  VHSR_AOI22_2 U222 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n251) );
  VHSR_IN_2 U223 ( .I(a[7]), .ZN(n282) );
  VHSR_IN_2 U224 ( .I(b[1]), .ZN(n402) );
  VHSR_NOR2_1 U225 ( .A1(n282), .A2(n402), .ZN(n217) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n316) );
  VHSR_IN_2 U227 ( .I(a[5]), .ZN(n287) );
  VHSR_AOI211_2 U228 ( .A1(b[2]), .A2(a[4]), .B(n316), .C(n287), .ZN(n218) );
  VHSR_CLKNAND2_2 U229 ( .A1(a[6]), .A2(b[2]), .ZN(n220) );
  VHSR_IN_2 U230 ( .I(n220), .ZN(n216) );
  VHSR_MAOI222_2 U231 ( .A(n217), .B(n218), .C(n216), .ZN(n230) );
  VHSR_AOI21_2 U232 ( .A1(b[1]), .A2(a[7]), .B(n218), .ZN(n221) );
  VHSR_IN_2 U233 ( .I(n230), .ZN(n219) );
  VHSR_AOI21_2 U234 ( .A1(n221), .A2(n220), .B(n219), .ZN(n258) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[6]), .A2(b[1]), .ZN(n227) );
  VHSR_IN_2 U236 ( .I(n227), .ZN(n224) );
  VHSR_IN_2 U237 ( .I(a[4]), .ZN(n286) );
  VHSR_IN_2 U238 ( .I(b[0]), .ZN(n400) );
  VHSR_NOR4_2 U239 ( .A1(n287), .A2(n286), .A3(n402), .A4(n400), .ZN(n276) );
  VHSR_CLKNAND2_2 U240 ( .A1(b[2]), .A2(a[5]), .ZN(n223) );
  VHSR_CLKNAND2_2 U241 ( .A1(b[3]), .A2(a[4]), .ZN(n222) );
  VHSR_AOI21_2 U242 ( .A1(n223), .A2(n222), .B(n249), .ZN(n225) );
  VHSR_MAOI222_2 U243 ( .A(n224), .B(n276), .C(n225), .ZN(n229) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[2]), .A2(a[4]), .ZN(n272) );
  VHSR_OAI21_2 U245 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n271) );
  VHSR_OAI211_2 U246 ( .A1(n286), .A2(n400), .B(a[5]), .C(b[1]), .ZN(n270) );
  VHSR_MAOI222_2 U247 ( .A(n272), .B(n271), .C(n270), .ZN(n269) );
  VHSR_NOR2_1 U248 ( .A1(n276), .A2(n225), .ZN(n228) );
  VHSR_IN_2 U249 ( .I(n229), .ZN(n226) );
  VHSR_AOI21_2 U250 ( .A1(n228), .A2(n227), .B(n226), .ZN(n261) );
  VHSR_CLKNAND2_2 U251 ( .A1(n269), .A2(n261), .ZN(n260) );
  VHSR_CLKNAND2_2 U252 ( .A1(n229), .A2(n260), .ZN(n257) );
  VHSR_CLKNAND2_2 U253 ( .A1(n258), .A2(n257), .ZN(n256) );
  VHSR_CLKNAND2_2 U254 ( .A1(n230), .A2(n256), .ZN(n248) );
  VHSR_AND3_2 U255 ( .A1(n240), .A2(b[3]), .A3(a[7]), .Z(n300) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n284) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n322) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n285) );
  VHSR_IN_2 U259 ( .I(a[2]), .ZN(n318) );
  VHSR_OAI22_2 U260 ( .A1(n285), .A2(n322), .B1(n284), .B2(n318), .ZN(n246) );
  VHSR_AOI22_2 U261 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n237) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[4]), .A2(a[2]), .ZN(n268) );
  VHSR_NAND3_2 U263 ( .A1(a[3]), .A2(b[5]), .A3(n268), .ZN(n236) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[7]), .A2(a[2]), .ZN(n231) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[1]), .ZN(n233) );
  VHSR_OAI22_2 U266 ( .A1(n237), .A2(n236), .B1(n231), .B2(n233), .ZN(n238) );
  VHSR_IN_2 U267 ( .I(b[4]), .ZN(n339) );
  VHSR_IN_2 U268 ( .I(a[0]), .ZN(n401) );
  VHSR_OAI211_2 U269 ( .A1(n339), .A2(n401), .B(b[5]), .C(a[1]), .ZN(n267) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[0]), .ZN(n266) );
  VHSR_MAOI222_2 U271 ( .A(n268), .B(n267), .C(n266), .ZN(n265) );
  VHSR_NAND4_2 U272 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n243) );
  VHSR_IN_2 U273 ( .I(b[5]), .ZN(n281) );
  VHSR_OAI22_2 U274 ( .A1(n339), .A2(n322), .B1(n281), .B2(n318), .ZN(n232) );
  VHSR_AND2_2 U275 ( .A1(n243), .A2(n232), .Z(n235) );
  VHSR_OAI21_2 U276 ( .A1(n284), .A2(n401), .B(n233), .ZN(n234) );
  VHSR_IN_2 U277 ( .I(a[1]), .ZN(n399) );
  VHSR_NOR4_2 U278 ( .A1(n339), .A2(n281), .A3(n399), .A4(n401), .ZN(n274) );
  VHSR_AND2_2 U279 ( .A1(n265), .A2(n264), .Z(n263) );
  VHSR_AD1_1 U280 ( .A(n235), .B(n234), .CI(n274), .CO(n252), .S(n264) );
  VHSR_AOI21_2 U281 ( .A1(n237), .A2(n236), .B(n238), .ZN(n255) );
  VHSR_OAI32_2 U282 ( .A1(n238), .A2(n263), .A3(n252), .B1(n255), .B2(n238), 
        .ZN(n244) );
  VHSR_CLKNAND2_2 U283 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_CLKNAND2_2 U284 ( .A1(n246), .A2(n242), .ZN(n241) );
  VHSR_NOR3_2 U285 ( .A1(n284), .A2(n322), .A3(n241), .ZN(n299) );
  VHSR_NOR2_1 U286 ( .A1(n316), .A2(n282), .ZN(n239) );
  VHSR_IAO21_2 U287 ( .A1(n240), .A2(n239), .B(n300), .ZN(n303) );
  VHSR_OAI32_2 U288 ( .A1(n299), .A2(n322), .A3(n284), .B1(n241), .B2(n299), 
        .ZN(n302) );
  VHSR_OAI21_2 U289 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U290 ( .A1(n246), .A2(n245), .ZN(n310) );
  VHSR_AOI21_2 U291 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_XNOR2_2 U292 ( .A1(n251), .A2(n250), .ZN(n309) );
  VHSR_NOR2_1 U293 ( .A1(n263), .A2(n252), .ZN(n254) );
  VHSR_AOI22_2 U294 ( .A1(n263), .A2(n252), .B1(n255), .B2(n254), .ZN(n253) );
  VHSR_OAI21_2 U295 ( .A1(n255), .A2(n254), .B(n253), .ZN(n315) );
  VHSR_OAI21_2 U296 ( .A1(n258), .A2(n257), .B(n256), .ZN(n259) );
  VHSR_IN_2 U297 ( .I(n259), .ZN(n314) );
  VHSR_OAI21_2 U298 ( .A1(n269), .A2(n261), .B(n260), .ZN(n262) );
  VHSR_IN_2 U299 ( .I(n262), .ZN(n329) );
  VHSR_IAO21_2 U300 ( .A1(n265), .A2(n264), .B(n263), .ZN(n328) );
  VHSR_AOI31_2 U301 ( .A1(n268), .A2(n267), .A3(n266), .B(n265), .ZN(n336) );
  VHSR_AOI31_2 U302 ( .A1(n272), .A2(n271), .A3(n270), .B(n269), .ZN(n335) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[5]), .A2(a[0]), .ZN(n273) );
  VHSR_OAI32_2 U304 ( .A1(n274), .A2(n399), .A3(n339), .B1(n273), .B2(n274), 
        .ZN(n344) );
  VHSR_NOR2_1 U305 ( .A1(n400), .A2(n401), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U306 ( .A1(n373), .A2(product[0]), .ZN(n341) );
  VHSR_IN_2 U307 ( .I(n341), .ZN(n343) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[1]), .ZN(n275) );
  VHSR_OAI32_2 U309 ( .A1(n276), .A2(n400), .A3(n287), .B1(n275), .B2(n276), 
        .ZN(n342) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[6]), .ZN(n364) );
  VHSR_IN_2 U311 ( .I(n364), .ZN(n391) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n307) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n307), .ZN(n278) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[6]), .ZN(n306) );
  VHSR_NAND3_2 U315 ( .A1(b[7]), .A2(a[5]), .A3(n306), .ZN(n277) );
  VHSR_CLKNAND2_2 U316 ( .A1(n278), .A2(n277), .ZN(n280) );
  VHSR_MAOI222_2 U317 ( .A(n364), .B(n278), .C(n277), .ZN(n348) );
  VHSR_IN_2 U318 ( .I(n348), .ZN(n279) );
  VHSR_OAI21_2 U319 ( .A1(n391), .A2(n280), .B(n279), .ZN(n295) );
  VHSR_IN_2 U320 ( .I(n373), .ZN(n289) );
  VHSR_NOR3_2 U321 ( .A1(n287), .A2(n281), .A3(n289), .ZN(n311) );
  VHSR_NOR3_2 U322 ( .A1(n282), .A2(n307), .A3(n281), .ZN(n356) );
  VHSR_AOI22_2 U323 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n283) );
  VHSR_NOR2_1 U324 ( .A1(n356), .A2(n283), .ZN(n291) );
  VHSR_NOR4_2 U325 ( .A1(n287), .A2(n286), .A3(n285), .A4(n284), .ZN(n354) );
  VHSR_AOI22_2 U326 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n288) );
  VHSR_NOR2_1 U327 ( .A1(n354), .A2(n288), .ZN(n290) );
  VHSR_NAND3_2 U328 ( .A1(b[5]), .A2(a[5]), .A3(n289), .ZN(n305) );
  VHSR_MAOI222_2 U329 ( .A(n307), .B(n306), .C(n305), .ZN(n304) );
  VHSR_AND2_2 U330 ( .A1(n297), .A2(n304), .Z(n296) );
  VHSR_AD1_1 U331 ( .A(n311), .B(n291), .CI(n290), .CO(n292), .S(n297) );
  VHSR_CLKNAND2_2 U332 ( .A1(n296), .A2(n292), .ZN(n293) );
  VHSR_AOI22_2 U333 ( .A1(n295), .A2(n294), .B1(n293), .B2(n349), .ZN(n389) );
  VHSR_IAO21_2 U334 ( .A1(n297), .A2(n304), .B(n296), .ZN(n387) );
  VHSR_AD1_1 U335 ( .A(n300), .B(n299), .CI(n298), .CO(n390), .S(n386) );
  VHSR_AD1_1 U336 ( .A(n303), .B(n302), .CI(n301), .CO(n298), .S(n368) );
  VHSR_AOI31_2 U337 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n367) );
  VHSR_AD1_1 U338 ( .A(n310), .B(n309), .CI(n308), .CO(n301), .S(n371) );
  VHSR_AOI22_2 U339 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n312) );
  VHSR_NOR2_1 U340 ( .A1(n312), .A2(n311), .ZN(n370) );
  VHSR_AD1_1 U341 ( .A(n315), .B(n314), .CI(n313), .CO(n308), .S(n374) );
  VHSR_CLKNAND2_2 U342 ( .A1(b[2]), .A2(a[2]), .ZN(n326) );
  VHSR_IN_2 U343 ( .I(n326), .ZN(n333) );
  VHSR_CLKNAND2_2 U344 ( .A1(b[2]), .A2(a[0]), .ZN(n406) );
  VHSR_NOR3_2 U345 ( .A1(n316), .A2(n399), .A3(n406), .ZN(n347) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n317) );
  VHSR_NOR2_1 U347 ( .A1(n317), .A2(n347), .ZN(n381) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[1]), .A2(a[1]), .ZN(n405) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[0]), .A2(a[2]), .ZN(n404) );
  VHSR_MAOI222_2 U350 ( .A(n406), .B(n405), .C(n404), .ZN(n403) );
  VHSR_OAI22_2 U351 ( .A1(n402), .A2(n318), .B1(n400), .B2(n322), .ZN(n380) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[1]), .A2(a[3]), .ZN(n320) );
  VHSR_NAND3_2 U353 ( .A1(a[1]), .A2(b[3]), .A3(n406), .ZN(n319) );
  VHSR_MAOI222_2 U354 ( .A(n320), .B(n319), .C(n326), .ZN(n323) );
  VHSR_AOI31_2 U355 ( .A1(a[1]), .A2(b[3]), .A3(n406), .B(n333), .ZN(n321) );
  VHSR_OAI32_2 U356 ( .A1(n323), .A2(n322), .A3(n402), .B1(n321), .B2(n323), 
        .ZN(n337) );
  VHSR_AOI21_2 U357 ( .A1(n338), .A2(n337), .B(n323), .ZN(n324) );
  VHSR_IN_2 U358 ( .I(n324), .ZN(n346) );
  VHSR_CLKNAND2_2 U359 ( .A1(b[3]), .A2(a[3]), .ZN(n331) );
  VHSR_AOI22_2 U360 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n325) );
  VHSR_IAO21_2 U361 ( .A1(n331), .A2(n326), .B(n325), .ZN(n345) );
  VHSR_IAO21_2 U362 ( .A1(n333), .A2(n332), .B(n331), .ZN(n377) );
  VHSR_AD1_1 U363 ( .A(n329), .B(n328), .CI(n327), .CO(n313), .S(n376) );
  VHSR_OAI21_2 U364 ( .A1(n333), .A2(n331), .B(n332), .ZN(n330) );
  VHSR_OAI31_2 U365 ( .A1(n333), .A2(n332), .A3(n331), .B(n330), .ZN(n384) );
  VHSR_AD1_1 U366 ( .A(n336), .B(n335), .CI(n334), .CO(n327), .S(n383) );
  VHSR_XNOR2_2 U367 ( .A1(n338), .A2(n337), .ZN(n398) );
  VHSR_NOR2_1 U368 ( .A1(n339), .A2(n401), .ZN(n340) );
  VHSR_AOI32_2 U369 ( .A1(b[0]), .A2(n341), .A3(a[4]), .B1(n340), .B2(n341), 
        .ZN(n397) );
  VHSR_AD1_1 U370 ( .A(n344), .B(n343), .CI(n342), .CO(n334), .S(n379) );
  VHSR_AD1_1 U371 ( .A(n347), .B(n346), .CI(n345), .CO(n332), .S(n378) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[6]), .ZN(n351) );
  VHSR_AOI21_2 U373 ( .A1(a[6]), .A2(b[7]), .B(n351), .ZN(n350) );
  VHSR_AOI31_2 U374 ( .A1(a[6]), .A2(n351), .A3(b[7]), .B(n350), .ZN(n352) );
  VHSR_IN_2 U375 ( .I(n352), .ZN(n353) );
  VHSR_OR2_2 U376 ( .A1(n354), .A2(n353), .Z(n355) );
  VHSR_MAOI222_2 U377 ( .A(n356), .B(n354), .C(n353), .ZN(n363) );
  VHSR_OAI21_2 U378 ( .A1(n356), .A2(n355), .B(n363), .ZN(n360) );
  VHSR_CLKXOR2_2 U379 ( .A1(n361), .A2(n360), .Z(n357) );
  VHSR_CLKNAND2_2 U380 ( .A1(n358), .A2(n357), .ZN(n393) );
  VHSR_OAI21_2 U381 ( .A1(n358), .A2(n357), .B(n393), .ZN(n359) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[7]), .A2(b[7]), .ZN(n392) );
  VHSR_NOR2_1 U383 ( .A1(n361), .A2(n360), .ZN(n362) );
  VHSR_AND3_2 U384 ( .A1(n394), .A2(n364), .A3(n393), .Z(n365) );
  VHSR_NOR2_1 U385 ( .A1(n392), .A2(n365), .ZN(product[15]) );
  VHSR_AD1_1 U386 ( .A(n384), .B(n383), .CI(n382), .CO(n375), .S(product[6])
         );
  VHSR_AD1_1 U387 ( .A(n387), .B(n386), .CI(n385), .CO(n388), .S(product[11])
         );
  VHSR_AD1_1 U388 ( .A(n390), .B(n389), .CI(n388), .CO(n358), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U389 ( .A1(n392), .A2(n391), .ZN(n395) );
  VHSR_XOR3_2 U390 ( .A1(n395), .A2(n394), .A3(n393), .Z(product[14]) );
  VHSR_AOI21_2 U391 ( .A1(n398), .A2(n397), .B(n396), .ZN(product[4]) );
  VHSR_OAI22_2 U392 ( .A1(n402), .A2(n401), .B1(n400), .B2(n399), .ZN(
        product[1]) );
  VHSR_AOI31_2 U393 ( .A1(n406), .A2(n405), .A3(n404), .B(n403), .ZN(
        product[2]) );
endmodule



module mul8_35 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n216, n217, n218, n219, n220,
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
         n397, n398, n399, n400, n401, n402, n403;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U209 ( .A1(n267), .B1(b[3]), .B2(a[5]), .ZN(n226) );
  VHSR_INOR2_2 U210 ( .A1(n238), .B1(n256), .ZN(n244) );
  VHSR_NOR2_1 U211 ( .A1(n348), .A2(n347), .ZN(n358) );
  VHSR_INAND2_2 U212 ( .A1(n323), .B1(n335), .ZN(n342) );
  VHSR_NOR2_1 U213 ( .A1(n294), .A2(n293), .ZN(n292) );
  VHSR_NOR2_1 U214 ( .A1(n396), .A2(n395), .ZN(n394) );
  VHSR_NOR2_1 U215 ( .A1(n338), .A2(n309), .ZN(n371) );
  VHSR_IN_2 U216 ( .I(n357), .ZN(product[13]) );
  VHSR_NOR2_2 U217 ( .A1(n361), .A2(n360), .ZN(n392) );
  VHSR_INOR2_1 U218 ( .A1(n359), .B1(n358), .ZN(n361) );
  VHSR_NOR2_2 U219 ( .A1(n291), .A2(n290), .ZN(n348) );
  VHSR_NOR2_2 U220 ( .A1(n285), .A2(n292), .ZN(n291) );
  VHSR_INAND2_1 U221 ( .A1(n278), .B1(n354), .ZN(n281) );
  VHSR_IOA21_1 U222 ( .A1(n220), .A2(n219), .B(n248), .ZN(n222) );
  VHSR_AD1_1 U223 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(product[6])
         );
  VHSR_AD1_1 U224 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n382), .B(n403), .CI(n381), .CO(n337), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n394), .B(n380), .CI(n379), .CO(n376), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U228 ( .A(n369), .B(n368), .CI(n367), .CO(n383), .S(product[9])
         );
  VHSR_AD1_1 U229 ( .A(n366), .B(n365), .CI(n364), .CO(n386), .S(product[11])
         );
  VHSR_IN_2 U230 ( .I(a[7]), .ZN(n240) );
  VHSR_IN_2 U231 ( .I(b[3]), .ZN(n317) );
  VHSR_IN_2 U232 ( .I(a[6]), .ZN(n218) );
  VHSR_IN_2 U233 ( .I(b[2]), .ZN(n401) );
  VHSR_OAI22_2 U234 ( .A1(n218), .A2(n317), .B1(n240), .B2(n401), .ZN(n251) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[7]), .A2(b[1]), .ZN(n217) );
  VHSR_IN_2 U236 ( .I(a[4]), .ZN(n309) );
  VHSR_NOR2_1 U237 ( .A1(n309), .A2(n401), .ZN(n267) );
  VHSR_CLKNAND2_2 U238 ( .A1(a[6]), .A2(b[2]), .ZN(n216) );
  VHSR_MAOI222_2 U239 ( .A(n217), .B(n226), .C(n216), .ZN(n229) );
  VHSR_IN_2 U240 ( .I(a[5]), .ZN(n308) );
  VHSR_IN_2 U241 ( .I(b[1]), .ZN(n399) );
  VHSR_AOI211_2 U242 ( .A1(a[4]), .A2(b[0]), .B(n308), .C(n399), .ZN(n266) );
  VHSR_IN_2 U243 ( .I(b[0]), .ZN(n398) );
  VHSR_AOI21_2 U244 ( .A1(n218), .A2(n240), .B(n398), .ZN(n265) );
  VHSR_MAOI222_2 U245 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_CLKNAND2_2 U246 ( .A1(a[6]), .A2(b[1]), .ZN(n221) );
  VHSR_NAND4_2 U247 ( .A1(a[4]), .A2(a[5]), .A3(b[1]), .A4(b[0]), .ZN(n275) );
  VHSR_CLKNAND2_2 U248 ( .A1(a[5]), .A2(b[2]), .ZN(n220) );
  VHSR_CLKNAND2_2 U249 ( .A1(a[4]), .A2(b[3]), .ZN(n219) );
  VHSR_NAND4_2 U250 ( .A1(a[4]), .A2(a[5]), .A3(b[3]), .A4(b[2]), .ZN(n248) );
  VHSR_MAOI222_2 U251 ( .A(n221), .B(n275), .C(n222), .ZN(n225) );
  VHSR_IN_2 U252 ( .I(n225), .ZN(n224) );
  VHSR_CLKNAND2_2 U253 ( .A1(n275), .A2(n222), .ZN(n223) );
  VHSR_AOI32_2 U254 ( .A1(b[1]), .A2(n224), .A3(a[6]), .B1(n223), .B2(n224), 
        .ZN(n261) );
  VHSR_NOR2_1 U255 ( .A1(n264), .A2(n261), .ZN(n260) );
  VHSR_NOR2_1 U256 ( .A1(n260), .A2(n225), .ZN(n254) );
  VHSR_IN_2 U257 ( .I(n229), .ZN(n228) );
  VHSR_OAI21_2 U258 ( .A1(n399), .A2(n240), .B(n226), .ZN(n227) );
  VHSR_AOI32_2 U259 ( .A1(b[2]), .A2(n228), .A3(a[6]), .B1(n227), .B2(n228), 
        .ZN(n253) );
  VHSR_NOR2_1 U260 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_NOR2_1 U261 ( .A1(n229), .A2(n252), .ZN(n249) );
  VHSR_CLKNAND2_2 U262 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U263 ( .A1(n251), .A2(n247), .ZN(n239) );
  VHSR_NOR3_2 U264 ( .A1(n240), .A2(n317), .A3(n239), .ZN(n297) );
  VHSR_IN_2 U265 ( .I(b[7]), .ZN(n279) );
  VHSR_IN_2 U266 ( .I(a[3]), .ZN(n320) );
  VHSR_IN_2 U267 ( .I(b[6]), .ZN(n233) );
  VHSR_IN_2 U268 ( .I(a[2]), .ZN(n321) );
  VHSR_OAI22_2 U269 ( .A1(n233), .A2(n320), .B1(n279), .B2(n321), .ZN(n246) );
  VHSR_NOR2_1 U270 ( .A1(n279), .A2(n321), .ZN(n231) );
  VHSR_IN_2 U271 ( .I(a[1]), .ZN(n397) );
  VHSR_NOR2_1 U272 ( .A1(n233), .A2(n397), .ZN(n230) );
  VHSR_IN_2 U273 ( .I(b[5]), .ZN(n310) );
  VHSR_AOI211_2 U274 ( .A1(a[2]), .A2(b[4]), .B(n310), .C(n320), .ZN(n237) );
  VHSR_OAI22_2 U275 ( .A1(n233), .A2(n321), .B1(n279), .B2(n397), .ZN(n236) );
  VHSR_AOI22_2 U276 ( .A1(n231), .A2(n230), .B1(n237), .B2(n236), .ZN(n238) );
  VHSR_CLKNAND2_2 U277 ( .A1(b[4]), .A2(a[2]), .ZN(n272) );
  VHSR_CLKNAND2_2 U278 ( .A1(b[6]), .A2(a[0]), .ZN(n271) );
  VHSR_IN_2 U279 ( .I(b[4]), .ZN(n338) );
  VHSR_IN_2 U280 ( .I(a[0]), .ZN(n402) );
  VHSR_OAI211_2 U281 ( .A1(n338), .A2(n402), .B(b[5]), .C(a[1]), .ZN(n270) );
  VHSR_MAOI222_2 U282 ( .A(n272), .B(n271), .C(n270), .ZN(n269) );
  VHSR_NAND4_2 U283 ( .A1(b[5]), .A2(b[4]), .A3(a[2]), .A4(a[3]), .ZN(n243) );
  VHSR_OAI22_2 U284 ( .A1(n310), .A2(n321), .B1(n338), .B2(n320), .ZN(n232) );
  VHSR_AND2_2 U285 ( .A1(n243), .A2(n232), .Z(n235) );
  VHSR_OAI22_2 U286 ( .A1(n233), .A2(n397), .B1(n279), .B2(n402), .ZN(n234) );
  VHSR_NOR4_2 U287 ( .A1(n310), .A2(n338), .A3(n397), .A4(n402), .ZN(n274) );
  VHSR_AND2_2 U288 ( .A1(n269), .A2(n263), .Z(n262) );
  VHSR_AD1_1 U289 ( .A(n235), .B(n234), .CI(n274), .CO(n255), .S(n263) );
  VHSR_NOR2_1 U290 ( .A1(n262), .A2(n255), .ZN(n258) );
  VHSR_OAI21_2 U291 ( .A1(n237), .A2(n236), .B(n238), .ZN(n259) );
  VHSR_NOR2_1 U292 ( .A1(n258), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U293 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_CLKNAND2_2 U294 ( .A1(n246), .A2(n242), .ZN(n241) );
  VHSR_NOR3_2 U295 ( .A1(n279), .A2(n320), .A3(n241), .ZN(n296) );
  VHSR_OAI32_2 U296 ( .A1(n297), .A2(n317), .A3(n240), .B1(n239), .B2(n297), 
        .ZN(n300) );
  VHSR_OAI32_2 U297 ( .A1(n296), .A2(n320), .A3(n279), .B1(n241), .B2(n296), 
        .ZN(n299) );
  VHSR_OAI21_2 U298 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U299 ( .A1(n246), .A2(n245), .ZN(n307) );
  VHSR_OAI21_2 U300 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_XNOR2_2 U301 ( .A1(n251), .A2(n250), .ZN(n306) );
  VHSR_AOI21_2 U302 ( .A1(n254), .A2(n253), .B(n252), .ZN(n315) );
  VHSR_CLKNAND2_2 U303 ( .A1(n262), .A2(n255), .ZN(n257) );
  VHSR_AOI22_2 U304 ( .A1(n259), .A2(n258), .B1(n257), .B2(n256), .ZN(n314) );
  VHSR_AOI21_2 U305 ( .A1(n264), .A2(n261), .B(n260), .ZN(n327) );
  VHSR_IAO21_2 U306 ( .A1(n269), .A2(n263), .B(n262), .ZN(n326) );
  VHSR_OAI31_2 U307 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n268) );
  VHSR_IN_2 U308 ( .I(n268), .ZN(n330) );
  VHSR_AOI31_2 U309 ( .A1(n272), .A2(n271), .A3(n270), .B(n269), .ZN(n329) );
  VHSR_CLKNAND2_2 U310 ( .A1(b[4]), .A2(a[1]), .ZN(n273) );
  VHSR_OAI32_2 U311 ( .A1(n274), .A2(n402), .A3(n310), .B1(n273), .B2(n274), 
        .ZN(n346) );
  VHSR_IN_2 U312 ( .I(n275), .ZN(n277) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[5]), .A2(b[0]), .ZN(n276) );
  VHSR_OAI32_2 U314 ( .A1(n277), .A2(n399), .A3(n309), .B1(n276), .B2(n277), 
        .ZN(n345) );
  VHSR_NOR2_1 U315 ( .A1(n398), .A2(n402), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U316 ( .A1(n371), .A2(product[0]), .ZN(n340) );
  VHSR_IN_2 U317 ( .I(n340), .ZN(n344) );
  VHSR_AOI22_2 U318 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n278) );
  VHSR_NAND4_2 U319 ( .A1(a[6]), .A2(a[7]), .A3(b[5]), .A4(b[4]), .ZN(n354) );
  VHSR_NAND3_2 U320 ( .A1(b[5]), .A2(a[5]), .A3(n371), .ZN(n312) );
  VHSR_NAND4_2 U321 ( .A1(b[6]), .A2(b[7]), .A3(a[4]), .A4(a[5]), .ZN(n352) );
  VHSR_NOR2_1 U322 ( .A1(n279), .A2(n309), .ZN(n280) );
  VHSR_AOI32_2 U323 ( .A1(b[6]), .A2(n352), .A3(a[5]), .B1(n280), .B2(n352), 
        .ZN(n283) );
  VHSR_MAOI222_2 U324 ( .A(n281), .B(n312), .C(n283), .ZN(n285) );
  VHSR_AND2_2 U325 ( .A1(n281), .A2(n312), .Z(n282) );
  VHSR_AOI21_2 U326 ( .A1(n283), .A2(n282), .B(n285), .ZN(n284) );
  VHSR_IN_2 U327 ( .I(n284), .ZN(n294) );
  VHSR_CLKNAND2_2 U328 ( .A1(a[6]), .A2(b[4]), .ZN(n304) );
  VHSR_CLKNAND2_2 U329 ( .A1(b[6]), .A2(a[4]), .ZN(n303) );
  VHSR_OR3_2 U330 ( .A1(n371), .A2(n308), .A3(n310), .Z(n302) );
  VHSR_MAOI222_2 U331 ( .A(n304), .B(n303), .C(n302), .ZN(n301) );
  VHSR_IN_2 U332 ( .I(n301), .ZN(n293) );
  VHSR_CLKNAND2_2 U333 ( .A1(a[6]), .A2(b[6]), .ZN(n362) );
  VHSR_IN_2 U334 ( .I(n362), .ZN(n389) );
  VHSR_NAND3_2 U335 ( .A1(b[5]), .A2(a[7]), .A3(n304), .ZN(n287) );
  VHSR_NAND3_2 U336 ( .A1(a[5]), .A2(b[7]), .A3(n303), .ZN(n286) );
  VHSR_CLKNAND2_2 U337 ( .A1(n287), .A2(n286), .ZN(n289) );
  VHSR_MAOI222_2 U338 ( .A(n362), .B(n287), .C(n286), .ZN(n347) );
  VHSR_IN_2 U339 ( .I(n347), .ZN(n288) );
  VHSR_OAI21_2 U340 ( .A1(n389), .A2(n289), .B(n288), .ZN(n290) );
  VHSR_AOI21_2 U341 ( .A1(n291), .A2(n290), .B(n348), .ZN(n387) );
  VHSR_AOI21_2 U342 ( .A1(n294), .A2(n293), .B(n292), .ZN(n366) );
  VHSR_AD1_1 U343 ( .A(n297), .B(n296), .CI(n295), .CO(n388), .S(n365) );
  VHSR_AD1_1 U344 ( .A(n300), .B(n299), .CI(n298), .CO(n295), .S(n385) );
  VHSR_AOI31_2 U345 ( .A1(n304), .A2(n303), .A3(n302), .B(n301), .ZN(n384) );
  VHSR_AD1_1 U346 ( .A(n307), .B(n306), .CI(n305), .CO(n298), .S(n369) );
  VHSR_OAI22_2 U347 ( .A1(n310), .A2(n309), .B1(n338), .B2(n308), .ZN(n311) );
  VHSR_AND2_2 U348 ( .A1(n312), .A2(n311), .Z(n368) );
  VHSR_AD1_1 U349 ( .A(n315), .B(n314), .CI(n313), .CO(n305), .S(n372) );
  VHSR_NOR2_1 U350 ( .A1(n401), .A2(n321), .ZN(n334) );
  VHSR_IN_2 U351 ( .I(n334), .ZN(n324) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[3]), .A2(a[3]), .ZN(n333) );
  VHSR_AOI22_2 U353 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n316) );
  VHSR_IAO21_2 U354 ( .A1(n324), .A2(n333), .B(n316), .ZN(n343) );
  VHSR_AOI22_2 U355 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n322) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[1]), .A2(a[1]), .ZN(n318) );
  VHSR_OAI22_2 U357 ( .A1(n324), .A2(n322), .B1(n333), .B2(n318), .ZN(n323) );
  VHSR_OAI22_2 U358 ( .A1(n317), .A2(n402), .B1(n401), .B2(n397), .ZN(n382) );
  VHSR_OAI21_2 U359 ( .A1(n321), .A2(n398), .B(n318), .ZN(n319) );
  VHSR_IN_2 U360 ( .I(n319), .ZN(n400) );
  VHSR_NOR3_2 U361 ( .A1(n400), .A2(n402), .A3(n401), .ZN(n403) );
  VHSR_OAI22_2 U362 ( .A1(n399), .A2(n321), .B1(n398), .B2(n320), .ZN(n381) );
  VHSR_AOI21_2 U363 ( .A1(n322), .A2(n324), .B(n323), .ZN(n336) );
  VHSR_CLKNAND2_2 U364 ( .A1(n337), .A2(n336), .ZN(n335) );
  VHSR_CLKNAND2_2 U365 ( .A1(n343), .A2(n342), .ZN(n331) );
  VHSR_AOI21_2 U366 ( .A1(n324), .A2(n331), .B(n333), .ZN(n375) );
  VHSR_AD1_1 U367 ( .A(n327), .B(n326), .CI(n325), .CO(n313), .S(n374) );
  VHSR_AD1_1 U368 ( .A(n330), .B(n329), .CI(n328), .CO(n325), .S(n378) );
  VHSR_IN_2 U369 ( .I(n331), .ZN(n341) );
  VHSR_CLKNAND2_2 U370 ( .A1(n341), .A2(n333), .ZN(n332) );
  VHSR_OAI31_2 U371 ( .A1(n334), .A2(n341), .A3(n333), .B(n332), .ZN(n377) );
  VHSR_OAI21_2 U372 ( .A1(n337), .A2(n336), .B(n335), .ZN(n396) );
  VHSR_NOR2_1 U373 ( .A1(n338), .A2(n402), .ZN(n339) );
  VHSR_AOI32_2 U374 ( .A1(b[0]), .A2(n340), .A3(a[4]), .B1(n339), .B2(n340), 
        .ZN(n395) );
  VHSR_IAO21_2 U375 ( .A1(n343), .A2(n342), .B(n341), .ZN(n380) );
  VHSR_AD1_1 U376 ( .A(n346), .B(n345), .CI(n344), .CO(n328), .S(n379) );
  VHSR_CLKNAND2_2 U377 ( .A1(b[6]), .A2(a[7]), .ZN(n350) );
  VHSR_AOI21_2 U378 ( .A1(a[6]), .A2(b[7]), .B(n350), .ZN(n349) );
  VHSR_AOI31_2 U379 ( .A1(a[6]), .A2(n350), .A3(b[7]), .B(n349), .ZN(n351) );
  VHSR_AND2_2 U380 ( .A1(n352), .A2(n351), .Z(n353) );
  VHSR_MAOI222_2 U381 ( .A(n354), .B(n352), .C(n351), .ZN(n360) );
  VHSR_AOI21_2 U382 ( .A1(n354), .A2(n353), .B(n360), .ZN(n359) );
  VHSR_XNOR2_2 U383 ( .A1(n358), .A2(n359), .ZN(n355) );
  VHSR_CLKNAND2_2 U384 ( .A1(n356), .A2(n355), .ZN(n391) );
  VHSR_OAI21_2 U385 ( .A1(n356), .A2(n355), .B(n391), .ZN(n357) );
  VHSR_CLKNAND2_2 U386 ( .A1(a[7]), .A2(b[7]), .ZN(n390) );
  VHSR_AND3_2 U387 ( .A1(n392), .A2(n362), .A3(n391), .Z(n363) );
  VHSR_NOR2_1 U388 ( .A1(n390), .A2(n363), .ZN(product[15]) );
  VHSR_AD1_1 U389 ( .A(n385), .B(n384), .CI(n383), .CO(n364), .S(product[10])
         );
  VHSR_AD1_1 U390 ( .A(n388), .B(n387), .CI(n386), .CO(n356), .S(product[12])
         );
  VHSR_NOR2_1 U391 ( .A1(n390), .A2(n389), .ZN(n393) );
  VHSR_XOR3_2 U392 ( .A1(n393), .A2(n392), .A3(n391), .Z(product[14]) );
  VHSR_AOI21_2 U393 ( .A1(n396), .A2(n395), .B(n394), .ZN(product[4]) );
  VHSR_OAI22_2 U394 ( .A1(n399), .A2(n402), .B1(n398), .B2(n397), .ZN(
        product[1]) );
  VHSR_OAI32_2 U395 ( .A1(n403), .A2(n402), .A3(n401), .B1(n400), .B2(n403), 
        .ZN(product[2]) );
endmodule


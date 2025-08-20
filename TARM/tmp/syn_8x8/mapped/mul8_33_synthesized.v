
module mul8_33 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n218, n219,
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
         n396, n397, n398, n399, n400, n401, n402, n403, n404, n405;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_IN_2 U208 ( .I(n247), .ZN(n221) );
  VHSR_INAND3_2 U209 ( .A1(n269), .B1(a[5]), .B2(b[3]), .ZN(n219) );
  VHSR_NOR2_1 U210 ( .A1(n218), .A2(n285), .ZN(n269) );
  VHSR_INOR2_2 U211 ( .A1(n229), .B1(n255), .ZN(n248) );
  VHSR_NOR2_1 U212 ( .A1(n257), .A2(n256), .ZN(n255) );
  VHSR_NOR2_1 U213 ( .A1(n345), .A2(n344), .ZN(n357) );
  VHSR_NOR2_1 U214 ( .A1(n293), .A2(n294), .ZN(n345) );
  VHSR_IOA21_2 U215 ( .A1(n394), .A2(n393), .B(n392), .ZN(n396) );
  VHSR_INOR2_2 U216 ( .A1(n359), .B1(n358), .ZN(n390) );
  VHSR_IN_2 U217 ( .I(n355), .ZN(product[13]) );
  VHSR_NOR2_2 U218 ( .A1(n239), .A2(n238), .ZN(n299) );
  VHSR_INOR2_1 U219 ( .A1(n227), .B1(n258), .ZN(n257) );
  VHSR_NOR2_2 U220 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_MOAI22_1 U221 ( .A1(n280), .A2(n318), .B1(b[4]), .B2(a[3]), .ZN(n231)
         );
  VHSR_NOR2_2 U222 ( .A1(n275), .A2(n284), .ZN(n387) );
  VHSR_AD1_1 U223 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(product[9])
         );
  VHSR_AD1_1 U224 ( .A(n374), .B(n402), .CI(n373), .CO(n337), .S(product[3])
         );
  VHSR_AD1_1 U225 ( .A(n395), .B(n372), .CI(n371), .CO(n375), .S(product[5])
         );
  VHSR_AD1_1 U226 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U227 ( .A(n364), .B(n363), .CI(n362), .CO(n381), .S(product[10])
         );
  VHSR_CLKNAND2_2 U228 ( .A1(b[3]), .A2(a[7]), .ZN(n239) );
  VHSR_IN_2 U229 ( .I(b[3]), .ZN(n315) );
  VHSR_IN_2 U230 ( .I(a[6]), .ZN(n275) );
  VHSR_IN_2 U231 ( .I(a[7]), .ZN(n281) );
  VHSR_IN_2 U232 ( .I(b[2]), .ZN(n218) );
  VHSR_OAI22_2 U233 ( .A1(n315), .A2(n275), .B1(n281), .B2(n218), .ZN(n250) );
  VHSR_IN_2 U234 ( .I(b[1]), .ZN(n401) );
  VHSR_IN_2 U235 ( .I(a[4]), .ZN(n285) );
  VHSR_OAI21_2 U236 ( .A1(n401), .A2(n281), .B(n219), .ZN(n228) );
  VHSR_IN_2 U237 ( .I(a[5]), .ZN(n286) );
  VHSR_NOR4_2 U238 ( .A1(n269), .A2(n286), .A3(n239), .A4(n401), .ZN(n220) );
  VHSR_AOI31_2 U239 ( .A1(b[2]), .A2(a[6]), .A3(n228), .B(n220), .ZN(n229) );
  VHSR_NOR2_1 U240 ( .A1(n275), .A2(n401), .ZN(n224) );
  VHSR_IN_2 U241 ( .I(b[0]), .ZN(n399) );
  VHSR_NOR4_2 U242 ( .A1(n286), .A2(n285), .A3(n401), .A4(n399), .ZN(n274) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[2]), .A2(a[5]), .ZN(n223) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[3]), .A2(a[4]), .ZN(n222) );
  VHSR_NAND4_2 U245 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n247) );
  VHSR_AOI21_2 U246 ( .A1(n223), .A2(n222), .B(n221), .ZN(n225) );
  VHSR_MAOI222_2 U247 ( .A(n224), .B(n274), .C(n225), .ZN(n227) );
  VHSR_AOI211_2 U248 ( .A1(a[4]), .A2(b[0]), .B(n286), .C(n401), .ZN(n268) );
  VHSR_AOI21_2 U249 ( .A1(n281), .A2(n275), .B(n399), .ZN(n267) );
  VHSR_MAOI222_2 U250 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_OR2_2 U251 ( .A1(n274), .A2(n225), .Z(n226) );
  VHSR_AOI32_2 U252 ( .A1(b[1]), .A2(n227), .A3(a[6]), .B1(n226), .B2(n227), 
        .ZN(n259) );
  VHSR_NOR2_1 U253 ( .A1(n266), .A2(n259), .ZN(n258) );
  VHSR_AOI32_2 U254 ( .A1(b[2]), .A2(n229), .A3(a[6]), .B1(n228), .B2(n229), 
        .ZN(n256) );
  VHSR_CLKNAND2_2 U255 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_CLKNAND2_2 U256 ( .A1(n250), .A2(n246), .ZN(n238) );
  VHSR_IN_2 U257 ( .I(b[7]), .ZN(n283) );
  VHSR_IN_2 U258 ( .I(a[3]), .ZN(n323) );
  VHSR_IN_2 U259 ( .I(b[6]), .ZN(n284) );
  VHSR_IN_2 U260 ( .I(a[2]), .ZN(n318) );
  VHSR_OAI22_2 U261 ( .A1(n284), .A2(n323), .B1(n283), .B2(n318), .ZN(n245) );
  VHSR_AOI22_2 U262 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n236) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[4]), .A2(a[2]), .ZN(n265) );
  VHSR_NAND3_2 U264 ( .A1(a[3]), .A2(b[5]), .A3(n265), .ZN(n235) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[7]), .A2(a[2]), .ZN(n230) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[6]), .A2(a[1]), .ZN(n232) );
  VHSR_OAI22_2 U267 ( .A1(n236), .A2(n235), .B1(n230), .B2(n232), .ZN(n237) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[4]), .A2(a[0]), .ZN(n393) );
  VHSR_NAND3_2 U269 ( .A1(a[1]), .A2(b[5]), .A3(n393), .ZN(n264) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[0]), .ZN(n263) );
  VHSR_MAOI222_2 U271 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_NAND4_2 U272 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n242) );
  VHSR_IN_2 U273 ( .I(b[5]), .ZN(n280) );
  VHSR_AND2_2 U274 ( .A1(n242), .A2(n231), .Z(n234) );
  VHSR_IN_2 U275 ( .I(a[0]), .ZN(n400) );
  VHSR_OAI21_2 U276 ( .A1(n283), .A2(n400), .B(n232), .ZN(n233) );
  VHSR_IN_2 U277 ( .I(a[1]), .ZN(n398) );
  VHSR_NOR3_2 U278 ( .A1(n280), .A2(n398), .A3(n393), .ZN(n272) );
  VHSR_AND2_2 U279 ( .A1(n262), .A2(n261), .Z(n260) );
  VHSR_AD1_1 U280 ( .A(n234), .B(n233), .CI(n272), .CO(n251), .S(n261) );
  VHSR_AOI21_2 U281 ( .A1(n236), .A2(n235), .B(n237), .ZN(n254) );
  VHSR_OAI32_2 U282 ( .A1(n237), .A2(n260), .A3(n251), .B1(n254), .B2(n237), 
        .ZN(n243) );
  VHSR_CLKNAND2_2 U283 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U284 ( .A1(n245), .A2(n241), .ZN(n240) );
  VHSR_NOR3_2 U285 ( .A1(n283), .A2(n323), .A3(n240), .ZN(n298) );
  VHSR_AOI21_2 U286 ( .A1(n239), .A2(n238), .B(n299), .ZN(n302) );
  VHSR_OAI32_2 U287 ( .A1(n298), .A2(n323), .A3(n283), .B1(n240), .B2(n298), 
        .ZN(n301) );
  VHSR_OAI21_2 U288 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U289 ( .A1(n245), .A2(n244), .ZN(n309) );
  VHSR_OAI21_2 U290 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U291 ( .A1(n250), .A2(n249), .ZN(n308) );
  VHSR_NOR2_1 U292 ( .A1(n260), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U293 ( .A1(n260), .A2(n251), .B1(n254), .B2(n253), .ZN(n252) );
  VHSR_OAI21_2 U294 ( .A1(n254), .A2(n253), .B(n252), .ZN(n314) );
  VHSR_AOI21_2 U295 ( .A1(n257), .A2(n256), .B(n255), .ZN(n313) );
  VHSR_AOI21_2 U296 ( .A1(n266), .A2(n259), .B(n258), .ZN(n328) );
  VHSR_IAO21_2 U297 ( .A1(n262), .A2(n261), .B(n260), .ZN(n327) );
  VHSR_AOI31_2 U298 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n335) );
  VHSR_OAI31_2 U299 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n270) );
  VHSR_IN_2 U300 ( .I(n270), .ZN(n334) );
  VHSR_AOI22_2 U301 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n271) );
  VHSR_NOR2_1 U302 ( .A1(n272), .A2(n271), .ZN(n340) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[4]), .ZN(n288) );
  VHSR_IN_2 U304 ( .I(n288), .ZN(n369) );
  VHSR_NOR2_1 U305 ( .A1(n399), .A2(n400), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U306 ( .A1(n369), .A2(product[0]), .ZN(n392) );
  VHSR_IN_2 U307 ( .I(n392), .ZN(n339) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[1]), .ZN(n273) );
  VHSR_OAI32_2 U309 ( .A1(n274), .A2(n399), .A3(n286), .B1(n273), .B2(n274), 
        .ZN(n338) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[4]), .A2(b[6]), .ZN(n305) );
  VHSR_NAND3_2 U311 ( .A1(b[7]), .A2(a[5]), .A3(n305), .ZN(n277) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n306) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n306), .ZN(n276) );
  VHSR_CLKNAND2_2 U314 ( .A1(n277), .A2(n276), .ZN(n279) );
  VHSR_IN_2 U315 ( .I(n387), .ZN(n360) );
  VHSR_MAOI222_2 U316 ( .A(n360), .B(n277), .C(n276), .ZN(n344) );
  VHSR_IN_2 U317 ( .I(n344), .ZN(n278) );
  VHSR_OAI21_2 U318 ( .A1(n387), .A2(n279), .B(n278), .ZN(n294) );
  VHSR_NOR3_2 U319 ( .A1(n286), .A2(n280), .A3(n288), .ZN(n310) );
  VHSR_NOR3_2 U320 ( .A1(n281), .A2(n306), .A3(n280), .ZN(n352) );
  VHSR_AOI22_2 U321 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n282) );
  VHSR_NOR2_1 U322 ( .A1(n352), .A2(n282), .ZN(n290) );
  VHSR_NOR4_2 U323 ( .A1(n286), .A2(n285), .A3(n284), .A4(n283), .ZN(n350) );
  VHSR_AOI22_2 U324 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n287) );
  VHSR_NOR2_1 U325 ( .A1(n350), .A2(n287), .ZN(n289) );
  VHSR_NAND3_2 U326 ( .A1(b[5]), .A2(a[5]), .A3(n288), .ZN(n304) );
  VHSR_MAOI222_2 U327 ( .A(n306), .B(n305), .C(n304), .ZN(n303) );
  VHSR_AND2_2 U328 ( .A1(n296), .A2(n303), .Z(n295) );
  VHSR_AD1_1 U329 ( .A(n310), .B(n290), .CI(n289), .CO(n291), .S(n296) );
  VHSR_CLKNAND2_2 U330 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_AOI22_2 U331 ( .A1(n294), .A2(n293), .B1(n292), .B2(n345), .ZN(n385) );
  VHSR_IAO21_2 U332 ( .A1(n296), .A2(n303), .B(n295), .ZN(n383) );
  VHSR_AD1_1 U333 ( .A(n299), .B(n298), .CI(n297), .CO(n386), .S(n382) );
  VHSR_AD1_1 U334 ( .A(n302), .B(n301), .CI(n300), .CO(n297), .S(n364) );
  VHSR_AOI31_2 U335 ( .A1(n306), .A2(n305), .A3(n304), .B(n303), .ZN(n363) );
  VHSR_AD1_1 U336 ( .A(n309), .B(n308), .CI(n307), .CO(n300), .S(n367) );
  VHSR_AOI22_2 U337 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n311) );
  VHSR_NOR2_1 U338 ( .A1(n311), .A2(n310), .ZN(n366) );
  VHSR_AD1_1 U339 ( .A(n314), .B(n313), .CI(n312), .CO(n307), .S(n370) );
  VHSR_CLKNAND2_2 U340 ( .A1(b[2]), .A2(a[2]), .ZN(n320) );
  VHSR_IN_2 U341 ( .I(n320), .ZN(n332) );
  VHSR_CLKNAND2_2 U342 ( .A1(b[2]), .A2(a[0]), .ZN(n405) );
  VHSR_NOR3_2 U343 ( .A1(n315), .A2(n398), .A3(n405), .ZN(n343) );
  VHSR_CLKNAND2_2 U344 ( .A1(b[3]), .A2(a[3]), .ZN(n330) );
  VHSR_AOI22_2 U345 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n316) );
  VHSR_IAO21_2 U346 ( .A1(n330), .A2(n320), .B(n316), .ZN(n342) );
  VHSR_AOI22_2 U347 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n317) );
  VHSR_NOR2_1 U348 ( .A1(n317), .A2(n343), .ZN(n374) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[1]), .A2(a[1]), .ZN(n404) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[0]), .A2(a[2]), .ZN(n403) );
  VHSR_MAOI222_2 U351 ( .A(n405), .B(n404), .C(n403), .ZN(n402) );
  VHSR_OAI22_2 U352 ( .A1(n401), .A2(n318), .B1(n399), .B2(n323), .ZN(n373) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[1]), .A2(a[3]), .ZN(n321) );
  VHSR_NAND3_2 U354 ( .A1(a[1]), .A2(b[3]), .A3(n405), .ZN(n319) );
  VHSR_MAOI222_2 U355 ( .A(n321), .B(n320), .C(n319), .ZN(n324) );
  VHSR_AOI31_2 U356 ( .A1(a[1]), .A2(b[3]), .A3(n405), .B(n332), .ZN(n322) );
  VHSR_OAI32_2 U357 ( .A1(n324), .A2(n323), .A3(n401), .B1(n322), .B2(n324), 
        .ZN(n336) );
  VHSR_AOI21_2 U358 ( .A1(n337), .A2(n336), .B(n324), .ZN(n325) );
  VHSR_IN_2 U359 ( .I(n325), .ZN(n341) );
  VHSR_IAO21_2 U360 ( .A1(n332), .A2(n331), .B(n330), .ZN(n380) );
  VHSR_AD1_1 U361 ( .A(n328), .B(n327), .CI(n326), .CO(n312), .S(n379) );
  VHSR_OAI21_2 U362 ( .A1(n332), .A2(n330), .B(n331), .ZN(n329) );
  VHSR_OAI31_2 U363 ( .A1(n332), .A2(n331), .A3(n330), .B(n329), .ZN(n377) );
  VHSR_AD1_1 U364 ( .A(n335), .B(n334), .CI(n333), .CO(n326), .S(n376) );
  VHSR_CLKNAND2_2 U365 ( .A1(a[4]), .A2(b[0]), .ZN(n394) );
  VHSR_XNOR2_2 U366 ( .A1(n337), .A2(n336), .ZN(n397) );
  VHSR_AOI211_2 U367 ( .A1(n394), .A2(n393), .B(n339), .C(n397), .ZN(n395) );
  VHSR_AD1_1 U368 ( .A(n340), .B(n339), .CI(n338), .CO(n333), .S(n372) );
  VHSR_AD1_1 U369 ( .A(n343), .B(n342), .CI(n341), .CO(n331), .S(n371) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[6]), .A2(b[7]), .ZN(n347) );
  VHSR_AOI21_2 U371 ( .A1(a[7]), .A2(b[6]), .B(n347), .ZN(n346) );
  VHSR_AOI31_2 U372 ( .A1(a[7]), .A2(n347), .A3(b[6]), .B(n346), .ZN(n348) );
  VHSR_IN_2 U373 ( .I(n348), .ZN(n349) );
  VHSR_OR2_2 U374 ( .A1(n350), .A2(n349), .Z(n351) );
  VHSR_MAOI222_2 U375 ( .A(n352), .B(n350), .C(n349), .ZN(n359) );
  VHSR_OAI21_2 U376 ( .A1(n352), .A2(n351), .B(n359), .ZN(n356) );
  VHSR_CLKXOR2_2 U377 ( .A1(n357), .A2(n356), .Z(n353) );
  VHSR_CLKNAND2_2 U378 ( .A1(n354), .A2(n353), .ZN(n389) );
  VHSR_OAI21_2 U379 ( .A1(n354), .A2(n353), .B(n389), .ZN(n355) );
  VHSR_CLKNAND2_2 U380 ( .A1(a[7]), .A2(b[7]), .ZN(n388) );
  VHSR_NOR2_1 U381 ( .A1(n357), .A2(n356), .ZN(n358) );
  VHSR_AND3_2 U382 ( .A1(n390), .A2(n360), .A3(n389), .Z(n361) );
  VHSR_NOR2_1 U383 ( .A1(n388), .A2(n361), .ZN(product[15]) );
  VHSR_AD1_1 U384 ( .A(n377), .B(n376), .CI(n375), .CO(n378), .S(product[6])
         );
  VHSR_AD1_1 U385 ( .A(n380), .B(n379), .CI(n378), .CO(n368), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U386 ( .A(n383), .B(n382), .CI(n381), .CO(n384), .S(product[11])
         );
  VHSR_AD1_1 U387 ( .A(n386), .B(n385), .CI(n384), .CO(n354), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U388 ( .A1(n388), .A2(n387), .ZN(n391) );
  VHSR_XOR3_2 U389 ( .A1(n391), .A2(n390), .A3(n389), .Z(product[14]) );
  VHSR_AOI21_2 U390 ( .A1(n397), .A2(n396), .B(n395), .ZN(product[4]) );
  VHSR_OAI22_2 U391 ( .A1(n401), .A2(n400), .B1(n399), .B2(n398), .ZN(
        product[1]) );
  VHSR_AOI31_2 U392 ( .A1(n405), .A2(n404), .A3(n403), .B(n402), .ZN(
        product[2]) );
endmodule


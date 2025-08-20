
module mul8_38 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236, n237, n238,
         n239, n240, n241, n242, n243, n244, n245, n246, n247, n248, n249,
         n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, n260,
         n261, n262, n263, n264, n265, n266, n267, n268, n269, n270, n271,
         n272, n273, n274, n275, n276, n277, n278, n279, n280, n281, n282,
         n283, n284, n285, n286, n287, n288, n289, n290, n291, n292, n293,
         n294, n295, n296, n297, n298, n299, n300, n301, n302, n303, n304,
         n305, n306, n307, n308, n309, n310, n311, n312, n313, n314, n315,
         n316, n317, n318, n319, n320, n321, n322, n323, n324, n325, n326,
         n327, n328, n329, n330, n331, n332, n333, n334, n335, n336, n337,
         n338, n339, n340, n341, n342, n343, n344, n345, n346, n347, n348,
         n349, n350, n351, n352, n353, n354, n355, n356, n357, n358, n359,
         n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, n370,
         n371, n372, n373, n374, n375, n376, n377, n378, n379, n380, n381,
         n382, n383, n384, n385, n386, n387, n388, n389, n390, n391, n392,
         n393, n394, n395, n396, n397, n398, n399, n400, n401, n402, n403,
         n404, n405, n406, n407, n408;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U206 ( .A1(n245), .B1(n227), .ZN(n228) );
  VHSR_INOR2_2 U207 ( .A1(n235), .B1(n253), .ZN(n246) );
  VHSR_INOR2_2 U208 ( .A1(n379), .B1(n284), .ZN(n288) );
  VHSR_INOR2_2 U209 ( .A1(n231), .B1(n258), .ZN(n255) );
  VHSR_NOR2_1 U210 ( .A1(n273), .A2(n309), .ZN(n286) );
  VHSR_INAND2_2 U211 ( .A1(n328), .B1(n342), .ZN(n340) );
  VHSR_NOR2_1 U212 ( .A1(n295), .A2(n299), .ZN(n294) );
  VHSR_NOR2_1 U213 ( .A1(n238), .A2(n237), .ZN(n297) );
  VHSR_IOA21_2 U214 ( .A1(n325), .A2(n324), .B(n323), .ZN(n406) );
  VHSR_NOR2_1 U215 ( .A1(n345), .A2(n309), .ZN(n379) );
  VHSR_IN_2 U216 ( .I(n362), .ZN(product[13]) );
  VHSR_CLKN_1 U217 ( .I(n367), .ZN(n368) );
  VHSR_INAND3_1 U218 ( .A1(n394), .B1(n397), .B2(n396), .ZN(n367) );
  VHSR_INOR2_1 U219 ( .A1(n366), .B1(n365), .ZN(n397) );
  VHSR_INOR2_1 U220 ( .A1(n352), .B1(n351), .ZN(n364) );
  VHSR_NOR2_2 U221 ( .A1(n401), .A2(n400), .ZN(n399) );
  VHSR_INAND2_1 U222 ( .A1(n357), .B1(n355), .ZN(n358) );
  VHSR_INOR3_1 U223 ( .A1(n286), .B1(n278), .B2(n312), .ZN(n359) );
  VHSR_AD1_1 U224 ( .A(n386), .B(n385), .CI(n384), .CO(n381), .S(product[6])
         );
  VHSR_AD1_1 U225 ( .A(n380), .B(n379), .CI(n378), .CO(n375), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U226 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(product[10])
         );
  VHSR_AD1_1 U227 ( .A(n390), .B(n406), .CI(n389), .CO(n344), .S(product[3])
         );
  VHSR_AD1_1 U228 ( .A(n388), .B(n399), .CI(n387), .CO(n384), .S(product[5])
         );
  VHSR_AD1_1 U229 ( .A(n383), .B(n382), .CI(n381), .CO(n378), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U230 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(product[9])
         );
  VHSR_AD1_1 U231 ( .A(n371), .B(n370), .CI(n369), .CO(n391), .S(product[11])
         );
  VHSR_IN_2 U232 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U233 ( .I(a[3]), .ZN(n330) );
  VHSR_IN_2 U234 ( .I(b[6]), .ZN(n281) );
  VHSR_IN_2 U235 ( .I(a[2]), .ZN(n326) );
  VHSR_OAI22_2 U236 ( .A1(n281), .A2(n330), .B1(n280), .B2(n326), .ZN(n243) );
  VHSR_AOI22_2 U237 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n221) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_NAND3_2 U239 ( .A1(a[3]), .A2(b[5]), .A3(n263), .ZN(n220) );
  VHSR_CLKNAND2_2 U240 ( .A1(b[7]), .A2(a[2]), .ZN(n215) );
  VHSR_CLKNAND2_2 U241 ( .A1(b[6]), .A2(a[1]), .ZN(n217) );
  VHSR_OAI22_2 U242 ( .A1(n221), .A2(n220), .B1(n215), .B2(n217), .ZN(n222) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[6]), .A2(a[0]), .ZN(n262) );
  VHSR_IN_2 U244 ( .I(b[4]), .ZN(n309) );
  VHSR_IN_2 U245 ( .I(a[0]), .ZN(n404) );
  VHSR_OAI211_2 U246 ( .A1(n309), .A2(n404), .B(b[5]), .C(a[1]), .ZN(n261) );
  VHSR_MAOI222_2 U247 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_IN_2 U248 ( .I(b[5]), .ZN(n312) );
  VHSR_IN_2 U249 ( .I(a[1]), .ZN(n402) );
  VHSR_NOR4_2 U250 ( .A1(n309), .A2(n312), .A3(n402), .A4(n404), .ZN(n270) );
  VHSR_NAND4_2 U251 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n240) );
  VHSR_OAI22_2 U252 ( .A1(n309), .A2(n330), .B1(n312), .B2(n326), .ZN(n216) );
  VHSR_AND2_2 U253 ( .A1(n240), .A2(n216), .Z(n219) );
  VHSR_OAI21_2 U254 ( .A1(n280), .A2(n404), .B(n217), .ZN(n218) );
  VHSR_AND2_2 U255 ( .A1(n260), .A2(n257), .Z(n256) );
  VHSR_AD1_1 U256 ( .A(n270), .B(n219), .CI(n218), .CO(n249), .S(n257) );
  VHSR_AOI21_2 U257 ( .A1(n221), .A2(n220), .B(n222), .ZN(n252) );
  VHSR_OAI32_2 U258 ( .A1(n222), .A2(n256), .A3(n249), .B1(n252), .B2(n222), 
        .ZN(n241) );
  VHSR_CLKNAND2_2 U259 ( .A1(n241), .A2(n240), .ZN(n239) );
  VHSR_CLKNAND2_2 U260 ( .A1(n243), .A2(n239), .ZN(n236) );
  VHSR_NOR3_2 U261 ( .A1(n280), .A2(n330), .A3(n236), .ZN(n298) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[3]), .A2(a[7]), .ZN(n238) );
  VHSR_IN_2 U263 ( .I(b[3]), .ZN(n329) );
  VHSR_IN_2 U264 ( .I(a[6]), .ZN(n273) );
  VHSR_IN_2 U265 ( .I(a[7]), .ZN(n278) );
  VHSR_IN_2 U266 ( .I(b[2]), .ZN(n322) );
  VHSR_OAI22_2 U267 ( .A1(n329), .A2(n273), .B1(n278), .B2(n322), .ZN(n248) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[2]), .A2(a[4]), .ZN(n226) );
  VHSR_CLKNAND2_2 U269 ( .A1(a[6]), .A2(b[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[3]), .A2(a[6]), .ZN(n223) );
  VHSR_OAI22_2 U271 ( .A1(n238), .A2(n232), .B1(n223), .B2(n322), .ZN(n225) );
  VHSR_NOR3_2 U272 ( .A1(n278), .A2(n322), .A3(n232), .ZN(n224) );
  VHSR_AOI31_2 U273 ( .A1(a[5]), .A2(n226), .A3(n225), .B(n224), .ZN(n235) );
  VHSR_IN_2 U274 ( .I(n232), .ZN(n230) );
  VHSR_IN_2 U275 ( .I(a[4]), .ZN(n345) );
  VHSR_IN_2 U276 ( .I(a[5]), .ZN(n310) );
  VHSR_IN_2 U277 ( .I(b[1]), .ZN(n405) );
  VHSR_IN_2 U278 ( .I(b[0]), .ZN(n403) );
  VHSR_NOR4_2 U279 ( .A1(n345), .A2(n310), .A3(n405), .A4(n403), .ZN(n272) );
  VHSR_IN_2 U280 ( .I(n226), .ZN(n267) );
  VHSR_NAND3_2 U281 ( .A1(b[3]), .A2(n267), .A3(a[5]), .ZN(n245) );
  VHSR_AOI22_2 U282 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n227) );
  VHSR_MAOI222_2 U283 ( .A(n230), .B(n272), .C(n228), .ZN(n231) );
  VHSR_AOI211_2 U284 ( .A1(a[4]), .A2(b[0]), .B(n310), .C(n405), .ZN(n266) );
  VHSR_NOR2_1 U285 ( .A1(n273), .A2(n403), .ZN(n265) );
  VHSR_MAOI222_2 U286 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_OR2_2 U287 ( .A1(n272), .A2(n228), .Z(n229) );
  VHSR_OAI21_2 U288 ( .A1(n230), .A2(n229), .B(n231), .ZN(n259) );
  VHSR_NOR2_1 U289 ( .A1(n264), .A2(n259), .ZN(n258) );
  VHSR_CLKNAND2_2 U290 ( .A1(b[3]), .A2(a[5]), .ZN(n233) );
  VHSR_OAI22_2 U291 ( .A1(n267), .A2(n233), .B1(n278), .B2(n232), .ZN(n234) );
  VHSR_AOI32_2 U292 ( .A1(b[2]), .A2(n235), .A3(a[6]), .B1(n234), .B2(n235), 
        .ZN(n254) );
  VHSR_NOR2_1 U293 ( .A1(n255), .A2(n254), .ZN(n253) );
  VHSR_CLKNAND2_2 U294 ( .A1(n246), .A2(n245), .ZN(n244) );
  VHSR_CLKNAND2_2 U295 ( .A1(n248), .A2(n244), .ZN(n237) );
  VHSR_OAI32_2 U296 ( .A1(n298), .A2(n330), .A3(n280), .B1(n236), .B2(n298), 
        .ZN(n305) );
  VHSR_AOI21_2 U297 ( .A1(n238), .A2(n237), .B(n297), .ZN(n304) );
  VHSR_OAI21_2 U298 ( .A1(n241), .A2(n240), .B(n239), .ZN(n242) );
  VHSR_XNOR2_2 U299 ( .A1(n243), .A2(n242), .ZN(n308) );
  VHSR_OAI21_2 U300 ( .A1(n246), .A2(n245), .B(n244), .ZN(n247) );
  VHSR_XNOR2_2 U301 ( .A1(n248), .A2(n247), .ZN(n307) );
  VHSR_NOR2_1 U302 ( .A1(n256), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U303 ( .A1(n256), .A2(n249), .B1(n252), .B2(n251), .ZN(n250) );
  VHSR_OAI21_2 U304 ( .A1(n252), .A2(n251), .B(n250), .ZN(n316) );
  VHSR_AOI21_2 U305 ( .A1(n255), .A2(n254), .B(n253), .ZN(n315) );
  VHSR_IAO21_2 U306 ( .A1(n260), .A2(n257), .B(n256), .ZN(n319) );
  VHSR_AOI21_2 U307 ( .A1(n264), .A2(n259), .B(n258), .ZN(n318) );
  VHSR_AOI31_2 U308 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n334) );
  VHSR_OAI31_2 U309 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n268) );
  VHSR_IN_2 U310 ( .I(n268), .ZN(n333) );
  VHSR_CLKNAND2_2 U311 ( .A1(b[5]), .A2(a[0]), .ZN(n269) );
  VHSR_OAI32_2 U312 ( .A1(n270), .A2(n402), .A3(n309), .B1(n269), .B2(n270), 
        .ZN(n350) );
  VHSR_NOR2_1 U313 ( .A1(n403), .A2(n404), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U314 ( .A1(n379), .A2(product[0]), .ZN(n347) );
  VHSR_IN_2 U315 ( .I(n347), .ZN(n349) );
  VHSR_AOI22_2 U316 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n271) );
  VHSR_NOR2_1 U317 ( .A1(n272), .A2(n271), .ZN(n348) );
  VHSR_NOR2_1 U318 ( .A1(n273), .A2(n281), .ZN(n394) );
  VHSR_NOR2_1 U319 ( .A1(n345), .A2(n281), .ZN(n285) );
  VHSR_CLKNAND2_2 U320 ( .A1(a[5]), .A2(b[7]), .ZN(n275) );
  VHSR_CLKNAND2_2 U321 ( .A1(a[7]), .A2(b[5]), .ZN(n274) );
  VHSR_OAI22_2 U322 ( .A1(n285), .A2(n275), .B1(n286), .B2(n274), .ZN(n277) );
  VHSR_OR2_2 U323 ( .A1(n285), .A2(n286), .Z(n300) );
  VHSR_CLKNAND2_2 U324 ( .A1(a[5]), .A2(b[5]), .ZN(n284) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[7]), .A2(b[7]), .ZN(n395) );
  VHSR_NOR3_2 U326 ( .A1(n300), .A2(n284), .A3(n395), .ZN(n276) );
  VHSR_AOI31_2 U327 ( .A1(b[6]), .A2(a[6]), .A3(n277), .B(n276), .ZN(n352) );
  VHSR_OAI21_2 U328 ( .A1(n394), .A2(n277), .B(n352), .ZN(n293) );
  VHSR_AOI22_2 U329 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n279) );
  VHSR_NOR2_1 U330 ( .A1(n359), .A2(n279), .ZN(n289) );
  VHSR_NOR4_2 U331 ( .A1(n345), .A2(n310), .A3(n281), .A4(n280), .ZN(n357) );
  VHSR_AOI22_2 U332 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n282) );
  VHSR_NOR2_1 U333 ( .A1(n357), .A2(n282), .ZN(n287) );
  VHSR_IN_2 U334 ( .I(n283), .ZN(n295) );
  VHSR_NOR2_1 U335 ( .A1(n379), .A2(n284), .ZN(n301) );
  VHSR_AOI22_2 U336 ( .A1(n286), .A2(n285), .B1(n301), .B2(n300), .ZN(n299) );
  VHSR_AD1_1 U337 ( .A(n289), .B(n288), .CI(n287), .CO(n290), .S(n283) );
  VHSR_NOR2_1 U338 ( .A1(n294), .A2(n290), .ZN(n292) );
  VHSR_CLKNAND2_2 U339 ( .A1(n294), .A2(n290), .ZN(n291) );
  VHSR_NOR2_1 U340 ( .A1(n292), .A2(n293), .ZN(n351) );
  VHSR_AOI22_2 U341 ( .A1(n293), .A2(n292), .B1(n291), .B2(n351), .ZN(n392) );
  VHSR_AOI21_2 U342 ( .A1(n299), .A2(n295), .B(n294), .ZN(n371) );
  VHSR_AD1_1 U343 ( .A(n298), .B(n297), .CI(n296), .CO(n393), .S(n370) );
  VHSR_OAI21_2 U344 ( .A1(n301), .A2(n300), .B(n299), .ZN(n302) );
  VHSR_IN_2 U345 ( .I(n302), .ZN(n374) );
  VHSR_AD1_1 U346 ( .A(n305), .B(n304), .CI(n303), .CO(n296), .S(n373) );
  VHSR_AD1_1 U347 ( .A(n308), .B(n307), .CI(n306), .CO(n303), .S(n377) );
  VHSR_NOR2_1 U348 ( .A1(n310), .A2(n309), .ZN(n313) );
  VHSR_OAI21_2 U349 ( .A1(n345), .A2(n312), .B(n313), .ZN(n311) );
  VHSR_OAI31_2 U350 ( .A1(n345), .A2(n313), .A3(n312), .B(n311), .ZN(n376) );
  VHSR_AD1_1 U351 ( .A(n316), .B(n315), .CI(n314), .CO(n306), .S(n380) );
  VHSR_AD1_1 U352 ( .A(n319), .B(n318), .CI(n317), .CO(n314), .S(n383) );
  VHSR_NOR2_1 U353 ( .A1(n322), .A2(n326), .ZN(n338) );
  VHSR_IN_2 U354 ( .I(n338), .ZN(n331) );
  VHSR_NOR2_1 U355 ( .A1(n322), .A2(n330), .ZN(n321) );
  VHSR_OAI21_2 U356 ( .A1(n329), .A2(n326), .B(n321), .ZN(n320) );
  VHSR_OAI31_2 U357 ( .A1(n329), .A2(n321), .A3(n326), .B(n320), .ZN(n341) );
  VHSR_CLKNAND2_2 U358 ( .A1(b[3]), .A2(a[3]), .ZN(n337) );
  VHSR_CLKNAND2_2 U359 ( .A1(b[1]), .A2(a[1]), .ZN(n407) );
  VHSR_AOI22_2 U360 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n327) );
  VHSR_OAI22_2 U361 ( .A1(n337), .A2(n407), .B1(n331), .B2(n327), .ZN(n328) );
  VHSR_OAI22_2 U362 ( .A1(n329), .A2(n404), .B1(n322), .B2(n402), .ZN(n390) );
  VHSR_IN_2 U363 ( .I(n407), .ZN(n325) );
  VHSR_NOR2_1 U364 ( .A1(n403), .A2(n326), .ZN(n324) );
  VHSR_OAI211_2 U365 ( .A1(n324), .A2(n325), .B(b[2]), .C(a[0]), .ZN(n323) );
  VHSR_OAI22_2 U366 ( .A1(n405), .A2(n326), .B1(n403), .B2(n330), .ZN(n389) );
  VHSR_AOI21_2 U367 ( .A1(n327), .A2(n331), .B(n328), .ZN(n343) );
  VHSR_CLKNAND2_2 U368 ( .A1(n344), .A2(n343), .ZN(n342) );
  VHSR_CLKNAND2_2 U369 ( .A1(n341), .A2(n340), .ZN(n335) );
  VHSR_AOI211_2 U370 ( .A1(n331), .A2(n335), .B(n330), .C(n329), .ZN(n382) );
  VHSR_AD1_1 U371 ( .A(n334), .B(n333), .CI(n332), .CO(n317), .S(n386) );
  VHSR_IN_2 U372 ( .I(n335), .ZN(n339) );
  VHSR_CLKNAND2_2 U373 ( .A1(n339), .A2(n337), .ZN(n336) );
  VHSR_OAI31_2 U374 ( .A1(n338), .A2(n339), .A3(n337), .B(n336), .ZN(n385) );
  VHSR_IAO21_2 U375 ( .A1(n341), .A2(n340), .B(n339), .ZN(n388) );
  VHSR_OAI21_2 U376 ( .A1(n344), .A2(n343), .B(n342), .ZN(n401) );
  VHSR_NOR2_1 U377 ( .A1(n345), .A2(n403), .ZN(n346) );
  VHSR_AOI32_2 U378 ( .A1(b[4]), .A2(n347), .A3(a[0]), .B1(n346), .B2(n347), 
        .ZN(n400) );
  VHSR_AD1_1 U379 ( .A(n350), .B(n349), .CI(n348), .CO(n332), .S(n387) );
  VHSR_CLKNAND2_2 U380 ( .A1(a[6]), .A2(b[7]), .ZN(n354) );
  VHSR_AOI21_2 U381 ( .A1(a[7]), .A2(b[6]), .B(n354), .ZN(n353) );
  VHSR_AOI31_2 U382 ( .A1(a[7]), .A2(n354), .A3(b[6]), .B(n353), .ZN(n355) );
  VHSR_IN_2 U383 ( .I(n355), .ZN(n356) );
  VHSR_MAOI222_2 U384 ( .A(n359), .B(n357), .C(n356), .ZN(n366) );
  VHSR_OAI21_2 U385 ( .A1(n359), .A2(n358), .B(n366), .ZN(n363) );
  VHSR_CLKXOR2_2 U386 ( .A1(n364), .A2(n363), .Z(n360) );
  VHSR_CLKNAND2_2 U387 ( .A1(n361), .A2(n360), .ZN(n396) );
  VHSR_OAI21_2 U388 ( .A1(n361), .A2(n360), .B(n396), .ZN(n362) );
  VHSR_NOR2_1 U389 ( .A1(n364), .A2(n363), .ZN(n365) );
  VHSR_NOR2_1 U390 ( .A1(n395), .A2(n368), .ZN(product[15]) );
  VHSR_AD1_1 U391 ( .A(n393), .B(n392), .CI(n391), .CO(n361), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U392 ( .A1(n395), .A2(n394), .ZN(n398) );
  VHSR_XOR3_2 U393 ( .A1(n398), .A2(n397), .A3(n396), .Z(product[14]) );
  VHSR_AOI21_2 U394 ( .A1(n401), .A2(n400), .B(n399), .ZN(product[4]) );
  VHSR_OAI22_2 U395 ( .A1(n405), .A2(n404), .B1(n403), .B2(n402), .ZN(
        product[1]) );
  VHSR_AOI22_2 U396 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n408) );
  VHSR_AOI21_2 U397 ( .A1(n408), .A2(n407), .B(n406), .ZN(product[2]) );
endmodule


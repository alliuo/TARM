
module mul8_54 ( a, b, product );
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
         n404, n405;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U205 ( .A1(n321), .B1(n326), .ZN(n322) );
  VHSR_INOR2_2 U206 ( .A1(n363), .B1(n362), .ZN(n394) );
  VHSR_AD1_1 U207 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(product[9])
         );
  VHSR_AD1_1 U208 ( .A(n378), .B(n403), .CI(n377), .CO(n338), .S(product[3])
         );
  VHSR_AD1_1 U209 ( .A(n396), .B(n376), .CI(n375), .CO(n379), .S(product[5])
         );
  VHSR_AD1_1 U210 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U211 ( .A(n368), .B(n367), .CI(n366), .CO(n385), .S(product[10])
         );
  VHSR_AOI22_2 U212 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n250) );
  VHSR_IN_2 U213 ( .I(b[3]), .ZN(n319) );
  VHSR_IN_2 U214 ( .I(b[2]), .ZN(n316) );
  VHSR_IN_2 U215 ( .I(a[5]), .ZN(n286) );
  VHSR_IN_2 U216 ( .I(a[4]), .ZN(n285) );
  VHSR_NOR4_2 U217 ( .A1(n319), .A2(n316), .A3(n286), .A4(n285), .ZN(n248) );
  VHSR_IN_2 U218 ( .I(a[7]), .ZN(n281) );
  VHSR_IN_2 U219 ( .I(b[1]), .ZN(n402) );
  VHSR_NOR2_1 U220 ( .A1(n281), .A2(n402), .ZN(n216) );
  VHSR_AOI211_2 U221 ( .A1(b[2]), .A2(a[4]), .B(n319), .C(n286), .ZN(n217) );
  VHSR_CLKNAND2_2 U222 ( .A1(a[6]), .A2(b[2]), .ZN(n219) );
  VHSR_IN_2 U223 ( .I(n219), .ZN(n215) );
  VHSR_MAOI222_2 U224 ( .A(n216), .B(n217), .C(n215), .ZN(n229) );
  VHSR_AOI21_2 U225 ( .A1(b[1]), .A2(a[7]), .B(n217), .ZN(n220) );
  VHSR_IN_2 U226 ( .I(n229), .ZN(n218) );
  VHSR_AOI21_2 U227 ( .A1(n220), .A2(n219), .B(n218), .ZN(n257) );
  VHSR_CLKNAND2_2 U228 ( .A1(a[6]), .A2(b[1]), .ZN(n226) );
  VHSR_IN_2 U229 ( .I(n226), .ZN(n223) );
  VHSR_IN_2 U230 ( .I(b[0]), .ZN(n400) );
  VHSR_NOR4_2 U231 ( .A1(n286), .A2(n285), .A3(n402), .A4(n400), .ZN(n275) );
  VHSR_CLKNAND2_2 U232 ( .A1(b[2]), .A2(a[5]), .ZN(n222) );
  VHSR_CLKNAND2_2 U233 ( .A1(b[3]), .A2(a[4]), .ZN(n221) );
  VHSR_AOI21_2 U234 ( .A1(n222), .A2(n221), .B(n248), .ZN(n224) );
  VHSR_MAOI222_2 U235 ( .A(n223), .B(n275), .C(n224), .ZN(n228) );
  VHSR_CLKNAND2_2 U236 ( .A1(b[2]), .A2(a[4]), .ZN(n271) );
  VHSR_OAI21_2 U237 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n270) );
  VHSR_OAI211_2 U238 ( .A1(n285), .A2(n400), .B(a[5]), .C(b[1]), .ZN(n269) );
  VHSR_MAOI222_2 U239 ( .A(n271), .B(n270), .C(n269), .ZN(n268) );
  VHSR_NOR2_1 U240 ( .A1(n275), .A2(n224), .ZN(n227) );
  VHSR_IN_2 U241 ( .I(n228), .ZN(n225) );
  VHSR_AOI21_2 U242 ( .A1(n227), .A2(n226), .B(n225), .ZN(n260) );
  VHSR_CLKNAND2_2 U243 ( .A1(n268), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U244 ( .A1(n228), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U245 ( .A1(n257), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U246 ( .A1(n229), .A2(n255), .ZN(n247) );
  VHSR_NOR2_1 U247 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_NOR2_1 U248 ( .A1(n250), .A2(n246), .ZN(n239) );
  VHSR_AND3_2 U249 ( .A1(n239), .A2(b[3]), .A3(a[7]), .Z(n299) );
  VHSR_IN_2 U250 ( .I(b[7]), .ZN(n283) );
  VHSR_IN_2 U251 ( .I(a[3]), .ZN(n318) );
  VHSR_IN_2 U252 ( .I(b[6]), .ZN(n284) );
  VHSR_IN_2 U253 ( .I(a[2]), .ZN(n317) );
  VHSR_OAI22_2 U254 ( .A1(n284), .A2(n318), .B1(n283), .B2(n317), .ZN(n245) );
  VHSR_AOI22_2 U255 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n236) );
  VHSR_CLKNAND2_2 U256 ( .A1(b[4]), .A2(a[2]), .ZN(n267) );
  VHSR_NAND3_2 U257 ( .A1(a[3]), .A2(b[5]), .A3(n267), .ZN(n235) );
  VHSR_CLKNAND2_2 U258 ( .A1(b[7]), .A2(a[2]), .ZN(n230) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[6]), .A2(a[1]), .ZN(n232) );
  VHSR_OAI22_2 U260 ( .A1(n236), .A2(n235), .B1(n230), .B2(n232), .ZN(n237) );
  VHSR_IN_2 U261 ( .I(b[4]), .ZN(n339) );
  VHSR_IN_2 U262 ( .I(a[0]), .ZN(n401) );
  VHSR_OAI211_2 U263 ( .A1(n339), .A2(n401), .B(b[5]), .C(a[1]), .ZN(n266) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[6]), .A2(a[0]), .ZN(n265) );
  VHSR_MAOI222_2 U265 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NAND4_2 U266 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n242) );
  VHSR_IN_2 U267 ( .I(b[5]), .ZN(n280) );
  VHSR_OAI22_2 U268 ( .A1(n339), .A2(n318), .B1(n280), .B2(n317), .ZN(n231) );
  VHSR_AND2_2 U269 ( .A1(n242), .A2(n231), .Z(n234) );
  VHSR_OAI21_2 U270 ( .A1(n283), .A2(n401), .B(n232), .ZN(n233) );
  VHSR_IN_2 U271 ( .I(a[1]), .ZN(n399) );
  VHSR_NOR4_2 U272 ( .A1(n339), .A2(n280), .A3(n399), .A4(n401), .ZN(n273) );
  VHSR_AND2_2 U273 ( .A1(n264), .A2(n263), .Z(n262) );
  VHSR_AD1_1 U274 ( .A(n234), .B(n233), .CI(n273), .CO(n251), .S(n263) );
  VHSR_AOI21_2 U275 ( .A1(n236), .A2(n235), .B(n237), .ZN(n254) );
  VHSR_OAI32_2 U276 ( .A1(n237), .A2(n262), .A3(n251), .B1(n254), .B2(n237), 
        .ZN(n243) );
  VHSR_CLKNAND2_2 U277 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U278 ( .A1(n245), .A2(n241), .ZN(n240) );
  VHSR_NOR3_2 U279 ( .A1(n283), .A2(n318), .A3(n240), .ZN(n298) );
  VHSR_NOR2_1 U280 ( .A1(n319), .A2(n281), .ZN(n238) );
  VHSR_IAO21_2 U281 ( .A1(n239), .A2(n238), .B(n299), .ZN(n302) );
  VHSR_OAI32_2 U282 ( .A1(n298), .A2(n318), .A3(n283), .B1(n240), .B2(n298), 
        .ZN(n301) );
  VHSR_OAI21_2 U283 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U284 ( .A1(n245), .A2(n244), .ZN(n309) );
  VHSR_AOI21_2 U285 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U286 ( .A1(n250), .A2(n249), .ZN(n308) );
  VHSR_NOR2_1 U287 ( .A1(n262), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U288 ( .A1(n262), .A2(n251), .B1(n254), .B2(n253), .ZN(n252) );
  VHSR_OAI21_2 U289 ( .A1(n254), .A2(n253), .B(n252), .ZN(n314) );
  VHSR_OAI21_2 U290 ( .A1(n257), .A2(n256), .B(n255), .ZN(n258) );
  VHSR_IN_2 U291 ( .I(n258), .ZN(n313) );
  VHSR_OAI21_2 U292 ( .A1(n268), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U293 ( .I(n261), .ZN(n329) );
  VHSR_IAO21_2 U294 ( .A1(n264), .A2(n263), .B(n262), .ZN(n328) );
  VHSR_AOI31_2 U295 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n336) );
  VHSR_AOI31_2 U296 ( .A1(n271), .A2(n270), .A3(n269), .B(n268), .ZN(n335) );
  VHSR_CLKNAND2_2 U297 ( .A1(b[5]), .A2(a[0]), .ZN(n272) );
  VHSR_OAI32_2 U298 ( .A1(n273), .A2(n399), .A3(n339), .B1(n272), .B2(n273), 
        .ZN(n344) );
  VHSR_NOR2_1 U299 ( .A1(n285), .A2(n339), .ZN(n373) );
  VHSR_NOR2_1 U300 ( .A1(n400), .A2(n401), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U301 ( .A1(n373), .A2(product[0]), .ZN(n341) );
  VHSR_IN_2 U302 ( .I(n341), .ZN(n343) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[1]), .ZN(n274) );
  VHSR_OAI32_2 U304 ( .A1(n275), .A2(n286), .A3(n400), .B1(n274), .B2(n275), 
        .ZN(n342) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[6]), .A2(b[6]), .ZN(n364) );
  VHSR_IN_2 U306 ( .I(n364), .ZN(n391) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[6]), .A2(b[4]), .ZN(n306) );
  VHSR_NAND3_2 U308 ( .A1(a[7]), .A2(b[5]), .A3(n306), .ZN(n277) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[4]), .A2(b[6]), .ZN(n305) );
  VHSR_NAND3_2 U310 ( .A1(b[7]), .A2(a[5]), .A3(n305), .ZN(n276) );
  VHSR_CLKNAND2_2 U311 ( .A1(n277), .A2(n276), .ZN(n279) );
  VHSR_MAOI222_2 U312 ( .A(n364), .B(n277), .C(n276), .ZN(n348) );
  VHSR_IN_2 U313 ( .I(n348), .ZN(n278) );
  VHSR_OAI21_2 U314 ( .A1(n391), .A2(n279), .B(n278), .ZN(n294) );
  VHSR_IN_2 U315 ( .I(n373), .ZN(n288) );
  VHSR_NOR3_2 U316 ( .A1(n286), .A2(n280), .A3(n288), .ZN(n310) );
  VHSR_NOR3_2 U317 ( .A1(n281), .A2(n306), .A3(n280), .ZN(n356) );
  VHSR_AOI22_2 U318 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n282) );
  VHSR_NOR2_1 U319 ( .A1(n356), .A2(n282), .ZN(n290) );
  VHSR_NOR4_2 U320 ( .A1(n286), .A2(n285), .A3(n284), .A4(n283), .ZN(n354) );
  VHSR_AOI22_2 U321 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n287) );
  VHSR_NOR2_1 U322 ( .A1(n354), .A2(n287), .ZN(n289) );
  VHSR_NAND3_2 U323 ( .A1(b[5]), .A2(a[5]), .A3(n288), .ZN(n304) );
  VHSR_MAOI222_2 U324 ( .A(n306), .B(n305), .C(n304), .ZN(n303) );
  VHSR_AND2_2 U325 ( .A1(n296), .A2(n303), .Z(n295) );
  VHSR_AD1_1 U326 ( .A(n310), .B(n290), .CI(n289), .CO(n291), .S(n296) );
  VHSR_NOR2_1 U327 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U328 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_NOR2_1 U329 ( .A1(n293), .A2(n294), .ZN(n349) );
  VHSR_AOI22_2 U330 ( .A1(n294), .A2(n293), .B1(n292), .B2(n349), .ZN(n389) );
  VHSR_IAO21_2 U331 ( .A1(n296), .A2(n303), .B(n295), .ZN(n387) );
  VHSR_AD1_1 U332 ( .A(n299), .B(n298), .CI(n297), .CO(n390), .S(n386) );
  VHSR_AD1_1 U333 ( .A(n302), .B(n301), .CI(n300), .CO(n297), .S(n368) );
  VHSR_AOI31_2 U334 ( .A1(n306), .A2(n305), .A3(n304), .B(n303), .ZN(n367) );
  VHSR_AD1_1 U335 ( .A(n309), .B(n308), .CI(n307), .CO(n300), .S(n371) );
  VHSR_AOI22_2 U336 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n311) );
  VHSR_NOR2_1 U337 ( .A1(n311), .A2(n310), .ZN(n370) );
  VHSR_AD1_1 U338 ( .A(n314), .B(n313), .CI(n312), .CO(n307), .S(n374) );
  VHSR_NOR2_1 U339 ( .A1(n316), .A2(n317), .ZN(n333) );
  VHSR_NOR4_2 U340 ( .A1(n319), .A2(n316), .A3(n399), .A4(n401), .ZN(n347) );
  VHSR_CLKNAND2_2 U341 ( .A1(b[2]), .A2(a[1]), .ZN(n315) );
  VHSR_OAI32_2 U342 ( .A1(n347), .A2(n401), .A3(n319), .B1(n315), .B2(n347), 
        .ZN(n378) );
  VHSR_AOI22_2 U343 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n405) );
  VHSR_NOR3_2 U344 ( .A1(n405), .A2(n401), .A3(n316), .ZN(n403) );
  VHSR_OAI22_2 U345 ( .A1(n402), .A2(n317), .B1(n400), .B2(n318), .ZN(n377) );
  VHSR_IN_2 U346 ( .I(n338), .ZN(n324) );
  VHSR_NOR2_1 U347 ( .A1(n402), .A2(n318), .ZN(n320) );
  VHSR_AOI211_2 U348 ( .A1(b[2]), .A2(a[0]), .B(n319), .C(n399), .ZN(n321) );
  VHSR_MAOI222_2 U349 ( .A(n320), .B(n321), .C(n333), .ZN(n323) );
  VHSR_IN_2 U350 ( .I(n333), .ZN(n326) );
  VHSR_AOI32_2 U351 ( .A1(a[3]), .A2(n323), .A3(b[1]), .B1(n322), .B2(n323), 
        .ZN(n337) );
  VHSR_OAI21_2 U352 ( .A1(n324), .A2(n337), .B(n323), .ZN(n346) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[3]), .A2(a[3]), .ZN(n331) );
  VHSR_AOI22_2 U354 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n325) );
  VHSR_IAO21_2 U355 ( .A1(n331), .A2(n326), .B(n325), .ZN(n345) );
  VHSR_IAO21_2 U356 ( .A1(n333), .A2(n332), .B(n331), .ZN(n384) );
  VHSR_AD1_1 U357 ( .A(n329), .B(n328), .CI(n327), .CO(n312), .S(n383) );
  VHSR_OAI21_2 U358 ( .A1(n333), .A2(n331), .B(n332), .ZN(n330) );
  VHSR_OAI31_2 U359 ( .A1(n333), .A2(n332), .A3(n331), .B(n330), .ZN(n381) );
  VHSR_AD1_1 U360 ( .A(n336), .B(n335), .CI(n334), .CO(n327), .S(n380) );
  VHSR_CLKXOR2_2 U361 ( .A1(n338), .A2(n337), .Z(n398) );
  VHSR_NOR2_1 U362 ( .A1(n339), .A2(n401), .ZN(n340) );
  VHSR_AOI32_2 U363 ( .A1(b[0]), .A2(n341), .A3(a[4]), .B1(n340), .B2(n341), 
        .ZN(n397) );
  VHSR_NOR2_1 U364 ( .A1(n398), .A2(n397), .ZN(n396) );
  VHSR_AD1_1 U365 ( .A(n344), .B(n343), .CI(n342), .CO(n334), .S(n376) );
  VHSR_AD1_1 U366 ( .A(n347), .B(n346), .CI(n345), .CO(n332), .S(n375) );
  VHSR_NOR2_1 U367 ( .A1(n349), .A2(n348), .ZN(n361) );
  VHSR_CLKNAND2_2 U368 ( .A1(a[7]), .A2(b[6]), .ZN(n351) );
  VHSR_AOI21_2 U369 ( .A1(a[6]), .A2(b[7]), .B(n351), .ZN(n350) );
  VHSR_AOI31_2 U370 ( .A1(a[6]), .A2(n351), .A3(b[7]), .B(n350), .ZN(n352) );
  VHSR_IN_2 U371 ( .I(n352), .ZN(n353) );
  VHSR_OR2_2 U372 ( .A1(n354), .A2(n353), .Z(n355) );
  VHSR_MAOI222_2 U373 ( .A(n356), .B(n354), .C(n353), .ZN(n363) );
  VHSR_OAI21_2 U374 ( .A1(n356), .A2(n355), .B(n363), .ZN(n360) );
  VHSR_CLKXOR2_2 U375 ( .A1(n361), .A2(n360), .Z(n357) );
  VHSR_CLKNAND2_2 U376 ( .A1(n358), .A2(n357), .ZN(n393) );
  VHSR_OAI21_2 U377 ( .A1(n358), .A2(n357), .B(n393), .ZN(n359) );
  VHSR_IN_2 U378 ( .I(n359), .ZN(product[13]) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[7]), .A2(b[7]), .ZN(n392) );
  VHSR_NOR2_1 U380 ( .A1(n361), .A2(n360), .ZN(n362) );
  VHSR_AND3_2 U381 ( .A1(n394), .A2(n364), .A3(n393), .Z(n365) );
  VHSR_NOR2_1 U382 ( .A1(n392), .A2(n365), .ZN(product[15]) );
  VHSR_AD1_1 U383 ( .A(n381), .B(n380), .CI(n379), .CO(n382), .S(product[6])
         );
  VHSR_AD1_1 U384 ( .A(n384), .B(n383), .CI(n382), .CO(n372), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U385 ( .A(n387), .B(n386), .CI(n385), .CO(n388), .S(product[11])
         );
  VHSR_AD1_1 U386 ( .A(n390), .B(n389), .CI(n388), .CO(n358), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U387 ( .A1(n392), .A2(n391), .ZN(n395) );
  VHSR_XOR3_2 U388 ( .A1(n395), .A2(n394), .A3(n393), .Z(product[14]) );
  VHSR_AOI21_2 U389 ( .A1(n398), .A2(n397), .B(n396), .ZN(product[4]) );
  VHSR_OAI22_2 U390 ( .A1(n402), .A2(n401), .B1(n400), .B2(n399), .ZN(
        product[1]) );
  VHSR_CLKNAND2_2 U391 ( .A1(b[2]), .A2(a[0]), .ZN(n404) );
  VHSR_AOI21_2 U392 ( .A1(n405), .A2(n404), .B(n403), .ZN(product[2]) );
endmodule


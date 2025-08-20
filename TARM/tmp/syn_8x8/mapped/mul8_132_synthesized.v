
module mul8_132 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n212, n213,
         n214, n215, n216, n217, n218, n219, n220, n221, n222, n223, n224,
         n225, n226, n227, n228, n229, n230, n231, n232, n233, n234, n235,
         n236, n237, n238, n239, n240, n241, n242, n243, n244, n245, n246,
         n247, n248, n249, n250, n251, n252, n253, n254, n255, n256, n257,
         n258, n259, n260, n261, n262, n263, n264, n265, n266, n267, n268,
         n269, n270, n271, n272, n273, n274, n275, n276, n277, n278, n279,
         n280, n281, n282, n283, n284, n285, n286, n287, n288, n289, n290,
         n291, n292, n293, n294, n295, n296, n297, n298, n299, n300, n301,
         n302, n303, n304, n305, n306, n307, n308, n309, n310, n311, n312,
         n313, n314, n315, n316, n317, n318, n319, n320, n321, n322, n323,
         n324, n325, n326, n327, n328, n329, n330, n331, n332, n333, n334,
         n335, n336, n337, n338, n339, n340, n341, n342, n343, n344, n345,
         n346, n347, n348, n349, n350, n351, n352, n353, n354, n355, n356,
         n357, n358, n359, n360, n361, n362, n363, n364, n365, n366, n367,
         n368, n369, n370, n371, n372, n373, n374, n375, n376, n377, n378,
         n379, n380, n381, n382, n383, n384, n385, n386, n387, n388, n389,
         n390, n391, n392, n393, n394, n395, n396, n397, n398, n399, n400,
         n401, n402;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U204 ( .A1(n238), .B1(n222), .ZN(n224) );
  VHSR_NOR2_1 U205 ( .A1(n314), .A2(n307), .ZN(n260) );
  VHSR_INAND2_2 U206 ( .A1(n355), .B1(n353), .ZN(n356) );
  VHSR_NOR2_1 U207 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_INOR2_2 U208 ( .A1(n226), .B1(n249), .ZN(n248) );
  VHSR_NOR2_1 U209 ( .A1(n266), .A2(n343), .ZN(n280) );
  VHSR_INAND2_2 U210 ( .A1(n322), .B1(n339), .ZN(n337) );
  VHSR_NOR2_1 U211 ( .A1(n289), .A2(n293), .ZN(n288) );
  VHSR_NOR2_1 U212 ( .A1(n231), .A2(n230), .ZN(n291) );
  VHSR_IOA21_2 U213 ( .A1(n317), .A2(n316), .B(n315), .ZN(n400) );
  VHSR_INOR2_2 U214 ( .A1(n364), .B1(n363), .ZN(n395) );
  VHSR_IN_2 U215 ( .I(n360), .ZN(product[13]) );
  VHSR_CLKN_1 U216 ( .I(n365), .ZN(n366) );
  VHSR_INAND3_1 U217 ( .A1(n392), .B1(n395), .B2(n394), .ZN(n365) );
  VHSR_INOR2_1 U218 ( .A1(n228), .B1(n246), .ZN(n239) );
  VHSR_INOR2_1 U219 ( .A1(n350), .B1(n349), .ZN(n362) );
  VHSR_NOR2_2 U220 ( .A1(n399), .A2(n398), .ZN(n397) );
  VHSR_INOR3_1 U221 ( .A1(n280), .B1(n271), .B2(n305), .ZN(n357) );
  VHSR_AD1_1 U222 ( .A(n384), .B(n383), .CI(n382), .CO(n379), .S(product[6])
         );
  VHSR_AD1_1 U223 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(product[10])
         );
  VHSR_AD1_1 U225 ( .A(n388), .B(n400), .CI(n387), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n386), .B(n397), .CI(n385), .CO(n382), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U228 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(product[9])
         );
  VHSR_AD1_1 U229 ( .A(n369), .B(n368), .CI(n367), .CO(n389), .S(product[11])
         );
  VHSR_IN_2 U230 ( .I(b[0]), .ZN(n318) );
  VHSR_IN_2 U231 ( .I(a[1]), .ZN(n313) );
  VHSR_NOR2_1 U232 ( .A1(n318), .A2(n313), .ZN(product[1]) );
  VHSR_IN_2 U233 ( .I(b[1]), .ZN(n320) );
  VHSR_IN_2 U234 ( .I(a[0]), .ZN(n342) );
  VHSR_NOR2_1 U235 ( .A1(n320), .A2(n342), .ZN(product[0]) );
  VHSR_IN_2 U236 ( .I(b[7]), .ZN(n274) );
  VHSR_IN_2 U237 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U238 ( .I(b[6]), .ZN(n275) );
  VHSR_IN_2 U239 ( .I(a[2]), .ZN(n319) );
  VHSR_OAI22_2 U240 ( .A1(n275), .A2(n324), .B1(n274), .B2(n319), .ZN(n236) );
  VHSR_AOI22_2 U241 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n218) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[4]), .A2(a[2]), .ZN(n256) );
  VHSR_NAND3_2 U243 ( .A1(a[3]), .A2(b[5]), .A3(n256), .ZN(n217) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[7]), .A2(a[2]), .ZN(n212) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[6]), .A2(a[1]), .ZN(n214) );
  VHSR_OAI22_2 U246 ( .A1(n218), .A2(n217), .B1(n212), .B2(n214), .ZN(n219) );
  VHSR_IN_2 U247 ( .I(b[4]), .ZN(n343) );
  VHSR_OAI211_2 U248 ( .A1(n343), .A2(n342), .B(b[5]), .C(a[1]), .ZN(n255) );
  VHSR_CLKNAND2_2 U249 ( .A1(b[6]), .A2(a[0]), .ZN(n254) );
  VHSR_MAOI222_2 U250 ( .A(n256), .B(n255), .C(n254), .ZN(n253) );
  VHSR_NAND4_2 U251 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n233) );
  VHSR_IN_2 U252 ( .I(b[5]), .ZN(n305) );
  VHSR_OAI22_2 U253 ( .A1(n343), .A2(n324), .B1(n305), .B2(n319), .ZN(n213) );
  VHSR_AND2_2 U254 ( .A1(n233), .A2(n213), .Z(n216) );
  VHSR_OAI21_2 U255 ( .A1(n274), .A2(n342), .B(n214), .ZN(n215) );
  VHSR_NOR4_2 U256 ( .A1(n343), .A2(n305), .A3(n313), .A4(n342), .ZN(n263) );
  VHSR_AND2_2 U257 ( .A1(n253), .A2(n252), .Z(n251) );
  VHSR_AD1_1 U258 ( .A(n216), .B(n215), .CI(n263), .CO(n242), .S(n252) );
  VHSR_AOI21_2 U259 ( .A1(n218), .A2(n217), .B(n219), .ZN(n245) );
  VHSR_OAI32_2 U260 ( .A1(n219), .A2(n251), .A3(n242), .B1(n245), .B2(n219), 
        .ZN(n234) );
  VHSR_CLKNAND2_2 U261 ( .A1(n234), .A2(n233), .ZN(n232) );
  VHSR_CLKNAND2_2 U262 ( .A1(n236), .A2(n232), .ZN(n229) );
  VHSR_NOR3_2 U263 ( .A1(n274), .A2(n324), .A3(n229), .ZN(n292) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[3]), .A2(a[7]), .ZN(n231) );
  VHSR_IN_2 U265 ( .I(b[3]), .ZN(n323) );
  VHSR_IN_2 U266 ( .I(a[6]), .ZN(n266) );
  VHSR_IN_2 U267 ( .I(a[7]), .ZN(n271) );
  VHSR_IN_2 U268 ( .I(b[2]), .ZN(n314) );
  VHSR_OAI22_2 U269 ( .A1(n323), .A2(n266), .B1(n271), .B2(n314), .ZN(n241) );
  VHSR_IN_2 U270 ( .I(a[4]), .ZN(n307) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[3]), .A2(a[5]), .ZN(n220) );
  VHSR_OAI22_2 U272 ( .A1(n260), .A2(n220), .B1(n271), .B2(n320), .ZN(n227) );
  VHSR_IN_2 U273 ( .I(a[5]), .ZN(n303) );
  VHSR_NOR4_2 U274 ( .A1(n260), .A2(n231), .A3(n303), .A4(n320), .ZN(n221) );
  VHSR_AOI31_2 U275 ( .A1(b[2]), .A2(a[6]), .A3(n227), .B(n221), .ZN(n228) );
  VHSR_NOR2_1 U276 ( .A1(n266), .A2(n320), .ZN(n223) );
  VHSR_NOR4_2 U277 ( .A1(n307), .A2(n303), .A3(n320), .A4(n318), .ZN(n265) );
  VHSR_NAND3_2 U278 ( .A1(b[3]), .A2(n260), .A3(a[5]), .ZN(n238) );
  VHSR_AOI22_2 U279 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n222) );
  VHSR_MAOI222_2 U280 ( .A(n223), .B(n265), .C(n224), .ZN(n226) );
  VHSR_AOI211_2 U281 ( .A1(a[4]), .A2(b[0]), .B(n303), .C(n320), .ZN(n259) );
  VHSR_AOI21_2 U282 ( .A1(n271), .A2(n266), .B(n318), .ZN(n258) );
  VHSR_MAOI222_2 U283 ( .A(n260), .B(n259), .C(n258), .ZN(n257) );
  VHSR_OR2_2 U284 ( .A1(n265), .A2(n224), .Z(n225) );
  VHSR_AOI32_2 U285 ( .A1(b[1]), .A2(n226), .A3(a[6]), .B1(n225), .B2(n226), 
        .ZN(n250) );
  VHSR_NOR2_1 U286 ( .A1(n257), .A2(n250), .ZN(n249) );
  VHSR_AOI32_2 U287 ( .A1(b[2]), .A2(n228), .A3(a[6]), .B1(n227), .B2(n228), 
        .ZN(n247) );
  VHSR_CLKNAND2_2 U288 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U289 ( .A1(n241), .A2(n237), .ZN(n230) );
  VHSR_OAI32_2 U290 ( .A1(n292), .A2(n324), .A3(n274), .B1(n229), .B2(n292), 
        .ZN(n299) );
  VHSR_AOI21_2 U291 ( .A1(n231), .A2(n230), .B(n291), .ZN(n298) );
  VHSR_OAI21_2 U292 ( .A1(n234), .A2(n233), .B(n232), .ZN(n235) );
  VHSR_XNOR2_2 U293 ( .A1(n236), .A2(n235), .ZN(n302) );
  VHSR_OAI21_2 U294 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U295 ( .A1(n241), .A2(n240), .ZN(n301) );
  VHSR_NOR2_1 U296 ( .A1(n251), .A2(n242), .ZN(n244) );
  VHSR_AOI22_2 U297 ( .A1(n251), .A2(n242), .B1(n245), .B2(n244), .ZN(n243) );
  VHSR_OAI21_2 U298 ( .A1(n245), .A2(n244), .B(n243), .ZN(n310) );
  VHSR_AOI21_2 U299 ( .A1(n248), .A2(n247), .B(n246), .ZN(n309) );
  VHSR_AOI21_2 U300 ( .A1(n257), .A2(n250), .B(n249), .ZN(n328) );
  VHSR_IAO21_2 U301 ( .A1(n253), .A2(n252), .B(n251), .ZN(n327) );
  VHSR_AOI31_2 U302 ( .A1(n256), .A2(n255), .A3(n254), .B(n253), .ZN(n331) );
  VHSR_OAI31_2 U303 ( .A1(n260), .A2(n259), .A3(n258), .B(n257), .ZN(n261) );
  VHSR_IN_2 U304 ( .I(n261), .ZN(n330) );
  VHSR_CLKNAND2_2 U305 ( .A1(b[5]), .A2(a[0]), .ZN(n262) );
  VHSR_OAI32_2 U306 ( .A1(n263), .A2(n313), .A3(n343), .B1(n262), .B2(n263), 
        .ZN(n348) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[4]), .A2(b[4]), .ZN(n273) );
  VHSR_IN_2 U308 ( .I(n273), .ZN(n377) );
  VHSR_NAND3_2 U309 ( .A1(b[0]), .A2(n377), .A3(a[0]), .ZN(n345) );
  VHSR_IN_2 U310 ( .I(n345), .ZN(n347) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[5]), .A2(b[0]), .ZN(n264) );
  VHSR_OAI32_2 U312 ( .A1(n265), .A2(n320), .A3(n307), .B1(n264), .B2(n265), 
        .ZN(n346) );
  VHSR_NOR2_1 U313 ( .A1(n266), .A2(n275), .ZN(n392) );
  VHSR_NOR2_1 U314 ( .A1(n307), .A2(n275), .ZN(n279) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[5]), .A2(b[7]), .ZN(n268) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[7]), .A2(b[5]), .ZN(n267) );
  VHSR_OAI22_2 U317 ( .A1(n279), .A2(n268), .B1(n280), .B2(n267), .ZN(n270) );
  VHSR_OR2_2 U318 ( .A1(n279), .A2(n280), .Z(n294) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[5]), .A2(b[5]), .ZN(n278) );
  VHSR_CLKNAND2_2 U320 ( .A1(a[7]), .A2(b[7]), .ZN(n393) );
  VHSR_NOR3_2 U321 ( .A1(n294), .A2(n278), .A3(n393), .ZN(n269) );
  VHSR_AOI31_2 U322 ( .A1(b[6]), .A2(a[6]), .A3(n270), .B(n269), .ZN(n350) );
  VHSR_OAI21_2 U323 ( .A1(n392), .A2(n270), .B(n350), .ZN(n287) );
  VHSR_AOI22_2 U324 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n272) );
  VHSR_NOR2_1 U325 ( .A1(n357), .A2(n272), .ZN(n283) );
  VHSR_NOR2_1 U326 ( .A1(n278), .A2(n273), .ZN(n282) );
  VHSR_NOR4_2 U327 ( .A1(n307), .A2(n303), .A3(n275), .A4(n274), .ZN(n355) );
  VHSR_AOI22_2 U328 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n276) );
  VHSR_NOR2_1 U329 ( .A1(n355), .A2(n276), .ZN(n281) );
  VHSR_IN_2 U330 ( .I(n277), .ZN(n289) );
  VHSR_NOR2_1 U331 ( .A1(n377), .A2(n278), .ZN(n295) );
  VHSR_AOI22_2 U332 ( .A1(n280), .A2(n279), .B1(n295), .B2(n294), .ZN(n293) );
  VHSR_AD1_1 U333 ( .A(n283), .B(n282), .CI(n281), .CO(n284), .S(n277) );
  VHSR_NOR2_1 U334 ( .A1(n288), .A2(n284), .ZN(n286) );
  VHSR_CLKNAND2_2 U335 ( .A1(n288), .A2(n284), .ZN(n285) );
  VHSR_NOR2_1 U336 ( .A1(n286), .A2(n287), .ZN(n349) );
  VHSR_AOI22_2 U337 ( .A1(n287), .A2(n286), .B1(n285), .B2(n349), .ZN(n390) );
  VHSR_AOI21_2 U338 ( .A1(n293), .A2(n289), .B(n288), .ZN(n369) );
  VHSR_AD1_1 U339 ( .A(n292), .B(n291), .CI(n290), .CO(n391), .S(n368) );
  VHSR_OAI21_2 U340 ( .A1(n295), .A2(n294), .B(n293), .ZN(n296) );
  VHSR_IN_2 U341 ( .I(n296), .ZN(n372) );
  VHSR_AD1_1 U342 ( .A(n299), .B(n298), .CI(n297), .CO(n290), .S(n371) );
  VHSR_AD1_1 U343 ( .A(n302), .B(n301), .CI(n300), .CO(n297), .S(n375) );
  VHSR_NOR2_1 U344 ( .A1(n303), .A2(n343), .ZN(n306) );
  VHSR_OAI21_2 U345 ( .A1(n307), .A2(n305), .B(n306), .ZN(n304) );
  VHSR_OAI31_2 U346 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n374) );
  VHSR_AD1_1 U347 ( .A(n310), .B(n309), .CI(n308), .CO(n300), .S(n378) );
  VHSR_NOR2_1 U348 ( .A1(n314), .A2(n319), .ZN(n335) );
  VHSR_IN_2 U349 ( .I(n335), .ZN(n325) );
  VHSR_NOR2_1 U350 ( .A1(n314), .A2(n324), .ZN(n312) );
  VHSR_OAI21_2 U351 ( .A1(n323), .A2(n319), .B(n312), .ZN(n311) );
  VHSR_OAI31_2 U352 ( .A1(n323), .A2(n312), .A3(n319), .B(n311), .ZN(n338) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[3]), .A2(a[3]), .ZN(n334) );
  VHSR_CLKNAND2_2 U354 ( .A1(b[1]), .A2(a[1]), .ZN(n401) );
  VHSR_AOI22_2 U355 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n321) );
  VHSR_OAI22_2 U356 ( .A1(n334), .A2(n401), .B1(n325), .B2(n321), .ZN(n322) );
  VHSR_OAI22_2 U357 ( .A1(n323), .A2(n342), .B1(n314), .B2(n313), .ZN(n388) );
  VHSR_IN_2 U358 ( .I(n401), .ZN(n317) );
  VHSR_NOR2_1 U359 ( .A1(n318), .A2(n319), .ZN(n316) );
  VHSR_OAI211_2 U360 ( .A1(n316), .A2(n317), .B(b[2]), .C(a[0]), .ZN(n315) );
  VHSR_OAI22_2 U361 ( .A1(n320), .A2(n319), .B1(n318), .B2(n324), .ZN(n387) );
  VHSR_AOI21_2 U362 ( .A1(n321), .A2(n325), .B(n322), .ZN(n340) );
  VHSR_CLKNAND2_2 U363 ( .A1(n341), .A2(n340), .ZN(n339) );
  VHSR_CLKNAND2_2 U364 ( .A1(n338), .A2(n337), .ZN(n332) );
  VHSR_AOI211_2 U365 ( .A1(n325), .A2(n332), .B(n324), .C(n323), .ZN(n381) );
  VHSR_AD1_1 U366 ( .A(n328), .B(n327), .CI(n326), .CO(n308), .S(n380) );
  VHSR_AD1_1 U367 ( .A(n331), .B(n330), .CI(n329), .CO(n326), .S(n384) );
  VHSR_IN_2 U368 ( .I(n332), .ZN(n336) );
  VHSR_CLKNAND2_2 U369 ( .A1(n336), .A2(n334), .ZN(n333) );
  VHSR_OAI31_2 U370 ( .A1(n335), .A2(n336), .A3(n334), .B(n333), .ZN(n383) );
  VHSR_IAO21_2 U371 ( .A1(n338), .A2(n337), .B(n336), .ZN(n386) );
  VHSR_OAI21_2 U372 ( .A1(n341), .A2(n340), .B(n339), .ZN(n399) );
  VHSR_NOR2_1 U373 ( .A1(n343), .A2(n342), .ZN(n344) );
  VHSR_AOI32_2 U374 ( .A1(b[0]), .A2(n345), .A3(a[4]), .B1(n344), .B2(n345), 
        .ZN(n398) );
  VHSR_AD1_1 U375 ( .A(n348), .B(n347), .CI(n346), .CO(n329), .S(n385) );
  VHSR_CLKNAND2_2 U376 ( .A1(a[6]), .A2(b[7]), .ZN(n352) );
  VHSR_AOI21_2 U377 ( .A1(a[7]), .A2(b[6]), .B(n352), .ZN(n351) );
  VHSR_AOI31_2 U378 ( .A1(a[7]), .A2(n352), .A3(b[6]), .B(n351), .ZN(n353) );
  VHSR_IN_2 U379 ( .I(n353), .ZN(n354) );
  VHSR_MAOI222_2 U380 ( .A(n357), .B(n355), .C(n354), .ZN(n364) );
  VHSR_OAI21_2 U381 ( .A1(n357), .A2(n356), .B(n364), .ZN(n361) );
  VHSR_CLKXOR2_2 U382 ( .A1(n362), .A2(n361), .Z(n358) );
  VHSR_CLKNAND2_2 U383 ( .A1(n359), .A2(n358), .ZN(n394) );
  VHSR_OAI21_2 U384 ( .A1(n359), .A2(n358), .B(n394), .ZN(n360) );
  VHSR_NOR2_1 U385 ( .A1(n362), .A2(n361), .ZN(n363) );
  VHSR_NOR2_1 U386 ( .A1(n393), .A2(n366), .ZN(product[15]) );
  VHSR_AD1_1 U387 ( .A(n391), .B(n390), .CI(n389), .CO(n359), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U388 ( .A1(n393), .A2(n392), .ZN(n396) );
  VHSR_XOR3_2 U389 ( .A1(n396), .A2(n395), .A3(n394), .Z(product[14]) );
  VHSR_AOI21_2 U390 ( .A1(n399), .A2(n398), .B(n397), .ZN(product[4]) );
  VHSR_AOI22_2 U391 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n402) );
  VHSR_AOI21_2 U392 ( .A1(n402), .A2(n401), .B(n400), .ZN(product[2]) );
endmodule


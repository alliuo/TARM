
module mul8_109 ( a, b, product );
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
         n401;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U203 ( .A1(n350), .B1(n348), .ZN(n351) );
  VHSR_NOR2_1 U204 ( .A1(n345), .A2(n344), .ZN(n357) );
  VHSR_NOR2_1 U205 ( .A1(n394), .A2(n393), .ZN(n392) );
  VHSR_INOR3_2 U206 ( .A1(n372), .B1(n284), .B2(n285), .ZN(n308) );
  VHSR_INOR3_2 U207 ( .A1(n235), .B1(n277), .B2(n314), .ZN(n297) );
  VHSR_NOR2_1 U208 ( .A1(n281), .A2(n276), .ZN(n372) );
  VHSR_IN_2 U209 ( .I(n355), .ZN(product[13]) );
  VHSR_INOR2_1 U210 ( .A1(n359), .B1(n358), .ZN(n390) );
  VHSR_INAND2_1 U211 ( .A1(n324), .B1(n341), .ZN(n337) );
  VHSR_NOR2_2 U212 ( .A1(n294), .A2(n293), .ZN(n292) );
  VHSR_AD1_1 U213 ( .A(n379), .B(n378), .CI(n377), .CO(n374), .S(product[6])
         );
  VHSR_AD1_1 U214 ( .A(n373), .B(n372), .CI(n371), .CO(n368), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(product[10])
         );
  VHSR_AD1_1 U216 ( .A(n383), .B(n399), .CI(n382), .CO(n343), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n381), .B(n380), .CI(n396), .CO(n377), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n376), .B(n375), .CI(n374), .CO(n371), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U219 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(product[9])
         );
  VHSR_AD1_1 U220 ( .A(n364), .B(n363), .CI(n362), .CO(n384), .S(product[11])
         );
  VHSR_IN_2 U221 ( .I(b[0]), .ZN(n320) );
  VHSR_IN_2 U222 ( .I(a[1]), .ZN(n315) );
  VHSR_NOR2_1 U223 ( .A1(n320), .A2(n315), .ZN(product[1]) );
  VHSR_AOI22_2 U224 ( .A1(b[2]), .A2(a[7]), .B1(b[3]), .B2(a[6]), .ZN(n246) );
  VHSR_IN_2 U225 ( .I(b[3]), .ZN(n314) );
  VHSR_CLKNAND2_2 U226 ( .A1(b[2]), .A2(a[4]), .ZN(n267) );
  VHSR_IN_2 U227 ( .I(a[5]), .ZN(n284) );
  VHSR_NOR3_2 U228 ( .A1(n314), .A2(n267), .A3(n284), .ZN(n244) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n277) );
  VHSR_IN_2 U230 ( .I(b[1]), .ZN(n322) );
  VHSR_NOR2_1 U231 ( .A1(n277), .A2(n322), .ZN(n213) );
  VHSR_AOI211_2 U232 ( .A1(b[2]), .A2(a[4]), .B(n314), .C(n284), .ZN(n214) );
  VHSR_CLKNAND2_2 U233 ( .A1(b[2]), .A2(a[6]), .ZN(n216) );
  VHSR_IN_2 U234 ( .I(n216), .ZN(n212) );
  VHSR_MAOI222_2 U235 ( .A(n213), .B(n214), .C(n212), .ZN(n225) );
  VHSR_AOI21_2 U236 ( .A1(b[1]), .A2(a[7]), .B(n214), .ZN(n217) );
  VHSR_IN_2 U237 ( .I(n225), .ZN(n215) );
  VHSR_AOI21_2 U238 ( .A1(n217), .A2(n216), .B(n215), .ZN(n253) );
  VHSR_CLKNAND2_2 U239 ( .A1(a[6]), .A2(b[1]), .ZN(n222) );
  VHSR_IN_2 U240 ( .I(n222), .ZN(n219) );
  VHSR_IN_2 U241 ( .I(a[4]), .ZN(n281) );
  VHSR_NOR4_2 U242 ( .A1(n281), .A2(n284), .A3(n322), .A4(n320), .ZN(n271) );
  VHSR_AOI22_2 U243 ( .A1(b[2]), .A2(a[5]), .B1(b[3]), .B2(a[4]), .ZN(n218) );
  VHSR_NOR2_1 U244 ( .A1(n244), .A2(n218), .ZN(n220) );
  VHSR_MAOI222_2 U245 ( .A(n219), .B(n271), .C(n220), .ZN(n224) );
  VHSR_OAI21_2 U246 ( .A1(a[7]), .A2(a[6]), .B(b[0]), .ZN(n266) );
  VHSR_CLKNAND2_2 U247 ( .A1(a[4]), .A2(b[0]), .ZN(n394) );
  VHSR_NAND3_2 U248 ( .A1(b[1]), .A2(a[5]), .A3(n394), .ZN(n265) );
  VHSR_MAOI222_2 U249 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NOR2_1 U250 ( .A1(n271), .A2(n220), .ZN(n223) );
  VHSR_IN_2 U251 ( .I(n224), .ZN(n221) );
  VHSR_AOI21_2 U252 ( .A1(n223), .A2(n222), .B(n221), .ZN(n256) );
  VHSR_CLKNAND2_2 U253 ( .A1(n264), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U254 ( .A1(n224), .A2(n255), .ZN(n252) );
  VHSR_CLKNAND2_2 U255 ( .A1(n253), .A2(n252), .ZN(n251) );
  VHSR_CLKNAND2_2 U256 ( .A1(n225), .A2(n251), .ZN(n243) );
  VHSR_NOR2_1 U257 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_NOR2_1 U258 ( .A1(n246), .A2(n242), .ZN(n235) );
  VHSR_IN_2 U259 ( .I(b[7]), .ZN(n279) );
  VHSR_IN_2 U260 ( .I(a[3]), .ZN(n319) );
  VHSR_IN_2 U261 ( .I(b[6]), .ZN(n280) );
  VHSR_IN_2 U262 ( .I(a[2]), .ZN(n321) );
  VHSR_OAI22_2 U263 ( .A1(n280), .A2(n319), .B1(n279), .B2(n321), .ZN(n241) );
  VHSR_AOI22_2 U264 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_NAND3_2 U266 ( .A1(a[3]), .A2(b[5]), .A3(n263), .ZN(n231) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[7]), .A2(a[2]), .ZN(n226) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[6]), .A2(a[1]), .ZN(n228) );
  VHSR_OAI22_2 U269 ( .A1(n232), .A2(n231), .B1(n226), .B2(n228), .ZN(n233) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[4]), .A2(a[0]), .ZN(n393) );
  VHSR_NAND3_2 U271 ( .A1(a[1]), .A2(b[5]), .A3(n393), .ZN(n262) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[0]), .ZN(n261) );
  VHSR_MAOI222_2 U273 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NAND4_2 U274 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_IN_2 U275 ( .I(b[4]), .ZN(n276) );
  VHSR_IN_2 U276 ( .I(b[5]), .ZN(n285) );
  VHSR_OAI22_2 U277 ( .A1(n276), .A2(n319), .B1(n285), .B2(n321), .ZN(n227) );
  VHSR_AND2_2 U278 ( .A1(n238), .A2(n227), .Z(n230) );
  VHSR_IN_2 U279 ( .I(a[0]), .ZN(n316) );
  VHSR_OAI21_2 U280 ( .A1(n279), .A2(n316), .B(n228), .ZN(n229) );
  VHSR_NOR4_2 U281 ( .A1(n276), .A2(n285), .A3(n315), .A4(n316), .ZN(n269) );
  VHSR_AND2_2 U282 ( .A1(n260), .A2(n259), .Z(n258) );
  VHSR_AD1_1 U283 ( .A(n230), .B(n229), .CI(n269), .CO(n247), .S(n259) );
  VHSR_AOI21_2 U284 ( .A1(n232), .A2(n231), .B(n233), .ZN(n250) );
  VHSR_OAI32_2 U285 ( .A1(n233), .A2(n258), .A3(n247), .B1(n250), .B2(n233), 
        .ZN(n239) );
  VHSR_CLKNAND2_2 U286 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U287 ( .A1(n241), .A2(n237), .ZN(n236) );
  VHSR_NOR3_2 U288 ( .A1(n279), .A2(n319), .A3(n236), .ZN(n296) );
  VHSR_NOR2_1 U289 ( .A1(n277), .A2(n314), .ZN(n234) );
  VHSR_IAO21_2 U290 ( .A1(n235), .A2(n234), .B(n297), .ZN(n300) );
  VHSR_OAI32_2 U291 ( .A1(n296), .A2(n319), .A3(n279), .B1(n236), .B2(n296), 
        .ZN(n299) );
  VHSR_OAI21_2 U292 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U293 ( .A1(n241), .A2(n240), .ZN(n307) );
  VHSR_AOI21_2 U294 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U295 ( .A1(n246), .A2(n245), .ZN(n306) );
  VHSR_NOR2_1 U296 ( .A1(n258), .A2(n247), .ZN(n249) );
  VHSR_AOI22_2 U297 ( .A1(n258), .A2(n247), .B1(n250), .B2(n249), .ZN(n248) );
  VHSR_OAI21_2 U298 ( .A1(n250), .A2(n249), .B(n248), .ZN(n312) );
  VHSR_OAI21_2 U299 ( .A1(n253), .A2(n252), .B(n251), .ZN(n254) );
  VHSR_IN_2 U300 ( .I(n254), .ZN(n311) );
  VHSR_OAI21_2 U301 ( .A1(n264), .A2(n256), .B(n255), .ZN(n257) );
  VHSR_IN_2 U302 ( .I(n257), .ZN(n328) );
  VHSR_IAO21_2 U303 ( .A1(n260), .A2(n259), .B(n258), .ZN(n327) );
  VHSR_AOI31_2 U304 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n335) );
  VHSR_AOI31_2 U305 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n334) );
  VHSR_CLKNAND2_2 U306 ( .A1(b[5]), .A2(a[0]), .ZN(n268) );
  VHSR_OAI32_2 U307 ( .A1(n269), .A2(n315), .A3(n276), .B1(n268), .B2(n269), 
        .ZN(n340) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[5]), .A2(b[0]), .ZN(n270) );
  VHSR_OAI32_2 U309 ( .A1(n271), .A2(n322), .A3(n281), .B1(n270), .B2(n271), 
        .ZN(n339) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[6]), .ZN(n360) );
  VHSR_IN_2 U311 ( .I(n360), .ZN(n387) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n304) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n304), .ZN(n273) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[6]), .ZN(n303) );
  VHSR_NAND3_2 U315 ( .A1(b[7]), .A2(a[5]), .A3(n303), .ZN(n272) );
  VHSR_CLKNAND2_2 U316 ( .A1(n273), .A2(n272), .ZN(n275) );
  VHSR_MAOI222_2 U317 ( .A(n360), .B(n273), .C(n272), .ZN(n344) );
  VHSR_IN_2 U318 ( .I(n344), .ZN(n274) );
  VHSR_OAI21_2 U319 ( .A1(n387), .A2(n275), .B(n274), .ZN(n291) );
  VHSR_NOR3_2 U320 ( .A1(n277), .A2(n304), .A3(n285), .ZN(n352) );
  VHSR_AOI22_2 U321 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n278) );
  VHSR_NOR2_1 U322 ( .A1(n352), .A2(n278), .ZN(n287) );
  VHSR_NOR4_2 U323 ( .A1(n281), .A2(n284), .A3(n280), .A4(n279), .ZN(n350) );
  VHSR_AOI22_2 U324 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n282) );
  VHSR_NOR2_1 U325 ( .A1(n350), .A2(n282), .ZN(n286) );
  VHSR_IN_2 U326 ( .I(n283), .ZN(n294) );
  VHSR_OR3_2 U327 ( .A1(n372), .A2(n285), .A3(n284), .Z(n302) );
  VHSR_MAOI222_2 U328 ( .A(n304), .B(n303), .C(n302), .ZN(n301) );
  VHSR_IN_2 U329 ( .I(n301), .ZN(n293) );
  VHSR_AD1_1 U330 ( .A(n308), .B(n287), .CI(n286), .CO(n288), .S(n283) );
  VHSR_NOR2_1 U331 ( .A1(n292), .A2(n288), .ZN(n290) );
  VHSR_CLKNAND2_2 U332 ( .A1(n292), .A2(n288), .ZN(n289) );
  VHSR_NOR2_1 U333 ( .A1(n290), .A2(n291), .ZN(n345) );
  VHSR_AOI22_2 U334 ( .A1(n291), .A2(n290), .B1(n289), .B2(n345), .ZN(n385) );
  VHSR_AOI21_2 U335 ( .A1(n294), .A2(n293), .B(n292), .ZN(n364) );
  VHSR_AD1_1 U336 ( .A(n297), .B(n296), .CI(n295), .CO(n386), .S(n363) );
  VHSR_AD1_1 U337 ( .A(n300), .B(n299), .CI(n298), .CO(n295), .S(n367) );
  VHSR_AOI31_2 U338 ( .A1(n304), .A2(n303), .A3(n302), .B(n301), .ZN(n366) );
  VHSR_AD1_1 U339 ( .A(n307), .B(n306), .CI(n305), .CO(n298), .S(n370) );
  VHSR_AOI22_2 U340 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n309) );
  VHSR_NOR2_1 U341 ( .A1(n309), .A2(n308), .ZN(n369) );
  VHSR_AD1_1 U342 ( .A(n312), .B(n311), .CI(n310), .CO(n305), .S(n373) );
  VHSR_IN_2 U343 ( .I(b[2]), .ZN(n318) );
  VHSR_NOR2_1 U344 ( .A1(n318), .A2(n321), .ZN(n332) );
  VHSR_IN_2 U345 ( .I(n332), .ZN(n325) );
  VHSR_CLKNAND2_2 U346 ( .A1(b[3]), .A2(a[3]), .ZN(n331) );
  VHSR_AOI22_2 U347 ( .A1(b[2]), .A2(a[3]), .B1(b[3]), .B2(a[2]), .ZN(n313) );
  VHSR_IAO21_2 U348 ( .A1(n325), .A2(n331), .B(n313), .ZN(n338) );
  VHSR_AOI22_2 U349 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n323) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[1]), .A2(a[1]), .ZN(n401) );
  VHSR_OAI22_2 U351 ( .A1(n325), .A2(n323), .B1(n331), .B2(n401), .ZN(n324) );
  VHSR_OAI22_2 U352 ( .A1(n318), .A2(n315), .B1(n314), .B2(n316), .ZN(n383) );
  VHSR_AOI21_2 U353 ( .A1(n322), .A2(n320), .B(n316), .ZN(product[0]) );
  VHSR_AOI32_2 U354 ( .A1(b[0]), .A2(product[0]), .A3(a[2]), .B1(a[1]), .B2(
        product[0]), .ZN(n317) );
  VHSR_AOI211_2 U355 ( .A1(n322), .A2(n321), .B(n318), .C(n317), .ZN(n399) );
  VHSR_OAI22_2 U356 ( .A1(n322), .A2(n321), .B1(n320), .B2(n319), .ZN(n382) );
  VHSR_AOI21_2 U357 ( .A1(n323), .A2(n325), .B(n324), .ZN(n342) );
  VHSR_CLKNAND2_2 U358 ( .A1(n343), .A2(n342), .ZN(n341) );
  VHSR_CLKNAND2_2 U359 ( .A1(n338), .A2(n337), .ZN(n329) );
  VHSR_AOI21_2 U360 ( .A1(n325), .A2(n329), .B(n331), .ZN(n376) );
  VHSR_AD1_1 U361 ( .A(n328), .B(n327), .CI(n326), .CO(n310), .S(n375) );
  VHSR_IN_2 U362 ( .I(n329), .ZN(n336) );
  VHSR_CLKNAND2_2 U363 ( .A1(n336), .A2(n331), .ZN(n330) );
  VHSR_OAI31_2 U364 ( .A1(n332), .A2(n336), .A3(n331), .B(n330), .ZN(n379) );
  VHSR_AD1_1 U365 ( .A(n335), .B(n334), .CI(n333), .CO(n326), .S(n378) );
  VHSR_IAO21_2 U366 ( .A1(n338), .A2(n337), .B(n336), .ZN(n381) );
  VHSR_AD1_1 U367 ( .A(n340), .B(n392), .CI(n339), .CO(n333), .S(n380) );
  VHSR_OAI21_2 U368 ( .A1(n343), .A2(n342), .B(n341), .ZN(n398) );
  VHSR_AOI211_2 U369 ( .A1(n394), .A2(n393), .B(n392), .C(n398), .ZN(n396) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[6]), .A2(b[7]), .ZN(n347) );
  VHSR_AOI21_2 U371 ( .A1(a[7]), .A2(b[6]), .B(n347), .ZN(n346) );
  VHSR_AOI31_2 U372 ( .A1(a[7]), .A2(n347), .A3(b[6]), .B(n346), .ZN(n348) );
  VHSR_IN_2 U373 ( .I(n348), .ZN(n349) );
  VHSR_MAOI222_2 U374 ( .A(n352), .B(n350), .C(n349), .ZN(n359) );
  VHSR_OAI21_2 U375 ( .A1(n352), .A2(n351), .B(n359), .ZN(n356) );
  VHSR_CLKXOR2_2 U376 ( .A1(n357), .A2(n356), .Z(n353) );
  VHSR_CLKNAND2_2 U377 ( .A1(n354), .A2(n353), .ZN(n389) );
  VHSR_OAI21_2 U378 ( .A1(n354), .A2(n353), .B(n389), .ZN(n355) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[7]), .A2(b[7]), .ZN(n388) );
  VHSR_NOR2_1 U380 ( .A1(n357), .A2(n356), .ZN(n358) );
  VHSR_AND3_2 U381 ( .A1(n390), .A2(n360), .A3(n389), .Z(n361) );
  VHSR_NOR2_1 U382 ( .A1(n388), .A2(n361), .ZN(product[15]) );
  VHSR_AD1_1 U383 ( .A(n386), .B(n385), .CI(n384), .CO(n354), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U384 ( .A1(n388), .A2(n387), .ZN(n391) );
  VHSR_XOR3_2 U385 ( .A1(n391), .A2(n390), .A3(n389), .Z(product[14]) );
  VHSR_AOI21_2 U386 ( .A1(n394), .A2(n393), .B(n392), .ZN(n395) );
  VHSR_IN_2 U387 ( .I(n395), .ZN(n397) );
  VHSR_AOI21_2 U388 ( .A1(n398), .A2(n397), .B(n396), .ZN(product[4]) );
  VHSR_AOI22_2 U389 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n400) );
  VHSR_AOI21_2 U390 ( .A1(n401), .A2(n400), .B(n399), .ZN(product[2]) );
endmodule



module mul8_73 ( a, b, product );
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

  VHSR_INOR2_2 U203 ( .A1(n233), .B1(n252), .ZN(n239) );
  VHSR_NOR2_1 U204 ( .A1(n246), .A2(n242), .ZN(n235) );
  VHSR_NOR2_1 U205 ( .A1(n318), .A2(n320), .ZN(n331) );
  VHSR_INOR3_2 U206 ( .A1(n235), .B1(n319), .B2(n278), .ZN(n296) );
  VHSR_NOR2_1 U207 ( .A1(n394), .A2(n393), .ZN(n392) );
  VHSR_INOR2_2 U208 ( .A1(n359), .B1(n358), .ZN(n390) );
  VHSR_IN_2 U209 ( .I(n355), .ZN(product[13]) );
  VHSR_NOR2_2 U210 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_NOR2_2 U211 ( .A1(n345), .A2(n344), .ZN(n357) );
  VHSR_INAND2_1 U212 ( .A1(n350), .B1(n348), .ZN(n351) );
  VHSR_MOAI22_1 U213 ( .A1(n278), .A2(n398), .B1(a[6]), .B2(b[2]), .ZN(n215)
         );
  VHSR_AD1_1 U214 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(product[9])
         );
  VHSR_AD1_1 U215 ( .A(n377), .B(n399), .CI(n376), .CO(n334), .S(product[3])
         );
  VHSR_AD1_1 U216 ( .A(n392), .B(n375), .CI(n374), .CO(n378), .S(product[5])
         );
  VHSR_AD1_1 U217 ( .A(n373), .B(n372), .CI(n371), .CO(n368), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U218 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U219 ( .A(n364), .B(n363), .CI(n362), .CO(n381), .S(product[10])
         );
  VHSR_AOI22_2 U220 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n246) );
  VHSR_IN_2 U221 ( .I(b[3]), .ZN(n319) );
  VHSR_IN_2 U222 ( .I(b[2]), .ZN(n318) );
  VHSR_IN_2 U223 ( .I(a[5]), .ZN(n283) );
  VHSR_IN_2 U224 ( .I(a[4]), .ZN(n282) );
  VHSR_NOR4_2 U225 ( .A1(n319), .A2(n318), .A3(n283), .A4(n282), .ZN(n244) );
  VHSR_IN_2 U226 ( .I(a[7]), .ZN(n278) );
  VHSR_IN_2 U227 ( .I(b[1]), .ZN(n398) );
  VHSR_NOR2_1 U228 ( .A1(n278), .A2(n398), .ZN(n213) );
  VHSR_AND2_2 U229 ( .A1(a[6]), .A2(b[2]), .Z(n212) );
  VHSR_AOI211_2 U230 ( .A1(b[2]), .A2(a[4]), .B(n319), .C(n283), .ZN(n214) );
  VHSR_MAOI222_2 U231 ( .A(n213), .B(n212), .C(n214), .ZN(n225) );
  VHSR_OAI21_2 U232 ( .A1(n215), .A2(n214), .B(n225), .ZN(n216) );
  VHSR_IN_2 U233 ( .I(n216), .ZN(n249) );
  VHSR_IN_2 U234 ( .I(b[0]), .ZN(n396) );
  VHSR_NOR4_2 U235 ( .A1(n283), .A2(n282), .A3(n398), .A4(n396), .ZN(n272) );
  VHSR_CLKNAND2_2 U236 ( .A1(b[2]), .A2(a[5]), .ZN(n218) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[3]), .A2(a[4]), .ZN(n217) );
  VHSR_AOI21_2 U238 ( .A1(n218), .A2(n217), .B(n244), .ZN(n220) );
  VHSR_AOI22_2 U239 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n222) );
  VHSR_IN_2 U240 ( .I(n222), .ZN(n219) );
  VHSR_MAOI222_2 U241 ( .A(n272), .B(n220), .C(n219), .ZN(n224) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[2]), .A2(a[4]), .ZN(n268) );
  VHSR_OAI211_2 U243 ( .A1(n282), .A2(n396), .B(a[5]), .C(b[1]), .ZN(n267) );
  VHSR_CLKNAND2_2 U244 ( .A1(a[6]), .A2(b[0]), .ZN(n266) );
  VHSR_MAOI222_2 U245 ( .A(n268), .B(n267), .C(n266), .ZN(n265) );
  VHSR_NOR2_1 U246 ( .A1(n272), .A2(n220), .ZN(n223) );
  VHSR_IN_2 U247 ( .I(n224), .ZN(n221) );
  VHSR_AOI21_2 U248 ( .A1(n223), .A2(n222), .B(n221), .ZN(n259) );
  VHSR_CLKNAND2_2 U249 ( .A1(n265), .A2(n259), .ZN(n258) );
  VHSR_CLKNAND2_2 U250 ( .A1(n224), .A2(n258), .ZN(n248) );
  VHSR_CLKNAND2_2 U251 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U252 ( .A1(n225), .A2(n247), .ZN(n243) );
  VHSR_IN_2 U253 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U254 ( .I(a[3]), .ZN(n322) );
  VHSR_IN_2 U255 ( .I(b[6]), .ZN(n281) );
  VHSR_IN_2 U256 ( .I(a[2]), .ZN(n320) );
  VHSR_OAI22_2 U257 ( .A1(n281), .A2(n322), .B1(n280), .B2(n320), .ZN(n241) );
  VHSR_NOR2_1 U258 ( .A1(n280), .A2(n320), .ZN(n227) );
  VHSR_IN_2 U259 ( .I(a[1]), .ZN(n395) );
  VHSR_NOR2_1 U260 ( .A1(n281), .A2(n395), .ZN(n226) );
  VHSR_IN_2 U261 ( .I(b[5]), .ZN(n277) );
  VHSR_AOI211_2 U262 ( .A1(b[4]), .A2(a[2]), .B(n277), .C(n322), .ZN(n232) );
  VHSR_OAI22_2 U263 ( .A1(n281), .A2(n320), .B1(n280), .B2(n395), .ZN(n231) );
  VHSR_AOI22_2 U264 ( .A1(n227), .A2(n226), .B1(n232), .B2(n231), .ZN(n233) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[2]), .ZN(n264) );
  VHSR_IN_2 U266 ( .I(b[4]), .ZN(n335) );
  VHSR_IN_2 U267 ( .I(a[0]), .ZN(n397) );
  VHSR_OAI211_2 U268 ( .A1(n335), .A2(n397), .B(b[5]), .C(a[1]), .ZN(n263) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[0]), .ZN(n262) );
  VHSR_MAOI222_2 U270 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_NOR4_2 U271 ( .A1(n335), .A2(n277), .A3(n395), .A4(n397), .ZN(n270) );
  VHSR_NAND4_2 U272 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_OAI22_2 U273 ( .A1(n335), .A2(n322), .B1(n277), .B2(n320), .ZN(n228) );
  VHSR_AND2_2 U274 ( .A1(n238), .A2(n228), .Z(n230) );
  VHSR_OAI22_2 U275 ( .A1(n281), .A2(n395), .B1(n280), .B2(n397), .ZN(n229) );
  VHSR_AND2_2 U276 ( .A1(n261), .A2(n257), .Z(n256) );
  VHSR_AD1_1 U277 ( .A(n270), .B(n230), .CI(n229), .CO(n251), .S(n257) );
  VHSR_NOR2_1 U278 ( .A1(n256), .A2(n251), .ZN(n254) );
  VHSR_OAI21_2 U279 ( .A1(n232), .A2(n231), .B(n233), .ZN(n255) );
  VHSR_NOR2_1 U280 ( .A1(n254), .A2(n255), .ZN(n252) );
  VHSR_CLKNAND2_2 U281 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U282 ( .A1(n241), .A2(n237), .ZN(n236) );
  VHSR_NOR3_2 U283 ( .A1(n280), .A2(n322), .A3(n236), .ZN(n295) );
  VHSR_NOR2_1 U284 ( .A1(n319), .A2(n278), .ZN(n234) );
  VHSR_IAO21_2 U285 ( .A1(n235), .A2(n234), .B(n296), .ZN(n299) );
  VHSR_OAI32_2 U286 ( .A1(n295), .A2(n322), .A3(n280), .B1(n236), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U287 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U288 ( .A1(n241), .A2(n240), .ZN(n306) );
  VHSR_AOI21_2 U289 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U290 ( .A1(n246), .A2(n245), .ZN(n305) );
  VHSR_OAI21_2 U291 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_IN_2 U292 ( .I(n250), .ZN(n311) );
  VHSR_CLKNAND2_2 U293 ( .A1(n256), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U294 ( .A1(n255), .A2(n254), .B1(n253), .B2(n252), .ZN(n310) );
  VHSR_IAO21_2 U295 ( .A1(n261), .A2(n257), .B(n256), .ZN(n314) );
  VHSR_OAI21_2 U296 ( .A1(n265), .A2(n259), .B(n258), .ZN(n260) );
  VHSR_IN_2 U297 ( .I(n260), .ZN(n313) );
  VHSR_AOI31_2 U298 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n328) );
  VHSR_AOI31_2 U299 ( .A1(n268), .A2(n267), .A3(n266), .B(n265), .ZN(n327) );
  VHSR_CLKNAND2_2 U300 ( .A1(b[5]), .A2(a[0]), .ZN(n269) );
  VHSR_OAI32_2 U301 ( .A1(n270), .A2(n395), .A3(n335), .B1(n269), .B2(n270), 
        .ZN(n343) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[4]), .A2(b[1]), .ZN(n271) );
  VHSR_OAI32_2 U303 ( .A1(n272), .A2(n283), .A3(n396), .B1(n271), .B2(n272), 
        .ZN(n342) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[4]), .ZN(n285) );
  VHSR_IN_2 U305 ( .I(n285), .ZN(n369) );
  VHSR_NOR2_1 U306 ( .A1(n396), .A2(n397), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U307 ( .A1(n369), .A2(product[0]), .ZN(n337) );
  VHSR_IN_2 U308 ( .I(n337), .ZN(n341) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[6]), .ZN(n360) );
  VHSR_IN_2 U310 ( .I(n360), .ZN(n387) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_NAND3_2 U312 ( .A1(a[7]), .A2(b[5]), .A3(n303), .ZN(n274) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[4]), .A2(b[6]), .ZN(n302) );
  VHSR_NAND3_2 U314 ( .A1(b[7]), .A2(a[5]), .A3(n302), .ZN(n273) );
  VHSR_CLKNAND2_2 U315 ( .A1(n274), .A2(n273), .ZN(n276) );
  VHSR_MAOI222_2 U316 ( .A(n360), .B(n274), .C(n273), .ZN(n344) );
  VHSR_IN_2 U317 ( .I(n344), .ZN(n275) );
  VHSR_OAI21_2 U318 ( .A1(n387), .A2(n276), .B(n275), .ZN(n291) );
  VHSR_NOR3_2 U319 ( .A1(n283), .A2(n277), .A3(n285), .ZN(n307) );
  VHSR_NOR3_2 U320 ( .A1(n278), .A2(n303), .A3(n277), .ZN(n352) );
  VHSR_AOI22_2 U321 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n279) );
  VHSR_NOR2_1 U322 ( .A1(n352), .A2(n279), .ZN(n287) );
  VHSR_NOR4_2 U323 ( .A1(n283), .A2(n282), .A3(n281), .A4(n280), .ZN(n350) );
  VHSR_AOI22_2 U324 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n284) );
  VHSR_NOR2_1 U325 ( .A1(n350), .A2(n284), .ZN(n286) );
  VHSR_NAND3_2 U326 ( .A1(b[5]), .A2(a[5]), .A3(n285), .ZN(n301) );
  VHSR_MAOI222_2 U327 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_AND2_2 U328 ( .A1(n293), .A2(n300), .Z(n292) );
  VHSR_AD1_1 U329 ( .A(n307), .B(n287), .CI(n286), .CO(n288), .S(n293) );
  VHSR_NOR2_1 U330 ( .A1(n292), .A2(n288), .ZN(n290) );
  VHSR_CLKNAND2_2 U331 ( .A1(n292), .A2(n288), .ZN(n289) );
  VHSR_NOR2_1 U332 ( .A1(n290), .A2(n291), .ZN(n345) );
  VHSR_AOI22_2 U333 ( .A1(n291), .A2(n290), .B1(n289), .B2(n345), .ZN(n385) );
  VHSR_IAO21_2 U334 ( .A1(n293), .A2(n300), .B(n292), .ZN(n383) );
  VHSR_AD1_1 U335 ( .A(n296), .B(n295), .CI(n294), .CO(n386), .S(n382) );
  VHSR_AD1_1 U336 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n364) );
  VHSR_AOI31_2 U337 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n363) );
  VHSR_AD1_1 U338 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n367) );
  VHSR_AOI22_2 U339 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n308) );
  VHSR_NOR2_1 U340 ( .A1(n308), .A2(n307), .ZN(n366) );
  VHSR_AD1_1 U341 ( .A(n311), .B(n310), .CI(n309), .CO(n304), .S(n370) );
  VHSR_AD1_1 U342 ( .A(n314), .B(n313), .CI(n312), .CO(n309), .S(n373) );
  VHSR_AOI22_2 U343 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n315) );
  VHSR_AOI31_2 U344 ( .A1(a[3]), .A2(b[3]), .A3(n331), .B(n315), .ZN(n340) );
  VHSR_NOR2_1 U345 ( .A1(n319), .A2(n395), .ZN(n317) );
  VHSR_NOR2_1 U346 ( .A1(n398), .A2(n322), .ZN(n316) );
  VHSR_MAOI222_2 U347 ( .A(n331), .B(n317), .C(n316), .ZN(n324) );
  VHSR_OAI22_2 U348 ( .A1(n319), .A2(n397), .B1(n318), .B2(n395), .ZN(n377) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[0]), .A2(a[2]), .ZN(n402) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[2]), .A2(a[0]), .ZN(n401) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[1]), .A2(a[1]), .ZN(n400) );
  VHSR_MAOI222_2 U352 ( .A(n402), .B(n401), .C(n400), .ZN(n399) );
  VHSR_OAI22_2 U353 ( .A1(n398), .A2(n320), .B1(n396), .B2(n322), .ZN(n376) );
  VHSR_IN_2 U354 ( .I(n324), .ZN(n323) );
  VHSR_AOI21_2 U355 ( .A1(a[1]), .A2(b[3]), .B(n331), .ZN(n321) );
  VHSR_OAI32_2 U356 ( .A1(n323), .A2(n322), .A3(n398), .B1(n321), .B2(n323), 
        .ZN(n333) );
  VHSR_CLKNAND2_2 U357 ( .A1(n334), .A2(n333), .ZN(n332) );
  VHSR_CLKNAND2_2 U358 ( .A1(n324), .A2(n332), .ZN(n339) );
  VHSR_AND2_2 U359 ( .A1(n340), .A2(n339), .Z(n338) );
  VHSR_OAI211_2 U360 ( .A1(n331), .A2(n338), .B(a[3]), .C(b[3]), .ZN(n325) );
  VHSR_IN_2 U361 ( .I(n325), .ZN(n372) );
  VHSR_AD1_1 U362 ( .A(n328), .B(n327), .CI(n326), .CO(n312), .S(n380) );
  VHSR_CLKNAND2_2 U363 ( .A1(b[3]), .A2(a[3]), .ZN(n330) );
  VHSR_CLKNAND2_2 U364 ( .A1(n338), .A2(n330), .ZN(n329) );
  VHSR_OAI31_2 U365 ( .A1(n331), .A2(n338), .A3(n330), .B(n329), .ZN(n379) );
  VHSR_OAI21_2 U366 ( .A1(n334), .A2(n333), .B(n332), .ZN(n394) );
  VHSR_NOR2_1 U367 ( .A1(n335), .A2(n397), .ZN(n336) );
  VHSR_AOI32_2 U368 ( .A1(b[0]), .A2(n337), .A3(a[4]), .B1(n336), .B2(n337), 
        .ZN(n393) );
  VHSR_IAO21_2 U369 ( .A1(n340), .A2(n339), .B(n338), .ZN(n375) );
  VHSR_AD1_1 U370 ( .A(n343), .B(n342), .CI(n341), .CO(n326), .S(n374) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[7]), .A2(b[6]), .ZN(n347) );
  VHSR_AOI21_2 U372 ( .A1(a[6]), .A2(b[7]), .B(n347), .ZN(n346) );
  VHSR_AOI31_2 U373 ( .A1(a[6]), .A2(n347), .A3(b[7]), .B(n346), .ZN(n348) );
  VHSR_IN_2 U374 ( .I(n348), .ZN(n349) );
  VHSR_MAOI222_2 U375 ( .A(n352), .B(n350), .C(n349), .ZN(n359) );
  VHSR_OAI21_2 U376 ( .A1(n352), .A2(n351), .B(n359), .ZN(n356) );
  VHSR_CLKXOR2_2 U377 ( .A1(n357), .A2(n356), .Z(n353) );
  VHSR_CLKNAND2_2 U378 ( .A1(n354), .A2(n353), .ZN(n389) );
  VHSR_OAI21_2 U379 ( .A1(n354), .A2(n353), .B(n389), .ZN(n355) );
  VHSR_CLKNAND2_2 U380 ( .A1(a[7]), .A2(b[7]), .ZN(n388) );
  VHSR_NOR2_1 U381 ( .A1(n357), .A2(n356), .ZN(n358) );
  VHSR_AND3_2 U382 ( .A1(n390), .A2(n360), .A3(n389), .Z(n361) );
  VHSR_NOR2_1 U383 ( .A1(n388), .A2(n361), .ZN(product[15]) );
  VHSR_AD1_1 U384 ( .A(n380), .B(n379), .CI(n378), .CO(n371), .S(product[6])
         );
  VHSR_AD1_1 U385 ( .A(n383), .B(n382), .CI(n381), .CO(n384), .S(product[11])
         );
  VHSR_AD1_1 U386 ( .A(n386), .B(n385), .CI(n384), .CO(n354), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U387 ( .A1(n388), .A2(n387), .ZN(n391) );
  VHSR_XOR3_2 U388 ( .A1(n391), .A2(n390), .A3(n389), .Z(product[14]) );
  VHSR_AOI21_2 U389 ( .A1(n394), .A2(n393), .B(n392), .ZN(product[4]) );
  VHSR_OAI22_2 U390 ( .A1(n398), .A2(n397), .B1(n396), .B2(n395), .ZN(
        product[1]) );
  VHSR_AOI31_2 U391 ( .A1(n402), .A2(n401), .A3(n400), .B(n399), .ZN(
        product[2]) );
endmodule


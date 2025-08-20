
module mul8_63 ( a, b, product );
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

  VHSR_NOR2_1 U204 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_NOR2_1 U205 ( .A1(n247), .A2(n243), .ZN(n236) );
  VHSR_INAND2_2 U206 ( .A1(n321), .B1(n340), .ZN(n336) );
  VHSR_NOR2_1 U207 ( .A1(n289), .A2(n290), .ZN(n344) );
  VHSR_INOR2_2 U208 ( .A1(n358), .B1(n357), .ZN(n389) );
  VHSR_IN_2 U209 ( .I(n354), .ZN(product[13]) );
  VHSR_NOR2_2 U210 ( .A1(n344), .A2(n343), .ZN(n356) );
  VHSR_NOR2_2 U211 ( .A1(n291), .A2(n287), .ZN(n289) );
  VHSR_MOAI22_1 U212 ( .A1(n277), .A2(n318), .B1(b[4]), .B2(a[3]), .ZN(n228)
         );
  VHSR_AD1_1 U213 ( .A(n377), .B(n376), .CI(n395), .CO(n373), .S(product[5])
         );
  VHSR_AD1_1 U214 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U215 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(product[9])
         );
  VHSR_AD1_1 U216 ( .A(n379), .B(n401), .CI(n378), .CO(n342), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(product[6])
         );
  VHSR_AD1_1 U218 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U219 ( .A(n363), .B(n362), .CI(n361), .CO(n380), .S(product[10])
         );
  VHSR_IN_2 U220 ( .I(b[0]), .ZN(n317) );
  VHSR_IN_2 U221 ( .I(a[1]), .ZN(n315) );
  VHSR_NOR2_1 U222 ( .A1(n317), .A2(n315), .ZN(product[1]) );
  VHSR_IN_2 U223 ( .I(b[1]), .ZN(n319) );
  VHSR_IN_2 U224 ( .I(a[0]), .ZN(n400) );
  VHSR_NOR2_1 U225 ( .A1(n319), .A2(n400), .ZN(product[0]) );
  VHSR_AOI22_2 U226 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n247) );
  VHSR_IN_2 U227 ( .I(b[3]), .ZN(n322) );
  VHSR_CLKNAND2_2 U228 ( .A1(b[2]), .A2(a[4]), .ZN(n268) );
  VHSR_IN_2 U229 ( .I(a[5]), .ZN(n282) );
  VHSR_NOR3_2 U230 ( .A1(n322), .A2(n268), .A3(n282), .ZN(n245) );
  VHSR_CLKNAND2_2 U231 ( .A1(a[6]), .A2(b[1]), .ZN(n223) );
  VHSR_IN_2 U232 ( .I(n223), .ZN(n220) );
  VHSR_IN_2 U233 ( .I(n268), .ZN(n217) );
  VHSR_AOI21_2 U234 ( .A1(a[7]), .A2(b[1]), .B(b[2]), .ZN(n213) );
  VHSR_CLKNAND2_2 U235 ( .A1(b[3]), .A2(a[6]), .ZN(n212) );
  VHSR_NOR4_2 U236 ( .A1(n217), .A2(n213), .A3(n212), .A4(n282), .ZN(n214) );
  VHSR_AOI31_2 U237 ( .A1(a[7]), .A2(b[2]), .A3(n220), .B(n214), .ZN(n226) );
  VHSR_IN_2 U238 ( .I(n226), .ZN(n218) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[3]), .A2(a[5]), .ZN(n216) );
  VHSR_AOI32_2 U240 ( .A1(a[7]), .A2(a[6]), .A3(b[1]), .B1(b[2]), .B2(a[6]), 
        .ZN(n215) );
  VHSR_OAI32_2 U241 ( .A1(n218), .A2(n217), .A3(n216), .B1(n215), .B2(n218), 
        .ZN(n254) );
  VHSR_IN_2 U242 ( .I(a[4]), .ZN(n283) );
  VHSR_NOR4_2 U243 ( .A1(n283), .A2(n282), .A3(n319), .A4(n317), .ZN(n272) );
  VHSR_AOI22_2 U244 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n219) );
  VHSR_NOR2_1 U245 ( .A1(n245), .A2(n219), .ZN(n221) );
  VHSR_MAOI222_2 U246 ( .A(n220), .B(n272), .C(n221), .ZN(n225) );
  VHSR_CLKNAND2_2 U247 ( .A1(a[6]), .A2(b[0]), .ZN(n267) );
  VHSR_OAI211_2 U248 ( .A1(n283), .A2(n317), .B(a[5]), .C(b[1]), .ZN(n266) );
  VHSR_MAOI222_2 U249 ( .A(n268), .B(n267), .C(n266), .ZN(n265) );
  VHSR_NOR2_1 U250 ( .A1(n272), .A2(n221), .ZN(n224) );
  VHSR_IN_2 U251 ( .I(n225), .ZN(n222) );
  VHSR_AOI21_2 U252 ( .A1(n224), .A2(n223), .B(n222), .ZN(n257) );
  VHSR_CLKNAND2_2 U253 ( .A1(n265), .A2(n257), .ZN(n256) );
  VHSR_CLKNAND2_2 U254 ( .A1(n225), .A2(n256), .ZN(n253) );
  VHSR_CLKNAND2_2 U255 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U256 ( .A1(n226), .A2(n252), .ZN(n244) );
  VHSR_AND3_2 U257 ( .A1(n236), .A2(b[3]), .A3(a[7]), .Z(n295) );
  VHSR_IN_2 U258 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U259 ( .I(a[3]), .ZN(n323) );
  VHSR_IN_2 U260 ( .I(b[6]), .ZN(n281) );
  VHSR_IN_2 U261 ( .I(a[2]), .ZN(n318) );
  VHSR_OAI22_2 U262 ( .A1(n281), .A2(n323), .B1(n280), .B2(n318), .ZN(n242) );
  VHSR_AOI22_2 U263 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n233) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[4]), .A2(a[2]), .ZN(n264) );
  VHSR_NAND3_2 U265 ( .A1(a[3]), .A2(b[5]), .A3(n264), .ZN(n232) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[7]), .A2(a[2]), .ZN(n227) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[1]), .ZN(n229) );
  VHSR_OAI22_2 U268 ( .A1(n233), .A2(n232), .B1(n227), .B2(n229), .ZN(n234) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[4]), .A2(a[0]), .ZN(n392) );
  VHSR_NAND3_2 U270 ( .A1(a[1]), .A2(b[5]), .A3(n392), .ZN(n263) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[6]), .A2(a[0]), .ZN(n262) );
  VHSR_MAOI222_2 U272 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_NAND4_2 U273 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_IN_2 U274 ( .I(b[5]), .ZN(n277) );
  VHSR_AND2_2 U275 ( .A1(n239), .A2(n228), .Z(n231) );
  VHSR_OAI21_2 U276 ( .A1(n280), .A2(n400), .B(n229), .ZN(n230) );
  VHSR_NOR3_2 U277 ( .A1(n277), .A2(n315), .A3(n392), .ZN(n270) );
  VHSR_AND2_2 U278 ( .A1(n261), .A2(n260), .Z(n259) );
  VHSR_AD1_1 U279 ( .A(n231), .B(n230), .CI(n270), .CO(n248), .S(n260) );
  VHSR_AOI21_2 U280 ( .A1(n233), .A2(n232), .B(n234), .ZN(n251) );
  VHSR_OAI32_2 U281 ( .A1(n234), .A2(n259), .A3(n248), .B1(n251), .B2(n234), 
        .ZN(n240) );
  VHSR_CLKNAND2_2 U282 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U283 ( .A1(n242), .A2(n238), .ZN(n237) );
  VHSR_NOR3_2 U284 ( .A1(n280), .A2(n323), .A3(n237), .ZN(n294) );
  VHSR_IN_2 U285 ( .I(a[7]), .ZN(n278) );
  VHSR_NOR2_1 U286 ( .A1(n322), .A2(n278), .ZN(n235) );
  VHSR_IAO21_2 U287 ( .A1(n236), .A2(n235), .B(n295), .ZN(n298) );
  VHSR_OAI32_2 U288 ( .A1(n294), .A2(n323), .A3(n280), .B1(n237), .B2(n294), 
        .ZN(n297) );
  VHSR_OAI21_2 U289 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U290 ( .A1(n242), .A2(n241), .ZN(n305) );
  VHSR_AOI21_2 U291 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U292 ( .A1(n247), .A2(n246), .ZN(n304) );
  VHSR_NOR2_1 U293 ( .A1(n259), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U294 ( .A1(n259), .A2(n248), .B1(n251), .B2(n250), .ZN(n249) );
  VHSR_OAI21_2 U295 ( .A1(n251), .A2(n250), .B(n249), .ZN(n310) );
  VHSR_OAI21_2 U296 ( .A1(n254), .A2(n253), .B(n252), .ZN(n255) );
  VHSR_IN_2 U297 ( .I(n255), .ZN(n309) );
  VHSR_OAI21_2 U298 ( .A1(n265), .A2(n257), .B(n256), .ZN(n258) );
  VHSR_IN_2 U299 ( .I(n258), .ZN(n326) );
  VHSR_IAO21_2 U300 ( .A1(n261), .A2(n260), .B(n259), .ZN(n325) );
  VHSR_AOI31_2 U301 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n334) );
  VHSR_AOI31_2 U302 ( .A1(n268), .A2(n267), .A3(n266), .B(n265), .ZN(n333) );
  VHSR_AOI22_2 U303 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n269) );
  VHSR_NOR2_1 U304 ( .A1(n270), .A2(n269), .ZN(n339) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[4]), .ZN(n311) );
  VHSR_NOR3_2 U306 ( .A1(n317), .A2(n311), .A3(n400), .ZN(n391) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[5]), .A2(b[0]), .ZN(n271) );
  VHSR_OAI32_2 U308 ( .A1(n272), .A2(n319), .A3(n283), .B1(n271), .B2(n272), 
        .ZN(n338) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[6]), .ZN(n359) );
  VHSR_IN_2 U310 ( .I(n359), .ZN(n386) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[4]), .ZN(n302) );
  VHSR_NAND3_2 U312 ( .A1(a[7]), .A2(b[5]), .A3(n302), .ZN(n274) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[4]), .A2(b[6]), .ZN(n301) );
  VHSR_NAND3_2 U314 ( .A1(b[7]), .A2(a[5]), .A3(n301), .ZN(n273) );
  VHSR_CLKNAND2_2 U315 ( .A1(n274), .A2(n273), .ZN(n276) );
  VHSR_MAOI222_2 U316 ( .A(n359), .B(n274), .C(n273), .ZN(n343) );
  VHSR_IN_2 U317 ( .I(n343), .ZN(n275) );
  VHSR_OAI21_2 U318 ( .A1(n386), .A2(n276), .B(n275), .ZN(n290) );
  VHSR_NOR3_2 U319 ( .A1(n282), .A2(n277), .A3(n311), .ZN(n306) );
  VHSR_NOR3_2 U320 ( .A1(n278), .A2(n302), .A3(n277), .ZN(n351) );
  VHSR_AOI22_2 U321 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n279) );
  VHSR_NOR2_1 U322 ( .A1(n351), .A2(n279), .ZN(n286) );
  VHSR_NOR4_2 U323 ( .A1(n283), .A2(n282), .A3(n281), .A4(n280), .ZN(n349) );
  VHSR_AOI22_2 U324 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n284) );
  VHSR_NOR2_1 U325 ( .A1(n349), .A2(n284), .ZN(n285) );
  VHSR_NAND3_2 U326 ( .A1(b[5]), .A2(a[5]), .A3(n311), .ZN(n300) );
  VHSR_MAOI222_2 U327 ( .A(n302), .B(n301), .C(n300), .ZN(n299) );
  VHSR_AND2_2 U328 ( .A1(n292), .A2(n299), .Z(n291) );
  VHSR_AD1_1 U329 ( .A(n306), .B(n286), .CI(n285), .CO(n287), .S(n292) );
  VHSR_CLKNAND2_2 U330 ( .A1(n291), .A2(n287), .ZN(n288) );
  VHSR_AOI22_2 U331 ( .A1(n290), .A2(n289), .B1(n288), .B2(n344), .ZN(n384) );
  VHSR_IAO21_2 U332 ( .A1(n292), .A2(n299), .B(n291), .ZN(n382) );
  VHSR_AD1_1 U333 ( .A(n295), .B(n294), .CI(n293), .CO(n385), .S(n381) );
  VHSR_AD1_1 U334 ( .A(n298), .B(n297), .CI(n296), .CO(n293), .S(n363) );
  VHSR_AOI31_2 U335 ( .A1(n302), .A2(n301), .A3(n300), .B(n299), .ZN(n362) );
  VHSR_AD1_1 U336 ( .A(n305), .B(n304), .CI(n303), .CO(n296), .S(n366) );
  VHSR_AOI22_2 U337 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n307) );
  VHSR_NOR2_1 U338 ( .A1(n307), .A2(n306), .ZN(n365) );
  VHSR_AD1_1 U339 ( .A(n310), .B(n309), .CI(n308), .CO(n303), .S(n369) );
  VHSR_IN_2 U340 ( .I(n311), .ZN(n368) );
  VHSR_CLKNAND2_2 U341 ( .A1(b[2]), .A2(a[2]), .ZN(n327) );
  VHSR_IN_2 U342 ( .I(b[2]), .ZN(n399) );
  VHSR_NOR2_1 U343 ( .A1(n399), .A2(n323), .ZN(n313) );
  VHSR_OAI21_2 U344 ( .A1(n322), .A2(n318), .B(n313), .ZN(n312) );
  VHSR_OAI31_2 U345 ( .A1(n322), .A2(n313), .A3(n318), .B(n312), .ZN(n337) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n320) );
  VHSR_CLKNAND2_2 U347 ( .A1(b[3]), .A2(a[3]), .ZN(n330) );
  VHSR_NOR2_1 U348 ( .A1(n319), .A2(n315), .ZN(n316) );
  VHSR_IN_2 U349 ( .I(n316), .ZN(n314) );
  VHSR_OAI22_2 U350 ( .A1(n327), .A2(n320), .B1(n330), .B2(n314), .ZN(n321) );
  VHSR_OAI22_2 U351 ( .A1(n322), .A2(n400), .B1(n399), .B2(n315), .ZN(n379) );
  VHSR_AOI21_2 U352 ( .A1(a[2]), .A2(b[0]), .B(n316), .ZN(n398) );
  VHSR_NOR3_2 U353 ( .A1(n398), .A2(n400), .A3(n399), .ZN(n401) );
  VHSR_OAI22_2 U354 ( .A1(n319), .A2(n318), .B1(n317), .B2(n323), .ZN(n378) );
  VHSR_AOI21_2 U355 ( .A1(n320), .A2(n327), .B(n321), .ZN(n341) );
  VHSR_CLKNAND2_2 U356 ( .A1(n342), .A2(n341), .ZN(n340) );
  VHSR_CLKNAND2_2 U357 ( .A1(n337), .A2(n336), .ZN(n328) );
  VHSR_AOI211_2 U358 ( .A1(n327), .A2(n328), .B(n323), .C(n322), .ZN(n372) );
  VHSR_AD1_1 U359 ( .A(n326), .B(n325), .CI(n324), .CO(n308), .S(n371) );
  VHSR_IN_2 U360 ( .I(n327), .ZN(n331) );
  VHSR_IN_2 U361 ( .I(n328), .ZN(n335) );
  VHSR_CLKNAND2_2 U362 ( .A1(n335), .A2(n330), .ZN(n329) );
  VHSR_OAI31_2 U363 ( .A1(n331), .A2(n335), .A3(n330), .B(n329), .ZN(n375) );
  VHSR_AD1_1 U364 ( .A(n334), .B(n333), .CI(n332), .CO(n324), .S(n374) );
  VHSR_IAO21_2 U365 ( .A1(n337), .A2(n336), .B(n335), .ZN(n377) );
  VHSR_AD1_1 U366 ( .A(n339), .B(n391), .CI(n338), .CO(n332), .S(n376) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[4]), .A2(b[0]), .ZN(n393) );
  VHSR_OAI21_2 U368 ( .A1(n342), .A2(n341), .B(n340), .ZN(n396) );
  VHSR_AOI211_2 U369 ( .A1(n393), .A2(n392), .B(n391), .C(n396), .ZN(n395) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[7]), .A2(b[6]), .ZN(n346) );
  VHSR_AOI21_2 U371 ( .A1(a[6]), .A2(b[7]), .B(n346), .ZN(n345) );
  VHSR_AOI31_2 U372 ( .A1(a[6]), .A2(n346), .A3(b[7]), .B(n345), .ZN(n347) );
  VHSR_IN_2 U373 ( .I(n347), .ZN(n348) );
  VHSR_OR2_2 U374 ( .A1(n349), .A2(n348), .Z(n350) );
  VHSR_MAOI222_2 U375 ( .A(n351), .B(n349), .C(n348), .ZN(n358) );
  VHSR_OAI21_2 U376 ( .A1(n351), .A2(n350), .B(n358), .ZN(n355) );
  VHSR_CLKXOR2_2 U377 ( .A1(n356), .A2(n355), .Z(n352) );
  VHSR_CLKNAND2_2 U378 ( .A1(n353), .A2(n352), .ZN(n388) );
  VHSR_OAI21_2 U379 ( .A1(n353), .A2(n352), .B(n388), .ZN(n354) );
  VHSR_CLKNAND2_2 U380 ( .A1(a[7]), .A2(b[7]), .ZN(n387) );
  VHSR_NOR2_1 U381 ( .A1(n356), .A2(n355), .ZN(n357) );
  VHSR_AND3_2 U382 ( .A1(n389), .A2(n359), .A3(n388), .Z(n360) );
  VHSR_NOR2_1 U383 ( .A1(n387), .A2(n360), .ZN(product[15]) );
  VHSR_AD1_1 U384 ( .A(n382), .B(n381), .CI(n380), .CO(n383), .S(product[11])
         );
  VHSR_AD1_1 U385 ( .A(n385), .B(n384), .CI(n383), .CO(n353), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U386 ( .A1(n387), .A2(n386), .ZN(n390) );
  VHSR_XOR3_2 U387 ( .A1(n390), .A2(n389), .A3(n388), .Z(product[14]) );
  VHSR_AOI21_2 U388 ( .A1(n393), .A2(n392), .B(n391), .ZN(n394) );
  VHSR_IN_2 U389 ( .I(n394), .ZN(n397) );
  VHSR_AOI21_2 U390 ( .A1(n397), .A2(n396), .B(n395), .ZN(product[4]) );
  VHSR_OAI32_2 U391 ( .A1(n401), .A2(n400), .A3(n399), .B1(n398), .B2(n401), 
        .ZN(product[2]) );
endmodule


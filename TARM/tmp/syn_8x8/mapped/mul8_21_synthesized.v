
module mul8_21 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n214, n215, n216, n217, n218,
         n219, n220, n221, n222, n223, n224, n225, n226, n227, n228, n229,
         n230, n231, n232, n233, n234, n235, n236, n237, n238, n239, n240,
         n241, n242, n243, n244, n245, n246, n247, n248, n249, n250, n251,
         n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
         n263, n264, n265, n266, n267, n268, n269, n270, n271, n272, n273,
         n274, n275, n276, n277, n278, n279, n280, n281, n282, n283, n284,
         n285, n286, n287, n288, n289, n290, n291, n292, n293, n294, n295,
         n296, n297, n298, n299, n300, n301, n302, n303, n304, n305, n306,
         n307, n308, n309, n310, n311, n312, n313, n314, n315, n316, n317,
         n318, n319, n320, n321, n322, n323, n324, n325, n326, n327, n328,
         n329, n330, n331, n332, n333, n334, n335, n336, n337, n338, n339,
         n340, n341, n342, n343, n344, n345, n346, n347, n348, n349, n350,
         n351, n352, n353, n354, n355, n356, n357, n358, n359, n360, n361,
         n362, n363, n364, n365, n366, n367, n368, n369, n370, n371, n372,
         n373, n374, n375, n376, n377, n378, n379, n380, n381, n382, n383,
         n384, n385, n386, n387, n388, n389, n390, n391, n392, n393, n394,
         n395, n396, n397, n398, n399, n400, n401, n402, n403;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U207 ( .A1(n223), .B1(n246), .ZN(n243) );
  VHSR_INOR2_2 U208 ( .A1(n232), .B1(n250), .ZN(n238) );
  VHSR_INOR2_2 U209 ( .A1(n221), .B1(n254), .ZN(n248) );
  VHSR_INOR2_2 U210 ( .A1(n345), .B1(n344), .ZN(n355) );
  VHSR_INAND2_2 U211 ( .A1(n321), .B1(n335), .ZN(n339) );
  VHSR_NOR2_1 U212 ( .A1(n234), .A2(n233), .ZN(n295) );
  VHSR_IOA21_2 U213 ( .A1(n393), .A2(n392), .B(n391), .ZN(n395) );
  VHSR_NOR2_1 U214 ( .A1(n307), .A2(n308), .ZN(n378) );
  VHSR_IN_2 U215 ( .I(n354), .ZN(product[13]) );
  VHSR_CLKN_1 U216 ( .I(n359), .ZN(n360) );
  VHSR_INAND3_1 U217 ( .A1(n386), .B1(n389), .B2(n388), .ZN(n359) );
  VHSR_INOR2_1 U218 ( .A1(n356), .B1(n355), .ZN(n358) );
  VHSR_INAND2_1 U219 ( .A1(n271), .B1(n351), .ZN(n274) );
  VHSR_AD1_1 U220 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(product[6])
         );
  VHSR_AD1_1 U221 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(product[9])
         );
  VHSR_AD1_1 U222 ( .A(n376), .B(n403), .CI(n375), .CO(n337), .S(product[3])
         );
  VHSR_AD1_1 U223 ( .A(n394), .B(n374), .CI(n373), .CO(n370), .S(product[5])
         );
  VHSR_AD1_1 U224 ( .A(n369), .B(n368), .CI(n367), .CO(n377), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U225 ( .A(n363), .B(n362), .CI(n361), .CO(n380), .S(product[10])
         );
  VHSR_CLKNAND2_2 U226 ( .A1(a[7]), .A2(b[3]), .ZN(n234) );
  VHSR_IN_2 U227 ( .I(a[6]), .ZN(n280) );
  VHSR_IN_2 U228 ( .I(b[3]), .ZN(n322) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n283) );
  VHSR_IN_2 U230 ( .I(b[2]), .ZN(n401) );
  VHSR_OAI22_2 U231 ( .A1(n280), .A2(n322), .B1(n283), .B2(n401), .ZN(n245) );
  VHSR_IN_2 U232 ( .I(b[1]), .ZN(n399) );
  VHSR_IN_2 U233 ( .I(a[4]), .ZN(n308) );
  VHSR_NOR2_1 U234 ( .A1(n308), .A2(n401), .ZN(n261) );
  VHSR_IN_2 U235 ( .I(a[5]), .ZN(n306) );
  VHSR_OR3_2 U236 ( .A1(n261), .A2(n322), .A3(n306), .Z(n214) );
  VHSR_OAI21_2 U237 ( .A1(n399), .A2(n283), .B(n214), .ZN(n222) );
  VHSR_NOR4_2 U238 ( .A1(n261), .A2(n234), .A3(n306), .A4(n399), .ZN(n215) );
  VHSR_AOI31_2 U239 ( .A1(b[2]), .A2(a[6]), .A3(n222), .B(n215), .ZN(n223) );
  VHSR_NOR2_1 U240 ( .A1(n280), .A2(n399), .ZN(n218) );
  VHSR_IN_2 U241 ( .I(b[0]), .ZN(n398) );
  VHSR_NOR4_2 U242 ( .A1(n308), .A2(n306), .A3(n399), .A4(n398), .ZN(n270) );
  VHSR_CLKNAND2_2 U243 ( .A1(a[5]), .A2(b[2]), .ZN(n217) );
  VHSR_CLKNAND2_2 U244 ( .A1(a[4]), .A2(b[3]), .ZN(n216) );
  VHSR_NOR4_2 U245 ( .A1(n308), .A2(n306), .A3(n322), .A4(n401), .ZN(n224) );
  VHSR_AOI21_2 U246 ( .A1(n217), .A2(n216), .B(n224), .ZN(n219) );
  VHSR_MAOI222_2 U247 ( .A(n218), .B(n270), .C(n219), .ZN(n221) );
  VHSR_AOI211_2 U248 ( .A1(a[4]), .A2(b[0]), .B(n306), .C(n399), .ZN(n260) );
  VHSR_AOI21_2 U249 ( .A1(n280), .A2(n283), .B(n398), .ZN(n259) );
  VHSR_MAOI222_2 U250 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_OR2_2 U251 ( .A1(n270), .A2(n219), .Z(n220) );
  VHSR_AOI32_2 U252 ( .A1(b[1]), .A2(n221), .A3(a[6]), .B1(n220), .B2(n221), 
        .ZN(n255) );
  VHSR_NOR2_1 U253 ( .A1(n258), .A2(n255), .ZN(n254) );
  VHSR_AOI32_2 U254 ( .A1(b[2]), .A2(n223), .A3(a[6]), .B1(n222), .B2(n223), 
        .ZN(n247) );
  VHSR_NOR2_1 U255 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_IN_2 U256 ( .I(n224), .ZN(n242) );
  VHSR_CLKNAND2_2 U257 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U258 ( .A1(n245), .A2(n241), .ZN(n233) );
  VHSR_IN_2 U259 ( .I(b[7]), .ZN(n272) );
  VHSR_IN_2 U260 ( .I(a[3]), .ZN(n323) );
  VHSR_IN_2 U261 ( .I(b[6]), .ZN(n279) );
  VHSR_IN_2 U262 ( .I(a[2]), .ZN(n319) );
  VHSR_OAI22_2 U263 ( .A1(n279), .A2(n323), .B1(n272), .B2(n319), .ZN(n240) );
  VHSR_NOR2_1 U264 ( .A1(n272), .A2(n319), .ZN(n226) );
  VHSR_IN_2 U265 ( .I(a[1]), .ZN(n397) );
  VHSR_NOR2_1 U266 ( .A1(n279), .A2(n397), .ZN(n225) );
  VHSR_IN_2 U267 ( .I(b[5]), .ZN(n309) );
  VHSR_AOI211_2 U268 ( .A1(a[2]), .A2(b[4]), .B(n309), .C(n323), .ZN(n231) );
  VHSR_OAI22_2 U269 ( .A1(n279), .A2(n319), .B1(n272), .B2(n397), .ZN(n230) );
  VHSR_AOI22_2 U270 ( .A1(n226), .A2(n225), .B1(n231), .B2(n230), .ZN(n232) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[4]), .A2(a[2]), .ZN(n266) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[0]), .ZN(n265) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[4]), .A2(a[0]), .ZN(n392) );
  VHSR_NAND3_2 U274 ( .A1(a[1]), .A2(b[5]), .A3(n392), .ZN(n264) );
  VHSR_MAOI222_2 U275 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_NAND4_2 U276 ( .A1(b[5]), .A2(b[4]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_IN_2 U277 ( .I(b[4]), .ZN(n307) );
  VHSR_OAI22_2 U278 ( .A1(n309), .A2(n319), .B1(n307), .B2(n323), .ZN(n227) );
  VHSR_AND2_2 U279 ( .A1(n237), .A2(n227), .Z(n229) );
  VHSR_IN_2 U280 ( .I(a[0]), .ZN(n402) );
  VHSR_OAI22_2 U281 ( .A1(n279), .A2(n397), .B1(n272), .B2(n402), .ZN(n228) );
  VHSR_NOR3_2 U282 ( .A1(n309), .A2(n397), .A3(n392), .ZN(n268) );
  VHSR_AND2_2 U283 ( .A1(n263), .A2(n257), .Z(n256) );
  VHSR_AD1_1 U284 ( .A(n229), .B(n228), .CI(n268), .CO(n249), .S(n257) );
  VHSR_NOR2_1 U285 ( .A1(n256), .A2(n249), .ZN(n252) );
  VHSR_OAI21_2 U286 ( .A1(n231), .A2(n230), .B(n232), .ZN(n253) );
  VHSR_NOR2_1 U287 ( .A1(n252), .A2(n253), .ZN(n250) );
  VHSR_CLKNAND2_2 U288 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U289 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U290 ( .A1(n272), .A2(n323), .A3(n235), .ZN(n294) );
  VHSR_AOI21_2 U291 ( .A1(n234), .A2(n233), .B(n295), .ZN(n298) );
  VHSR_OAI32_2 U292 ( .A1(n294), .A2(n323), .A3(n272), .B1(n235), .B2(n294), 
        .ZN(n297) );
  VHSR_OAI21_2 U293 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U294 ( .A1(n240), .A2(n239), .ZN(n305) );
  VHSR_OAI21_2 U295 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U296 ( .A1(n245), .A2(n244), .ZN(n304) );
  VHSR_AOI21_2 U297 ( .A1(n248), .A2(n247), .B(n246), .ZN(n314) );
  VHSR_CLKNAND2_2 U298 ( .A1(n256), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U299 ( .A1(n253), .A2(n252), .B1(n251), .B2(n250), .ZN(n313) );
  VHSR_AOI21_2 U300 ( .A1(n258), .A2(n255), .B(n254), .ZN(n327) );
  VHSR_IAO21_2 U301 ( .A1(n263), .A2(n257), .B(n256), .ZN(n326) );
  VHSR_OAI31_2 U302 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n262) );
  VHSR_IN_2 U303 ( .I(n262), .ZN(n330) );
  VHSR_AOI31_2 U304 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n329) );
  VHSR_AOI22_2 U305 ( .A1(b[5]), .A2(a[0]), .B1(b[4]), .B2(a[1]), .ZN(n267) );
  VHSR_NOR2_1 U306 ( .A1(n268), .A2(n267), .ZN(n343) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[5]), .A2(b[0]), .ZN(n269) );
  VHSR_OAI32_2 U308 ( .A1(n270), .A2(n399), .A3(n308), .B1(n269), .B2(n270), 
        .ZN(n342) );
  VHSR_NOR2_1 U309 ( .A1(n398), .A2(n402), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U310 ( .A1(n378), .A2(product[0]), .ZN(n391) );
  VHSR_IN_2 U311 ( .I(n391), .ZN(n341) );
  VHSR_AOI22_2 U312 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n271) );
  VHSR_NAND4_2 U313 ( .A1(a[6]), .A2(a[7]), .A3(b[5]), .A4(b[4]), .ZN(n351) );
  VHSR_NAND3_2 U314 ( .A1(b[5]), .A2(a[5]), .A3(n378), .ZN(n311) );
  VHSR_NAND4_2 U315 ( .A1(b[6]), .A2(b[7]), .A3(a[4]), .A4(a[5]), .ZN(n349) );
  VHSR_NOR2_1 U316 ( .A1(n272), .A2(n308), .ZN(n273) );
  VHSR_AOI32_2 U317 ( .A1(b[6]), .A2(n349), .A3(a[5]), .B1(n273), .B2(n349), 
        .ZN(n276) );
  VHSR_MAOI222_2 U318 ( .A(n274), .B(n311), .C(n276), .ZN(n278) );
  VHSR_AND2_2 U319 ( .A1(n274), .A2(n311), .Z(n275) );
  VHSR_AOI21_2 U320 ( .A1(n276), .A2(n275), .B(n278), .ZN(n277) );
  VHSR_IN_2 U321 ( .I(n277), .ZN(n292) );
  VHSR_NOR2_1 U322 ( .A1(n280), .A2(n307), .ZN(n285) );
  VHSR_NOR2_1 U323 ( .A1(n279), .A2(n308), .ZN(n284) );
  VHSR_CLKNAND2_2 U324 ( .A1(b[5]), .A2(a[5]), .ZN(n286) );
  VHSR_NOR2_1 U325 ( .A1(n378), .A2(n286), .ZN(n301) );
  VHSR_MAOI222_2 U326 ( .A(n285), .B(n284), .C(n301), .ZN(n299) );
  VHSR_NOR2_1 U327 ( .A1(n292), .A2(n299), .ZN(n291) );
  VHSR_NOR2_1 U328 ( .A1(n278), .A2(n291), .ZN(n290) );
  VHSR_NOR2_1 U329 ( .A1(n280), .A2(n279), .ZN(n386) );
  VHSR_IN_2 U330 ( .I(n284), .ZN(n281) );
  VHSR_NAND3_2 U331 ( .A1(a[5]), .A2(b[7]), .A3(n281), .ZN(n282) );
  VHSR_OAI31_2 U332 ( .A1(n285), .A2(n309), .A3(n283), .B(n282), .ZN(n288) );
  VHSR_CLKNAND2_2 U333 ( .A1(a[7]), .A2(b[7]), .ZN(n387) );
  VHSR_OR2_2 U334 ( .A1(n285), .A2(n284), .Z(n300) );
  VHSR_NOR3_2 U335 ( .A1(n387), .A2(n300), .A3(n286), .ZN(n287) );
  VHSR_AOI31_2 U336 ( .A1(b[6]), .A2(a[6]), .A3(n288), .B(n287), .ZN(n345) );
  VHSR_OAI21_2 U337 ( .A1(n386), .A2(n288), .B(n345), .ZN(n289) );
  VHSR_NOR2_1 U338 ( .A1(n290), .A2(n289), .ZN(n344) );
  VHSR_AOI21_2 U339 ( .A1(n290), .A2(n289), .B(n344), .ZN(n384) );
  VHSR_AOI21_2 U340 ( .A1(n292), .A2(n299), .B(n291), .ZN(n382) );
  VHSR_AD1_1 U341 ( .A(n295), .B(n294), .CI(n293), .CO(n385), .S(n381) );
  VHSR_AD1_1 U342 ( .A(n298), .B(n297), .CI(n296), .CO(n293), .S(n363) );
  VHSR_OAI21_2 U343 ( .A1(n301), .A2(n300), .B(n299), .ZN(n302) );
  VHSR_IN_2 U344 ( .I(n302), .ZN(n362) );
  VHSR_AD1_1 U345 ( .A(n305), .B(n304), .CI(n303), .CO(n296), .S(n366) );
  VHSR_OAI22_2 U346 ( .A1(n309), .A2(n308), .B1(n307), .B2(n306), .ZN(n310) );
  VHSR_AND2_2 U347 ( .A1(n311), .A2(n310), .Z(n365) );
  VHSR_AD1_1 U348 ( .A(n314), .B(n313), .CI(n312), .CO(n303), .S(n379) );
  VHSR_NOR2_1 U349 ( .A1(n401), .A2(n319), .ZN(n334) );
  VHSR_IN_2 U350 ( .I(n334), .ZN(n324) );
  VHSR_NOR2_1 U351 ( .A1(n401), .A2(n323), .ZN(n316) );
  VHSR_OAI21_2 U352 ( .A1(n322), .A2(n319), .B(n316), .ZN(n315) );
  VHSR_OAI31_2 U353 ( .A1(n322), .A2(n316), .A3(n319), .B(n315), .ZN(n340) );
  VHSR_AOI22_2 U354 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n320) );
  VHSR_CLKNAND2_2 U355 ( .A1(b[3]), .A2(a[3]), .ZN(n333) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[1]), .A2(a[1]), .ZN(n317) );
  VHSR_OAI22_2 U357 ( .A1(n324), .A2(n320), .B1(n333), .B2(n317), .ZN(n321) );
  VHSR_OAI22_2 U358 ( .A1(n322), .A2(n402), .B1(n401), .B2(n397), .ZN(n376) );
  VHSR_OAI21_2 U359 ( .A1(n319), .A2(n398), .B(n317), .ZN(n318) );
  VHSR_IN_2 U360 ( .I(n318), .ZN(n400) );
  VHSR_NOR3_2 U361 ( .A1(n400), .A2(n402), .A3(n401), .ZN(n403) );
  VHSR_OAI22_2 U362 ( .A1(n399), .A2(n319), .B1(n398), .B2(n323), .ZN(n375) );
  VHSR_AOI21_2 U363 ( .A1(n320), .A2(n324), .B(n321), .ZN(n336) );
  VHSR_CLKNAND2_2 U364 ( .A1(n337), .A2(n336), .ZN(n335) );
  VHSR_CLKNAND2_2 U365 ( .A1(n340), .A2(n339), .ZN(n331) );
  VHSR_AOI211_2 U366 ( .A1(n324), .A2(n331), .B(n323), .C(n322), .ZN(n369) );
  VHSR_AD1_1 U367 ( .A(n327), .B(n326), .CI(n325), .CO(n312), .S(n368) );
  VHSR_AD1_1 U368 ( .A(n330), .B(n329), .CI(n328), .CO(n325), .S(n372) );
  VHSR_IN_2 U369 ( .I(n331), .ZN(n338) );
  VHSR_CLKNAND2_2 U370 ( .A1(n338), .A2(n333), .ZN(n332) );
  VHSR_OAI31_2 U371 ( .A1(n334), .A2(n338), .A3(n333), .B(n332), .ZN(n371) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[4]), .A2(b[0]), .ZN(n393) );
  VHSR_OAI21_2 U373 ( .A1(n337), .A2(n336), .B(n335), .ZN(n396) );
  VHSR_AOI211_2 U374 ( .A1(n393), .A2(n392), .B(n341), .C(n396), .ZN(n394) );
  VHSR_IAO21_2 U375 ( .A1(n340), .A2(n339), .B(n338), .ZN(n374) );
  VHSR_AD1_1 U376 ( .A(n343), .B(n342), .CI(n341), .CO(n328), .S(n373) );
  VHSR_CLKNAND2_2 U377 ( .A1(b[6]), .A2(a[7]), .ZN(n347) );
  VHSR_AOI21_2 U378 ( .A1(a[6]), .A2(b[7]), .B(n347), .ZN(n346) );
  VHSR_AOI31_2 U379 ( .A1(a[6]), .A2(n347), .A3(b[7]), .B(n346), .ZN(n348) );
  VHSR_AND2_2 U380 ( .A1(n349), .A2(n348), .Z(n350) );
  VHSR_MAOI222_2 U381 ( .A(n351), .B(n349), .C(n348), .ZN(n357) );
  VHSR_AOI21_2 U382 ( .A1(n351), .A2(n350), .B(n357), .ZN(n356) );
  VHSR_XNOR2_2 U383 ( .A1(n355), .A2(n356), .ZN(n352) );
  VHSR_CLKNAND2_2 U384 ( .A1(n353), .A2(n352), .ZN(n388) );
  VHSR_OAI21_2 U385 ( .A1(n353), .A2(n352), .B(n388), .ZN(n354) );
  VHSR_NOR2_1 U386 ( .A1(n358), .A2(n357), .ZN(n389) );
  VHSR_NOR2_1 U387 ( .A1(n387), .A2(n360), .ZN(product[15]) );
  VHSR_AD1_1 U388 ( .A(n379), .B(n378), .CI(n377), .CO(n364), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U389 ( .A(n382), .B(n381), .CI(n380), .CO(n383), .S(product[11])
         );
  VHSR_AD1_1 U390 ( .A(n385), .B(n384), .CI(n383), .CO(n353), .S(product[12])
         );
  VHSR_NOR2_1 U391 ( .A1(n387), .A2(n386), .ZN(n390) );
  VHSR_XOR3_2 U392 ( .A1(n390), .A2(n389), .A3(n388), .Z(product[14]) );
  VHSR_AOI21_2 U393 ( .A1(n396), .A2(n395), .B(n394), .ZN(product[4]) );
  VHSR_OAI22_2 U394 ( .A1(n399), .A2(n402), .B1(n398), .B2(n397), .ZN(
        product[1]) );
  VHSR_OAI32_2 U395 ( .A1(n403), .A2(n402), .A3(n401), .B1(n400), .B2(n403), 
        .ZN(product[2]) );
endmodule


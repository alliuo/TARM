
module mul8_87 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n217, n218,
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
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_IN_2 U207 ( .I(n245), .ZN(n219) );
  VHSR_INAND3_2 U208 ( .A1(n267), .B1(a[5]), .B2(b[3]), .ZN(n217) );
  VHSR_NOR2_1 U209 ( .A1(n315), .A2(n283), .ZN(n267) );
  VHSR_INOR2_2 U210 ( .A1(n227), .B1(n253), .ZN(n246) );
  VHSR_INAND2_2 U211 ( .A1(n322), .B1(n321), .ZN(n323) );
  VHSR_INOR2_2 U212 ( .A1(n225), .B1(n256), .ZN(n255) );
  VHSR_NOR2_1 U213 ( .A1(n344), .A2(n343), .ZN(n356) );
  VHSR_NOR2_1 U214 ( .A1(n315), .A2(n316), .ZN(n332) );
  VHSR_NOR2_1 U215 ( .A1(n237), .A2(n236), .ZN(n296) );
  VHSR_INOR2_2 U216 ( .A1(n358), .B1(n357), .ZN(n389) );
  VHSR_IN_2 U217 ( .I(n354), .ZN(product[13]) );
  VHSR_NOR2_2 U218 ( .A1(n273), .A2(n282), .ZN(n386) );
  VHSR_MOAI22_1 U219 ( .A1(n278), .A2(n316), .B1(b[4]), .B2(a[3]), .ZN(n229)
         );
  VHSR_AD1_1 U220 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(product[9])
         );
  VHSR_AD1_1 U221 ( .A(n373), .B(n401), .CI(n372), .CO(n337), .S(product[3])
         );
  VHSR_AD1_1 U222 ( .A(n395), .B(n371), .CI(n370), .CO(n374), .S(product[5])
         );
  VHSR_AD1_1 U223 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n363), .B(n362), .CI(n361), .CO(n380), .S(product[10])
         );
  VHSR_IN_2 U225 ( .I(b[0]), .ZN(n399) );
  VHSR_IN_2 U226 ( .I(a[1]), .ZN(n318) );
  VHSR_NOR2_1 U227 ( .A1(n399), .A2(n318), .ZN(product[1]) );
  VHSR_CLKNAND2_2 U228 ( .A1(b[3]), .A2(a[7]), .ZN(n237) );
  VHSR_IN_2 U229 ( .I(b[3]), .ZN(n319) );
  VHSR_IN_2 U230 ( .I(a[6]), .ZN(n273) );
  VHSR_IN_2 U231 ( .I(a[7]), .ZN(n279) );
  VHSR_IN_2 U232 ( .I(b[2]), .ZN(n315) );
  VHSR_OAI22_2 U233 ( .A1(n319), .A2(n273), .B1(n279), .B2(n315), .ZN(n248) );
  VHSR_IN_2 U234 ( .I(b[1]), .ZN(n400) );
  VHSR_IN_2 U235 ( .I(a[4]), .ZN(n283) );
  VHSR_OAI21_2 U236 ( .A1(n400), .A2(n279), .B(n217), .ZN(n226) );
  VHSR_IN_2 U237 ( .I(a[5]), .ZN(n284) );
  VHSR_NOR4_2 U238 ( .A1(n267), .A2(n284), .A3(n237), .A4(n400), .ZN(n218) );
  VHSR_AOI31_2 U239 ( .A1(b[2]), .A2(a[6]), .A3(n226), .B(n218), .ZN(n227) );
  VHSR_NOR2_1 U240 ( .A1(n273), .A2(n400), .ZN(n222) );
  VHSR_NOR4_2 U241 ( .A1(n284), .A2(n283), .A3(n400), .A4(n399), .ZN(n272) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[2]), .A2(a[5]), .ZN(n221) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[3]), .A2(a[4]), .ZN(n220) );
  VHSR_NAND4_2 U244 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n245) );
  VHSR_AOI21_2 U245 ( .A1(n221), .A2(n220), .B(n219), .ZN(n223) );
  VHSR_MAOI222_2 U246 ( .A(n222), .B(n272), .C(n223), .ZN(n225) );
  VHSR_AOI211_2 U247 ( .A1(a[4]), .A2(b[0]), .B(n284), .C(n400), .ZN(n266) );
  VHSR_AOI21_2 U248 ( .A1(n279), .A2(n273), .B(n399), .ZN(n265) );
  VHSR_MAOI222_2 U249 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_OR2_2 U250 ( .A1(n272), .A2(n223), .Z(n224) );
  VHSR_AOI32_2 U251 ( .A1(b[1]), .A2(n225), .A3(a[6]), .B1(n224), .B2(n225), 
        .ZN(n257) );
  VHSR_NOR2_1 U252 ( .A1(n264), .A2(n257), .ZN(n256) );
  VHSR_AOI32_2 U253 ( .A1(b[2]), .A2(n227), .A3(a[6]), .B1(n226), .B2(n227), 
        .ZN(n254) );
  VHSR_NOR2_1 U254 ( .A1(n255), .A2(n254), .ZN(n253) );
  VHSR_CLKNAND2_2 U255 ( .A1(n246), .A2(n245), .ZN(n244) );
  VHSR_CLKNAND2_2 U256 ( .A1(n248), .A2(n244), .ZN(n236) );
  VHSR_IN_2 U257 ( .I(b[7]), .ZN(n281) );
  VHSR_IN_2 U258 ( .I(a[3]), .ZN(n317) );
  VHSR_IN_2 U259 ( .I(b[6]), .ZN(n282) );
  VHSR_IN_2 U260 ( .I(a[2]), .ZN(n316) );
  VHSR_OAI22_2 U261 ( .A1(n282), .A2(n317), .B1(n281), .B2(n316), .ZN(n243) );
  VHSR_AOI22_2 U262 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n234) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_NAND3_2 U264 ( .A1(a[3]), .A2(b[5]), .A3(n263), .ZN(n233) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[7]), .A2(a[2]), .ZN(n228) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[6]), .A2(a[1]), .ZN(n230) );
  VHSR_OAI22_2 U267 ( .A1(n234), .A2(n233), .B1(n228), .B2(n230), .ZN(n235) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[4]), .A2(a[0]), .ZN(n392) );
  VHSR_NAND3_2 U269 ( .A1(a[1]), .A2(b[5]), .A3(n392), .ZN(n262) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[0]), .ZN(n261) );
  VHSR_MAOI222_2 U271 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NAND4_2 U272 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n240) );
  VHSR_IN_2 U273 ( .I(b[5]), .ZN(n278) );
  VHSR_AND2_2 U274 ( .A1(n240), .A2(n229), .Z(n232) );
  VHSR_IN_2 U275 ( .I(a[0]), .ZN(n398) );
  VHSR_OAI21_2 U276 ( .A1(n281), .A2(n398), .B(n230), .ZN(n231) );
  VHSR_NOR3_2 U277 ( .A1(n278), .A2(n318), .A3(n392), .ZN(n270) );
  VHSR_AND2_2 U278 ( .A1(n260), .A2(n259), .Z(n258) );
  VHSR_AD1_1 U279 ( .A(n232), .B(n231), .CI(n270), .CO(n249), .S(n259) );
  VHSR_AOI21_2 U280 ( .A1(n234), .A2(n233), .B(n235), .ZN(n252) );
  VHSR_OAI32_2 U281 ( .A1(n235), .A2(n258), .A3(n249), .B1(n252), .B2(n235), 
        .ZN(n241) );
  VHSR_CLKNAND2_2 U282 ( .A1(n241), .A2(n240), .ZN(n239) );
  VHSR_CLKNAND2_2 U283 ( .A1(n243), .A2(n239), .ZN(n238) );
  VHSR_NOR3_2 U284 ( .A1(n281), .A2(n317), .A3(n238), .ZN(n295) );
  VHSR_AOI21_2 U285 ( .A1(n237), .A2(n236), .B(n296), .ZN(n299) );
  VHSR_OAI32_2 U286 ( .A1(n295), .A2(n317), .A3(n281), .B1(n238), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U287 ( .A1(n241), .A2(n240), .B(n239), .ZN(n242) );
  VHSR_XNOR2_2 U288 ( .A1(n243), .A2(n242), .ZN(n306) );
  VHSR_OAI21_2 U289 ( .A1(n246), .A2(n245), .B(n244), .ZN(n247) );
  VHSR_XNOR2_2 U290 ( .A1(n248), .A2(n247), .ZN(n305) );
  VHSR_NOR2_1 U291 ( .A1(n258), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U292 ( .A1(n258), .A2(n249), .B1(n252), .B2(n251), .ZN(n250) );
  VHSR_OAI21_2 U293 ( .A1(n252), .A2(n251), .B(n250), .ZN(n311) );
  VHSR_AOI21_2 U294 ( .A1(n255), .A2(n254), .B(n253), .ZN(n310) );
  VHSR_AOI21_2 U295 ( .A1(n264), .A2(n257), .B(n256), .ZN(n328) );
  VHSR_IAO21_2 U296 ( .A1(n260), .A2(n259), .B(n258), .ZN(n327) );
  VHSR_AOI31_2 U297 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n335) );
  VHSR_OAI31_2 U298 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n268) );
  VHSR_IN_2 U299 ( .I(n268), .ZN(n334) );
  VHSR_AOI22_2 U300 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n269) );
  VHSR_NOR2_1 U301 ( .A1(n270), .A2(n269), .ZN(n339) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[4]), .A2(b[4]), .ZN(n312) );
  VHSR_NOR3_2 U303 ( .A1(n399), .A2(n312), .A3(n398), .ZN(n391) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[1]), .ZN(n271) );
  VHSR_OAI32_2 U305 ( .A1(n272), .A2(n399), .A3(n284), .B1(n271), .B2(n272), 
        .ZN(n338) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[4]), .A2(b[6]), .ZN(n302) );
  VHSR_NAND3_2 U307 ( .A1(b[7]), .A2(a[5]), .A3(n302), .ZN(n275) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_NAND3_2 U309 ( .A1(a[7]), .A2(b[5]), .A3(n303), .ZN(n274) );
  VHSR_CLKNAND2_2 U310 ( .A1(n275), .A2(n274), .ZN(n277) );
  VHSR_IN_2 U311 ( .I(n386), .ZN(n359) );
  VHSR_MAOI222_2 U312 ( .A(n359), .B(n275), .C(n274), .ZN(n343) );
  VHSR_IN_2 U313 ( .I(n343), .ZN(n276) );
  VHSR_OAI21_2 U314 ( .A1(n386), .A2(n277), .B(n276), .ZN(n291) );
  VHSR_NOR3_2 U315 ( .A1(n284), .A2(n278), .A3(n312), .ZN(n307) );
  VHSR_NOR3_2 U316 ( .A1(n279), .A2(n303), .A3(n278), .ZN(n351) );
  VHSR_AOI22_2 U317 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n280) );
  VHSR_NOR2_1 U318 ( .A1(n351), .A2(n280), .ZN(n287) );
  VHSR_NOR4_2 U319 ( .A1(n284), .A2(n283), .A3(n282), .A4(n281), .ZN(n349) );
  VHSR_AOI22_2 U320 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n285) );
  VHSR_NOR2_1 U321 ( .A1(n349), .A2(n285), .ZN(n286) );
  VHSR_NAND3_2 U322 ( .A1(b[5]), .A2(a[5]), .A3(n312), .ZN(n301) );
  VHSR_MAOI222_2 U323 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_AND2_2 U324 ( .A1(n293), .A2(n300), .Z(n292) );
  VHSR_AD1_1 U325 ( .A(n307), .B(n287), .CI(n286), .CO(n288), .S(n293) );
  VHSR_NOR2_1 U326 ( .A1(n292), .A2(n288), .ZN(n290) );
  VHSR_CLKNAND2_2 U327 ( .A1(n292), .A2(n288), .ZN(n289) );
  VHSR_NOR2_1 U328 ( .A1(n290), .A2(n291), .ZN(n344) );
  VHSR_AOI22_2 U329 ( .A1(n291), .A2(n290), .B1(n289), .B2(n344), .ZN(n384) );
  VHSR_IAO21_2 U330 ( .A1(n293), .A2(n300), .B(n292), .ZN(n382) );
  VHSR_AD1_1 U331 ( .A(n296), .B(n295), .CI(n294), .CO(n385), .S(n381) );
  VHSR_AD1_1 U332 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n363) );
  VHSR_AOI31_2 U333 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n362) );
  VHSR_AD1_1 U334 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n366) );
  VHSR_AOI22_2 U335 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n308) );
  VHSR_NOR2_1 U336 ( .A1(n308), .A2(n307), .ZN(n365) );
  VHSR_AD1_1 U337 ( .A(n311), .B(n310), .CI(n309), .CO(n304), .S(n369) );
  VHSR_IN_2 U338 ( .I(n312), .ZN(n368) );
  VHSR_NOR4_2 U339 ( .A1(n319), .A2(n315), .A3(n318), .A4(n398), .ZN(n342) );
  VHSR_CLKNAND2_2 U340 ( .A1(b[3]), .A2(a[3]), .ZN(n330) );
  VHSR_IN_2 U341 ( .I(n332), .ZN(n321) );
  VHSR_AOI22_2 U342 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n313) );
  VHSR_IAO21_2 U343 ( .A1(n330), .A2(n321), .B(n313), .ZN(n341) );
  VHSR_CLKNAND2_2 U344 ( .A1(b[2]), .A2(a[1]), .ZN(n314) );
  VHSR_OAI32_2 U345 ( .A1(n342), .A2(n398), .A3(n319), .B1(n314), .B2(n342), 
        .ZN(n373) );
  VHSR_AOI22_2 U346 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n403) );
  VHSR_NOR3_2 U347 ( .A1(n403), .A2(n398), .A3(n315), .ZN(n401) );
  VHSR_OAI22_2 U348 ( .A1(n400), .A2(n316), .B1(n399), .B2(n317), .ZN(n372) );
  VHSR_IN_2 U349 ( .I(n337), .ZN(n325) );
  VHSR_NOR2_1 U350 ( .A1(n400), .A2(n317), .ZN(n320) );
  VHSR_AOI211_2 U351 ( .A1(b[2]), .A2(a[0]), .B(n319), .C(n318), .ZN(n322) );
  VHSR_MAOI222_2 U352 ( .A(n320), .B(n332), .C(n322), .ZN(n324) );
  VHSR_AOI32_2 U353 ( .A1(a[3]), .A2(n324), .A3(b[1]), .B1(n323), .B2(n324), 
        .ZN(n336) );
  VHSR_OAI21_2 U354 ( .A1(n325), .A2(n336), .B(n324), .ZN(n340) );
  VHSR_IAO21_2 U355 ( .A1(n332), .A2(n331), .B(n330), .ZN(n379) );
  VHSR_AD1_1 U356 ( .A(n328), .B(n327), .CI(n326), .CO(n309), .S(n378) );
  VHSR_OAI21_2 U357 ( .A1(n332), .A2(n330), .B(n331), .ZN(n329) );
  VHSR_OAI31_2 U358 ( .A1(n332), .A2(n331), .A3(n330), .B(n329), .ZN(n376) );
  VHSR_AD1_1 U359 ( .A(n335), .B(n334), .CI(n333), .CO(n326), .S(n375) );
  VHSR_CLKNAND2_2 U360 ( .A1(a[4]), .A2(b[0]), .ZN(n393) );
  VHSR_CLKXOR2_2 U361 ( .A1(n337), .A2(n336), .Z(n397) );
  VHSR_AOI211_2 U362 ( .A1(n393), .A2(n392), .B(n391), .C(n397), .ZN(n395) );
  VHSR_AD1_1 U363 ( .A(n339), .B(n391), .CI(n338), .CO(n333), .S(n371) );
  VHSR_AD1_1 U364 ( .A(n342), .B(n341), .CI(n340), .CO(n331), .S(n370) );
  VHSR_CLKNAND2_2 U365 ( .A1(a[6]), .A2(b[7]), .ZN(n346) );
  VHSR_AOI21_2 U366 ( .A1(a[7]), .A2(b[6]), .B(n346), .ZN(n345) );
  VHSR_AOI31_2 U367 ( .A1(a[7]), .A2(n346), .A3(b[6]), .B(n345), .ZN(n347) );
  VHSR_IN_2 U368 ( .I(n347), .ZN(n348) );
  VHSR_OR2_2 U369 ( .A1(n349), .A2(n348), .Z(n350) );
  VHSR_MAOI222_2 U370 ( .A(n351), .B(n349), .C(n348), .ZN(n358) );
  VHSR_OAI21_2 U371 ( .A1(n351), .A2(n350), .B(n358), .ZN(n355) );
  VHSR_CLKXOR2_2 U372 ( .A1(n356), .A2(n355), .Z(n352) );
  VHSR_CLKNAND2_2 U373 ( .A1(n353), .A2(n352), .ZN(n388) );
  VHSR_OAI21_2 U374 ( .A1(n353), .A2(n352), .B(n388), .ZN(n354) );
  VHSR_CLKNAND2_2 U375 ( .A1(a[7]), .A2(b[7]), .ZN(n387) );
  VHSR_NOR2_1 U376 ( .A1(n356), .A2(n355), .ZN(n357) );
  VHSR_AND3_2 U377 ( .A1(n389), .A2(n359), .A3(n388), .Z(n360) );
  VHSR_NOR2_1 U378 ( .A1(n387), .A2(n360), .ZN(product[15]) );
  VHSR_AD1_1 U379 ( .A(n376), .B(n375), .CI(n374), .CO(n377), .S(product[6])
         );
  VHSR_AD1_1 U380 ( .A(n379), .B(n378), .CI(n377), .CO(n367), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U381 ( .A(n382), .B(n381), .CI(n380), .CO(n383), .S(product[11])
         );
  VHSR_AD1_1 U382 ( .A(n385), .B(n384), .CI(n383), .CO(n353), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U383 ( .A1(n387), .A2(n386), .ZN(n390) );
  VHSR_XOR3_2 U384 ( .A1(n390), .A2(n389), .A3(n388), .Z(product[14]) );
  VHSR_AOI21_2 U385 ( .A1(n393), .A2(n392), .B(n391), .ZN(n394) );
  VHSR_IN_2 U386 ( .I(n394), .ZN(n396) );
  VHSR_AOI21_2 U387 ( .A1(n397), .A2(n396), .B(n395), .ZN(product[4]) );
  VHSR_AOI21_2 U388 ( .A1(n400), .A2(n399), .B(n398), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U389 ( .A1(b[2]), .A2(a[0]), .ZN(n402) );
  VHSR_AOI21_2 U390 ( .A1(n403), .A2(n402), .B(n401), .ZN(product[2]) );
endmodule


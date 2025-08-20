
module mul8_26 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n214, n215,
         n216, n217, n218, n219, n220, n221, n222, n223, n224, n225, n226,
         n227, n228, n229, n230, n231, n232, n233, n234, n235, n236, n237,
         n238, n239, n240, n241, n242, n243, n244, n245, n246, n247, n248,
         n249, n250, n251, n252, n253, n254, n255, n256, n257, n258, n259,
         n260, n261, n262, n263, n264, n265, n266, n267, n268, n269, n270,
         n271, n272, n273, n274, n275, n276, n277, n278, n279, n280, n281,
         n282, n283, n284, n285, n286, n287, n288, n289, n290, n291, n292,
         n293, n294, n295, n296, n297, n298, n299, n300, n301, n302, n303,
         n304, n305, n306, n307, n308, n309, n310, n311, n312, n313, n314,
         n315, n316, n317, n318, n319, n320, n321, n322, n323, n324, n325,
         n326, n327, n328, n329, n330, n331, n332, n333, n334, n335, n336,
         n337, n338, n339, n340, n341, n342, n343, n344, n345, n346, n347,
         n348, n349, n350, n351, n352, n353, n354, n355, n356, n357, n358,
         n359, n360, n361, n362, n363, n364, n365, n366, n367, n368, n369,
         n370, n371, n372, n373, n374, n375, n376, n377, n378, n379, n380,
         n381, n382, n383, n384, n385, n386, n387, n388, n389, n390, n391,
         n392, n393, n394, n395, n396, n397, n398, n399, n400, n401, n402,
         n403, n404, n405, n406;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U205 ( .A1(n326), .B1(n331), .ZN(n327) );
  VHSR_NOR2_1 U206 ( .A1(n350), .A2(n349), .ZN(n362) );
  VHSR_NOR2_1 U207 ( .A1(n399), .A2(n398), .ZN(n397) );
  VHSR_INAND3_2 U208 ( .A1(n377), .B1(b[5]), .B2(a[5]), .ZN(n305) );
  VHSR_NOR2_1 U209 ( .A1(n285), .A2(n279), .ZN(n377) );
  VHSR_IN_2 U210 ( .I(n360), .ZN(product[13]) );
  VHSR_INOR2_1 U211 ( .A1(n364), .B1(n363), .ZN(n395) );
  VHSR_NOR2_2 U212 ( .A1(n297), .A2(n296), .ZN(n295) );
  VHSR_NOR2_2 U213 ( .A1(n316), .A2(n320), .ZN(n338) );
  VHSR_AD1_1 U214 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(product[10])
         );
  VHSR_AD1_1 U216 ( .A(n382), .B(n404), .CI(n381), .CO(n343), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n401), .B(n380), .CI(n379), .CO(n383), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(product[9])
         );
  VHSR_AD1_1 U219 ( .A(n369), .B(n368), .CI(n367), .CO(n389), .S(product[11])
         );
  VHSR_IN_2 U220 ( .I(b[0]), .ZN(n319) );
  VHSR_IN_2 U221 ( .I(a[1]), .ZN(n323) );
  VHSR_NOR2_1 U222 ( .A1(n319), .A2(n323), .ZN(product[1]) );
  VHSR_IN_2 U223 ( .I(b[1]), .ZN(n322) );
  VHSR_IN_2 U224 ( .I(a[0]), .ZN(n318) );
  VHSR_NOR2_1 U225 ( .A1(n322), .A2(n318), .ZN(product[0]) );
  VHSR_AOI22_2 U226 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n249) );
  VHSR_IN_2 U227 ( .I(b[3]), .ZN(n324) );
  VHSR_IN_2 U228 ( .I(b[2]), .ZN(n316) );
  VHSR_IN_2 U229 ( .I(a[5]), .ZN(n286) );
  VHSR_IN_2 U230 ( .I(a[4]), .ZN(n285) );
  VHSR_NOR4_2 U231 ( .A1(n324), .A2(n316), .A3(n286), .A4(n285), .ZN(n247) );
  VHSR_IN_2 U232 ( .I(a[7]), .ZN(n281) );
  VHSR_NOR2_1 U233 ( .A1(n281), .A2(n322), .ZN(n215) );
  VHSR_AOI211_2 U234 ( .A1(b[2]), .A2(a[4]), .B(n324), .C(n286), .ZN(n216) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[6]), .A2(b[2]), .ZN(n218) );
  VHSR_IN_2 U236 ( .I(n218), .ZN(n214) );
  VHSR_MAOI222_2 U237 ( .A(n215), .B(n216), .C(n214), .ZN(n228) );
  VHSR_AOI21_2 U238 ( .A1(b[1]), .A2(a[7]), .B(n216), .ZN(n219) );
  VHSR_IN_2 U239 ( .I(n228), .ZN(n217) );
  VHSR_AOI21_2 U240 ( .A1(n219), .A2(n218), .B(n217), .ZN(n256) );
  VHSR_CLKNAND2_2 U241 ( .A1(a[6]), .A2(b[1]), .ZN(n225) );
  VHSR_IN_2 U242 ( .I(n225), .ZN(n222) );
  VHSR_NOR4_2 U243 ( .A1(n286), .A2(n285), .A3(n322), .A4(n319), .ZN(n274) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[2]), .A2(a[5]), .ZN(n221) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[3]), .A2(a[4]), .ZN(n220) );
  VHSR_AOI21_2 U246 ( .A1(n221), .A2(n220), .B(n247), .ZN(n223) );
  VHSR_MAOI222_2 U247 ( .A(n222), .B(n274), .C(n223), .ZN(n227) );
  VHSR_CLKNAND2_2 U248 ( .A1(b[2]), .A2(a[4]), .ZN(n270) );
  VHSR_OAI21_2 U249 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n269) );
  VHSR_CLKNAND2_2 U250 ( .A1(a[4]), .A2(b[0]), .ZN(n399) );
  VHSR_NAND3_2 U251 ( .A1(b[1]), .A2(a[5]), .A3(n399), .ZN(n268) );
  VHSR_MAOI222_2 U252 ( .A(n270), .B(n269), .C(n268), .ZN(n267) );
  VHSR_NOR2_1 U253 ( .A1(n274), .A2(n223), .ZN(n226) );
  VHSR_IN_2 U254 ( .I(n227), .ZN(n224) );
  VHSR_AOI21_2 U255 ( .A1(n226), .A2(n225), .B(n224), .ZN(n259) );
  VHSR_CLKNAND2_2 U256 ( .A1(n267), .A2(n259), .ZN(n258) );
  VHSR_CLKNAND2_2 U257 ( .A1(n227), .A2(n258), .ZN(n255) );
  VHSR_CLKNAND2_2 U258 ( .A1(n256), .A2(n255), .ZN(n254) );
  VHSR_CLKNAND2_2 U259 ( .A1(n228), .A2(n254), .ZN(n246) );
  VHSR_NOR2_1 U260 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_NOR2_1 U261 ( .A1(n249), .A2(n245), .ZN(n238) );
  VHSR_AND3_2 U262 ( .A1(n238), .A2(b[3]), .A3(a[7]), .Z(n300) );
  VHSR_IN_2 U263 ( .I(b[7]), .ZN(n283) );
  VHSR_IN_2 U264 ( .I(a[3]), .ZN(n321) );
  VHSR_IN_2 U265 ( .I(b[6]), .ZN(n284) );
  VHSR_IN_2 U266 ( .I(a[2]), .ZN(n320) );
  VHSR_OAI22_2 U267 ( .A1(n284), .A2(n321), .B1(n283), .B2(n320), .ZN(n244) );
  VHSR_AOI22_2 U268 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n235) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[4]), .A2(a[2]), .ZN(n266) );
  VHSR_NAND3_2 U270 ( .A1(a[3]), .A2(b[5]), .A3(n266), .ZN(n234) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[7]), .A2(a[2]), .ZN(n229) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[1]), .ZN(n231) );
  VHSR_OAI22_2 U273 ( .A1(n235), .A2(n234), .B1(n229), .B2(n231), .ZN(n236) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[4]), .A2(a[0]), .ZN(n398) );
  VHSR_NAND3_2 U275 ( .A1(a[1]), .A2(b[5]), .A3(n398), .ZN(n265) );
  VHSR_CLKNAND2_2 U276 ( .A1(b[6]), .A2(a[0]), .ZN(n264) );
  VHSR_MAOI222_2 U277 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_NAND4_2 U278 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n241) );
  VHSR_IN_2 U279 ( .I(b[4]), .ZN(n279) );
  VHSR_IN_2 U280 ( .I(b[5]), .ZN(n280) );
  VHSR_OAI22_2 U281 ( .A1(n279), .A2(n321), .B1(n280), .B2(n320), .ZN(n230) );
  VHSR_AND2_2 U282 ( .A1(n241), .A2(n230), .Z(n233) );
  VHSR_OAI21_2 U283 ( .A1(n283), .A2(n318), .B(n231), .ZN(n232) );
  VHSR_NOR4_2 U284 ( .A1(n279), .A2(n280), .A3(n323), .A4(n318), .ZN(n272) );
  VHSR_AND2_2 U285 ( .A1(n263), .A2(n262), .Z(n261) );
  VHSR_AD1_1 U286 ( .A(n233), .B(n232), .CI(n272), .CO(n250), .S(n262) );
  VHSR_AOI21_2 U287 ( .A1(n235), .A2(n234), .B(n236), .ZN(n253) );
  VHSR_OAI32_2 U288 ( .A1(n236), .A2(n261), .A3(n250), .B1(n253), .B2(n236), 
        .ZN(n242) );
  VHSR_CLKNAND2_2 U289 ( .A1(n242), .A2(n241), .ZN(n240) );
  VHSR_CLKNAND2_2 U290 ( .A1(n244), .A2(n240), .ZN(n239) );
  VHSR_NOR3_2 U291 ( .A1(n283), .A2(n321), .A3(n239), .ZN(n299) );
  VHSR_NOR2_1 U292 ( .A1(n324), .A2(n281), .ZN(n237) );
  VHSR_IAO21_2 U293 ( .A1(n238), .A2(n237), .B(n300), .ZN(n303) );
  VHSR_OAI32_2 U294 ( .A1(n299), .A2(n321), .A3(n283), .B1(n239), .B2(n299), 
        .ZN(n302) );
  VHSR_OAI21_2 U295 ( .A1(n242), .A2(n241), .B(n240), .ZN(n243) );
  VHSR_XNOR2_2 U296 ( .A1(n244), .A2(n243), .ZN(n310) );
  VHSR_AOI21_2 U297 ( .A1(n247), .A2(n246), .B(n245), .ZN(n248) );
  VHSR_XNOR2_2 U298 ( .A1(n249), .A2(n248), .ZN(n309) );
  VHSR_NOR2_1 U299 ( .A1(n261), .A2(n250), .ZN(n252) );
  VHSR_AOI22_2 U300 ( .A1(n261), .A2(n250), .B1(n253), .B2(n252), .ZN(n251) );
  VHSR_OAI21_2 U301 ( .A1(n253), .A2(n252), .B(n251), .ZN(n315) );
  VHSR_OAI21_2 U302 ( .A1(n256), .A2(n255), .B(n254), .ZN(n257) );
  VHSR_IN_2 U303 ( .I(n257), .ZN(n314) );
  VHSR_OAI21_2 U304 ( .A1(n267), .A2(n259), .B(n258), .ZN(n260) );
  VHSR_IN_2 U305 ( .I(n260), .ZN(n334) );
  VHSR_IAO21_2 U306 ( .A1(n263), .A2(n262), .B(n261), .ZN(n333) );
  VHSR_AOI31_2 U307 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n341) );
  VHSR_AOI31_2 U308 ( .A1(n270), .A2(n269), .A3(n268), .B(n267), .ZN(n340) );
  VHSR_CLKNAND2_2 U309 ( .A1(b[5]), .A2(a[0]), .ZN(n271) );
  VHSR_OAI32_2 U310 ( .A1(n272), .A2(n323), .A3(n279), .B1(n271), .B2(n272), 
        .ZN(n345) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[5]), .A2(b[0]), .ZN(n273) );
  VHSR_OAI32_2 U312 ( .A1(n274), .A2(n322), .A3(n285), .B1(n273), .B2(n274), 
        .ZN(n344) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[6]), .A2(b[6]), .ZN(n365) );
  VHSR_IN_2 U314 ( .I(n365), .ZN(n392) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[6]), .A2(b[4]), .ZN(n307) );
  VHSR_NAND3_2 U316 ( .A1(a[7]), .A2(b[5]), .A3(n307), .ZN(n276) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[4]), .A2(b[6]), .ZN(n306) );
  VHSR_NAND3_2 U318 ( .A1(b[7]), .A2(a[5]), .A3(n306), .ZN(n275) );
  VHSR_CLKNAND2_2 U319 ( .A1(n276), .A2(n275), .ZN(n278) );
  VHSR_MAOI222_2 U320 ( .A(n365), .B(n276), .C(n275), .ZN(n349) );
  VHSR_IN_2 U321 ( .I(n349), .ZN(n277) );
  VHSR_OAI21_2 U322 ( .A1(n392), .A2(n278), .B(n277), .ZN(n294) );
  VHSR_AND3_2 U323 ( .A1(n377), .A2(a[5]), .A3(b[5]), .Z(n311) );
  VHSR_NOR3_2 U324 ( .A1(n281), .A2(n307), .A3(n280), .ZN(n357) );
  VHSR_AOI22_2 U325 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n282) );
  VHSR_NOR2_1 U326 ( .A1(n357), .A2(n282), .ZN(n290) );
  VHSR_NOR4_2 U327 ( .A1(n286), .A2(n285), .A3(n284), .A4(n283), .ZN(n355) );
  VHSR_AOI22_2 U328 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n287) );
  VHSR_NOR2_1 U329 ( .A1(n355), .A2(n287), .ZN(n289) );
  VHSR_IN_2 U330 ( .I(n288), .ZN(n297) );
  VHSR_MAOI222_2 U331 ( .A(n307), .B(n306), .C(n305), .ZN(n304) );
  VHSR_IN_2 U332 ( .I(n304), .ZN(n296) );
  VHSR_AD1_1 U333 ( .A(n311), .B(n290), .CI(n289), .CO(n291), .S(n288) );
  VHSR_NOR2_1 U334 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U335 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_NOR2_1 U336 ( .A1(n293), .A2(n294), .ZN(n350) );
  VHSR_AOI22_2 U337 ( .A1(n294), .A2(n293), .B1(n292), .B2(n350), .ZN(n390) );
  VHSR_AOI21_2 U338 ( .A1(n297), .A2(n296), .B(n295), .ZN(n369) );
  VHSR_AD1_1 U339 ( .A(n300), .B(n299), .CI(n298), .CO(n391), .S(n368) );
  VHSR_AD1_1 U340 ( .A(n303), .B(n302), .CI(n301), .CO(n298), .S(n372) );
  VHSR_AOI31_2 U341 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n371) );
  VHSR_AD1_1 U342 ( .A(n310), .B(n309), .CI(n308), .CO(n301), .S(n375) );
  VHSR_AOI22_2 U343 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n312) );
  VHSR_NOR2_1 U344 ( .A1(n312), .A2(n311), .ZN(n374) );
  VHSR_AD1_1 U345 ( .A(n315), .B(n314), .CI(n313), .CO(n308), .S(n378) );
  VHSR_NOR4_2 U346 ( .A1(n324), .A2(n316), .A3(n323), .A4(n318), .ZN(n348) );
  VHSR_CLKNAND2_2 U347 ( .A1(b[2]), .A2(a[1]), .ZN(n317) );
  VHSR_OAI32_2 U348 ( .A1(n348), .A2(n318), .A3(n324), .B1(n317), .B2(n348), 
        .ZN(n382) );
  VHSR_AOI22_2 U349 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n406) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[2]), .A2(a[0]), .ZN(n405) );
  VHSR_NOR2_1 U351 ( .A1(n406), .A2(n405), .ZN(n404) );
  VHSR_OAI22_2 U352 ( .A1(n322), .A2(n320), .B1(n319), .B2(n321), .ZN(n381) );
  VHSR_IN_2 U353 ( .I(n343), .ZN(n329) );
  VHSR_NOR2_1 U354 ( .A1(n322), .A2(n321), .ZN(n325) );
  VHSR_AOI211_2 U355 ( .A1(b[2]), .A2(a[0]), .B(n324), .C(n323), .ZN(n326) );
  VHSR_MAOI222_2 U356 ( .A(n325), .B(n326), .C(n338), .ZN(n328) );
  VHSR_IN_2 U357 ( .I(n338), .ZN(n331) );
  VHSR_AOI32_2 U358 ( .A1(a[3]), .A2(n328), .A3(b[1]), .B1(n327), .B2(n328), 
        .ZN(n342) );
  VHSR_OAI21_2 U359 ( .A1(n329), .A2(n342), .B(n328), .ZN(n347) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[3]), .A2(a[3]), .ZN(n336) );
  VHSR_AOI22_2 U361 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n330) );
  VHSR_IAO21_2 U362 ( .A1(n336), .A2(n331), .B(n330), .ZN(n346) );
  VHSR_IAO21_2 U363 ( .A1(n338), .A2(n337), .B(n336), .ZN(n388) );
  VHSR_AD1_1 U364 ( .A(n334), .B(n333), .CI(n332), .CO(n313), .S(n387) );
  VHSR_OAI21_2 U365 ( .A1(n338), .A2(n336), .B(n337), .ZN(n335) );
  VHSR_OAI31_2 U366 ( .A1(n338), .A2(n337), .A3(n336), .B(n335), .ZN(n385) );
  VHSR_AD1_1 U367 ( .A(n341), .B(n340), .CI(n339), .CO(n332), .S(n384) );
  VHSR_CLKXOR2_2 U368 ( .A1(n343), .A2(n342), .Z(n403) );
  VHSR_AOI211_2 U369 ( .A1(n399), .A2(n398), .B(n397), .C(n403), .ZN(n401) );
  VHSR_AD1_1 U370 ( .A(n345), .B(n397), .CI(n344), .CO(n339), .S(n380) );
  VHSR_AD1_1 U371 ( .A(n348), .B(n347), .CI(n346), .CO(n337), .S(n379) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[6]), .ZN(n352) );
  VHSR_AOI21_2 U373 ( .A1(a[6]), .A2(b[7]), .B(n352), .ZN(n351) );
  VHSR_AOI31_2 U374 ( .A1(a[6]), .A2(n352), .A3(b[7]), .B(n351), .ZN(n353) );
  VHSR_IN_2 U375 ( .I(n353), .ZN(n354) );
  VHSR_OR2_2 U376 ( .A1(n355), .A2(n354), .Z(n356) );
  VHSR_MAOI222_2 U377 ( .A(n357), .B(n355), .C(n354), .ZN(n364) );
  VHSR_OAI21_2 U378 ( .A1(n357), .A2(n356), .B(n364), .ZN(n361) );
  VHSR_CLKXOR2_2 U379 ( .A1(n362), .A2(n361), .Z(n358) );
  VHSR_CLKNAND2_2 U380 ( .A1(n359), .A2(n358), .ZN(n394) );
  VHSR_OAI21_2 U381 ( .A1(n359), .A2(n358), .B(n394), .ZN(n360) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[7]), .A2(b[7]), .ZN(n393) );
  VHSR_NOR2_1 U383 ( .A1(n362), .A2(n361), .ZN(n363) );
  VHSR_AND3_2 U384 ( .A1(n395), .A2(n365), .A3(n394), .Z(n366) );
  VHSR_NOR2_1 U385 ( .A1(n393), .A2(n366), .ZN(product[15]) );
  VHSR_AD1_1 U386 ( .A(n385), .B(n384), .CI(n383), .CO(n386), .S(product[6])
         );
  VHSR_AD1_1 U387 ( .A(n388), .B(n387), .CI(n386), .CO(n376), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U388 ( .A(n391), .B(n390), .CI(n389), .CO(n359), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U389 ( .A1(n393), .A2(n392), .ZN(n396) );
  VHSR_XOR3_2 U390 ( .A1(n396), .A2(n395), .A3(n394), .Z(product[14]) );
  VHSR_AOI21_2 U391 ( .A1(n399), .A2(n398), .B(n397), .ZN(n400) );
  VHSR_IN_2 U392 ( .I(n400), .ZN(n402) );
  VHSR_AOI21_2 U393 ( .A1(n403), .A2(n402), .B(n401), .ZN(product[4]) );
  VHSR_AOI21_2 U394 ( .A1(n406), .A2(n405), .B(n404), .ZN(product[2]) );
endmodule


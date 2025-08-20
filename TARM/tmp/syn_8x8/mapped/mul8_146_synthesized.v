
module mul8_146 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n228, n229,
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
         n395, n396, n397, n398, n399, n400, n401, n402, n403, n404, n405,
         n406, n407, n408, n409, n410, n411, n412, n413, n414, n415, n416,
         n417, n418, n419, n420, n421, n422, n423, n424, n425, n426, n427,
         n428, n429;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U219 ( .A1(n254), .B1(n238), .ZN(n240) );
  VHSR_NOR2_1 U220 ( .A1(n334), .A2(n324), .ZN(n276) );
  VHSR_INAND2_2 U221 ( .A1(n374), .B1(n372), .ZN(n375) );
  VHSR_INOR2_2 U222 ( .A1(n242), .B1(n265), .ZN(n264) );
  VHSR_INOR2_2 U223 ( .A1(n369), .B1(n368), .ZN(n381) );
  VHSR_INOR2_2 U224 ( .A1(n363), .B1(n339), .ZN(n362) );
  VHSR_INOR2_2 U225 ( .A1(n346), .B1(n360), .ZN(n353) );
  VHSR_NOR2_1 U226 ( .A1(n247), .A2(n246), .ZN(n307) );
  VHSR_INOR2_2 U227 ( .A1(n383), .B1(n382), .ZN(n414) );
  VHSR_IN_2 U228 ( .I(n379), .ZN(product[13]) );
  VHSR_CLKN_1 U229 ( .I(n384), .ZN(n385) );
  VHSR_INAND3_1 U230 ( .A1(n411), .B1(n414), .B2(n413), .ZN(n384) );
  VHSR_INOR2_1 U231 ( .A1(n244), .B1(n262), .ZN(n255) );
  VHSR_INOR3_1 U232 ( .A1(n296), .B1(n287), .B2(n322), .ZN(n376) );
  VHSR_AD1_1 U233 ( .A(n402), .B(n420), .CI(n401), .CO(n398), .S(product[5])
         );
  VHSR_AD1_1 U234 ( .A(n397), .B(n396), .CI(n395), .CO(n392), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U235 ( .A(n391), .B(n390), .CI(n389), .CO(n386), .S(product[10])
         );
  VHSR_AD1_1 U236 ( .A(n404), .B(n403), .CI(n427), .CO(n365), .S(product[3])
         );
  VHSR_AD1_1 U237 ( .A(n400), .B(n399), .CI(n398), .CO(n405), .S(product[6])
         );
  VHSR_AD1_1 U238 ( .A(n394), .B(n393), .CI(n392), .CO(n389), .S(product[9])
         );
  VHSR_AD1_1 U239 ( .A(n388), .B(n387), .CI(n386), .CO(n408), .S(product[11])
         );
  VHSR_CLKNAND2_2 U240 ( .A1(b[0]), .A2(a[0]), .ZN(n333) );
  VHSR_IN_2 U241 ( .I(n333), .ZN(product[0]) );
  VHSR_IN_2 U242 ( .I(b[7]), .ZN(n289) );
  VHSR_IN_2 U243 ( .I(a[3]), .ZN(n354) );
  VHSR_IN_2 U244 ( .I(b[6]), .ZN(n290) );
  VHSR_IN_2 U245 ( .I(a[2]), .ZN(n335) );
  VHSR_OAI22_2 U246 ( .A1(n290), .A2(n354), .B1(n289), .B2(n335), .ZN(n252) );
  VHSR_AOI22_2 U247 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n234) );
  VHSR_CLKNAND2_2 U248 ( .A1(b[4]), .A2(a[2]), .ZN(n272) );
  VHSR_NAND3_2 U249 ( .A1(a[3]), .A2(b[5]), .A3(n272), .ZN(n233) );
  VHSR_CLKNAND2_2 U250 ( .A1(b[7]), .A2(a[2]), .ZN(n228) );
  VHSR_CLKNAND2_2 U251 ( .A1(b[6]), .A2(a[1]), .ZN(n230) );
  VHSR_OAI22_2 U252 ( .A1(n234), .A2(n233), .B1(n228), .B2(n230), .ZN(n235) );
  VHSR_CLKNAND2_2 U253 ( .A1(b[4]), .A2(a[0]), .ZN(n417) );
  VHSR_NAND3_2 U254 ( .A1(a[1]), .A2(b[5]), .A3(n417), .ZN(n271) );
  VHSR_CLKNAND2_2 U255 ( .A1(b[6]), .A2(a[0]), .ZN(n270) );
  VHSR_MAOI222_2 U256 ( .A(n272), .B(n271), .C(n270), .ZN(n269) );
  VHSR_NAND4_2 U257 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n249) );
  VHSR_IN_2 U258 ( .I(b[4]), .ZN(n319) );
  VHSR_IN_2 U259 ( .I(b[5]), .ZN(n322) );
  VHSR_OAI22_2 U260 ( .A1(n319), .A2(n354), .B1(n322), .B2(n335), .ZN(n229) );
  VHSR_AND2_2 U261 ( .A1(n249), .A2(n229), .Z(n232) );
  VHSR_IN_2 U262 ( .I(a[0]), .ZN(n425) );
  VHSR_OAI21_2 U263 ( .A1(n289), .A2(n425), .B(n230), .ZN(n231) );
  VHSR_IN_2 U264 ( .I(a[1]), .ZN(n423) );
  VHSR_NOR3_2 U265 ( .A1(n322), .A2(n423), .A3(n417), .ZN(n279) );
  VHSR_AND2_2 U266 ( .A1(n269), .A2(n268), .Z(n267) );
  VHSR_AD1_1 U267 ( .A(n232), .B(n231), .CI(n279), .CO(n258), .S(n268) );
  VHSR_AOI21_2 U268 ( .A1(n234), .A2(n233), .B(n235), .ZN(n261) );
  VHSR_OAI32_2 U269 ( .A1(n235), .A2(n267), .A3(n258), .B1(n261), .B2(n235), 
        .ZN(n250) );
  VHSR_CLKNAND2_2 U270 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_CLKNAND2_2 U271 ( .A1(n252), .A2(n248), .ZN(n245) );
  VHSR_NOR3_2 U272 ( .A1(n289), .A2(n354), .A3(n245), .ZN(n308) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[3]), .A2(a[7]), .ZN(n247) );
  VHSR_IN_2 U274 ( .I(b[3]), .ZN(n355) );
  VHSR_IN_2 U275 ( .I(a[6]), .ZN(n282) );
  VHSR_IN_2 U276 ( .I(a[7]), .ZN(n287) );
  VHSR_IN_2 U277 ( .I(b[2]), .ZN(n334) );
  VHSR_OAI22_2 U278 ( .A1(n355), .A2(n282), .B1(n287), .B2(n334), .ZN(n257) );
  VHSR_IN_2 U279 ( .I(a[4]), .ZN(n324) );
  VHSR_CLKNAND2_2 U280 ( .A1(b[3]), .A2(a[5]), .ZN(n236) );
  VHSR_IN_2 U281 ( .I(b[1]), .ZN(n426) );
  VHSR_OAI22_2 U282 ( .A1(n276), .A2(n236), .B1(n287), .B2(n426), .ZN(n243) );
  VHSR_IN_2 U283 ( .I(a[5]), .ZN(n320) );
  VHSR_NOR4_2 U284 ( .A1(n276), .A2(n247), .A3(n320), .A4(n426), .ZN(n237) );
  VHSR_AOI31_2 U285 ( .A1(b[2]), .A2(a[6]), .A3(n243), .B(n237), .ZN(n244) );
  VHSR_NOR2_1 U286 ( .A1(n282), .A2(n426), .ZN(n239) );
  VHSR_IN_2 U287 ( .I(b[0]), .ZN(n424) );
  VHSR_NOR4_2 U288 ( .A1(n324), .A2(n320), .A3(n426), .A4(n424), .ZN(n281) );
  VHSR_NAND3_2 U289 ( .A1(b[3]), .A2(n276), .A3(a[5]), .ZN(n254) );
  VHSR_AOI22_2 U290 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n238) );
  VHSR_MAOI222_2 U291 ( .A(n239), .B(n281), .C(n240), .ZN(n242) );
  VHSR_AOI211_2 U292 ( .A1(a[4]), .A2(b[0]), .B(n320), .C(n426), .ZN(n275) );
  VHSR_AOI21_2 U293 ( .A1(n287), .A2(n282), .B(n424), .ZN(n274) );
  VHSR_MAOI222_2 U294 ( .A(n276), .B(n275), .C(n274), .ZN(n273) );
  VHSR_OR2_2 U295 ( .A1(n281), .A2(n240), .Z(n241) );
  VHSR_AOI32_2 U296 ( .A1(b[1]), .A2(n242), .A3(a[6]), .B1(n241), .B2(n242), 
        .ZN(n266) );
  VHSR_NOR2_1 U297 ( .A1(n273), .A2(n266), .ZN(n265) );
  VHSR_AOI32_2 U298 ( .A1(b[2]), .A2(n244), .A3(a[6]), .B1(n243), .B2(n244), 
        .ZN(n263) );
  VHSR_NOR2_1 U299 ( .A1(n264), .A2(n263), .ZN(n262) );
  VHSR_CLKNAND2_2 U300 ( .A1(n255), .A2(n254), .ZN(n253) );
  VHSR_CLKNAND2_2 U301 ( .A1(n257), .A2(n253), .ZN(n246) );
  VHSR_OAI32_2 U302 ( .A1(n308), .A2(n354), .A3(n289), .B1(n245), .B2(n308), 
        .ZN(n315) );
  VHSR_AOI21_2 U303 ( .A1(n247), .A2(n246), .B(n307), .ZN(n314) );
  VHSR_OAI21_2 U304 ( .A1(n250), .A2(n249), .B(n248), .ZN(n251) );
  VHSR_XNOR2_2 U305 ( .A1(n252), .A2(n251), .ZN(n318) );
  VHSR_OAI21_2 U306 ( .A1(n255), .A2(n254), .B(n253), .ZN(n256) );
  VHSR_XNOR2_2 U307 ( .A1(n257), .A2(n256), .ZN(n317) );
  VHSR_NOR2_1 U308 ( .A1(n267), .A2(n258), .ZN(n260) );
  VHSR_AOI22_2 U309 ( .A1(n267), .A2(n258), .B1(n261), .B2(n260), .ZN(n259) );
  VHSR_OAI21_2 U310 ( .A1(n261), .A2(n260), .B(n259), .ZN(n327) );
  VHSR_AOI21_2 U311 ( .A1(n264), .A2(n263), .B(n262), .ZN(n326) );
  VHSR_AOI21_2 U312 ( .A1(n273), .A2(n266), .B(n265), .ZN(n349) );
  VHSR_IAO21_2 U313 ( .A1(n269), .A2(n268), .B(n267), .ZN(n348) );
  VHSR_AOI31_2 U314 ( .A1(n272), .A2(n271), .A3(n270), .B(n269), .ZN(n352) );
  VHSR_OAI31_2 U315 ( .A1(n276), .A2(n275), .A3(n274), .B(n273), .ZN(n277) );
  VHSR_IN_2 U316 ( .I(n277), .ZN(n351) );
  VHSR_AOI22_2 U317 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n278) );
  VHSR_NOR2_1 U318 ( .A1(n279), .A2(n278), .ZN(n367) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[4]), .A2(b[4]), .ZN(n293) );
  VHSR_NOR2_1 U320 ( .A1(n293), .A2(n333), .ZN(n416) );
  VHSR_CLKNAND2_2 U321 ( .A1(a[5]), .A2(b[0]), .ZN(n280) );
  VHSR_OAI32_2 U322 ( .A1(n281), .A2(n426), .A3(n324), .B1(n280), .B2(n281), 
        .ZN(n366) );
  VHSR_NOR2_1 U323 ( .A1(n282), .A2(n290), .ZN(n411) );
  VHSR_NOR2_1 U324 ( .A1(n324), .A2(n290), .ZN(n295) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[5]), .A2(b[7]), .ZN(n284) );
  VHSR_NOR2_1 U326 ( .A1(n282), .A2(n319), .ZN(n296) );
  VHSR_CLKNAND2_2 U327 ( .A1(a[7]), .A2(b[5]), .ZN(n283) );
  VHSR_OAI22_2 U328 ( .A1(n295), .A2(n284), .B1(n296), .B2(n283), .ZN(n286) );
  VHSR_OR2_2 U329 ( .A1(n295), .A2(n296), .Z(n310) );
  VHSR_CLKNAND2_2 U330 ( .A1(a[5]), .A2(b[5]), .ZN(n294) );
  VHSR_CLKNAND2_2 U331 ( .A1(a[7]), .A2(b[7]), .ZN(n412) );
  VHSR_NOR3_2 U332 ( .A1(n310), .A2(n294), .A3(n412), .ZN(n285) );
  VHSR_AOI31_2 U333 ( .A1(b[6]), .A2(a[6]), .A3(n286), .B(n285), .ZN(n369) );
  VHSR_OAI21_2 U334 ( .A1(n411), .A2(n286), .B(n369), .ZN(n303) );
  VHSR_AOI22_2 U335 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n288) );
  VHSR_NOR2_1 U336 ( .A1(n376), .A2(n288), .ZN(n299) );
  VHSR_NOR2_1 U337 ( .A1(n294), .A2(n293), .ZN(n298) );
  VHSR_NOR4_2 U338 ( .A1(n324), .A2(n320), .A3(n290), .A4(n289), .ZN(n374) );
  VHSR_AOI22_2 U339 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n291) );
  VHSR_NOR2_1 U340 ( .A1(n374), .A2(n291), .ZN(n297) );
  VHSR_IN_2 U341 ( .I(n292), .ZN(n305) );
  VHSR_IN_2 U342 ( .I(n293), .ZN(n396) );
  VHSR_NOR2_1 U343 ( .A1(n396), .A2(n294), .ZN(n311) );
  VHSR_AOI22_2 U344 ( .A1(n296), .A2(n295), .B1(n311), .B2(n310), .ZN(n309) );
  VHSR_NOR2_1 U345 ( .A1(n305), .A2(n309), .ZN(n304) );
  VHSR_AD1_1 U346 ( .A(n299), .B(n298), .CI(n297), .CO(n300), .S(n292) );
  VHSR_NOR2_1 U347 ( .A1(n304), .A2(n300), .ZN(n302) );
  VHSR_CLKNAND2_2 U348 ( .A1(n304), .A2(n300), .ZN(n301) );
  VHSR_NOR2_1 U349 ( .A1(n302), .A2(n303), .ZN(n368) );
  VHSR_AOI22_2 U350 ( .A1(n303), .A2(n302), .B1(n301), .B2(n368), .ZN(n409) );
  VHSR_AOI21_2 U351 ( .A1(n309), .A2(n305), .B(n304), .ZN(n388) );
  VHSR_AD1_1 U352 ( .A(n308), .B(n307), .CI(n306), .CO(n410), .S(n387) );
  VHSR_OAI21_2 U353 ( .A1(n311), .A2(n310), .B(n309), .ZN(n312) );
  VHSR_IN_2 U354 ( .I(n312), .ZN(n391) );
  VHSR_AD1_1 U355 ( .A(n315), .B(n314), .CI(n313), .CO(n306), .S(n390) );
  VHSR_AD1_1 U356 ( .A(n318), .B(n317), .CI(n316), .CO(n313), .S(n394) );
  VHSR_NOR2_1 U357 ( .A1(n320), .A2(n319), .ZN(n323) );
  VHSR_OAI21_2 U358 ( .A1(n324), .A2(n322), .B(n323), .ZN(n321) );
  VHSR_OAI31_2 U359 ( .A1(n324), .A2(n323), .A3(n322), .B(n321), .ZN(n393) );
  VHSR_AD1_1 U360 ( .A(n327), .B(n326), .CI(n325), .CO(n316), .S(n397) );
  VHSR_NOR4_2 U361 ( .A1(n355), .A2(n334), .A3(n423), .A4(n425), .ZN(n340) );
  VHSR_NOR4_2 U362 ( .A1(n426), .A2(n424), .A3(n335), .A4(n354), .ZN(n342) );
  VHSR_CLKNAND2_2 U363 ( .A1(a[3]), .A2(b[2]), .ZN(n329) );
  VHSR_OAI21_2 U364 ( .A1(n335), .A2(n355), .B(n329), .ZN(n328) );
  VHSR_OAI31_2 U365 ( .A1(n335), .A2(n329), .A3(n355), .B(n328), .ZN(n330) );
  VHSR_IN_2 U366 ( .I(n330), .ZN(n341) );
  VHSR_MAOI222_2 U367 ( .A(n340), .B(n342), .C(n341), .ZN(n346) );
  VHSR_CLKNAND2_2 U368 ( .A1(b[2]), .A2(a[1]), .ZN(n331) );
  VHSR_OAI32_2 U369 ( .A1(n340), .A2(n425), .A3(n355), .B1(n331), .B2(n340), 
        .ZN(n404) );
  VHSR_CLKNAND2_2 U370 ( .A1(b[0]), .A2(a[3]), .ZN(n332) );
  VHSR_OAI32_2 U371 ( .A1(n342), .A2(n335), .A3(n426), .B1(n332), .B2(n342), 
        .ZN(n403) );
  VHSR_CLKNAND2_2 U372 ( .A1(b[1]), .A2(a[1]), .ZN(n428) );
  VHSR_AOI22_2 U373 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n429) );
  VHSR_CLKNAND2_2 U374 ( .A1(b[2]), .A2(a[2]), .ZN(n359) );
  VHSR_OAI22_2 U375 ( .A1(n428), .A2(n429), .B1(n333), .B2(n359), .ZN(n427) );
  VHSR_OAI211_2 U376 ( .A1(n334), .A2(n425), .B(b[3]), .C(a[1]), .ZN(n337) );
  VHSR_OAI211_2 U377 ( .A1(n335), .A2(n424), .B(b[1]), .C(a[3]), .ZN(n336) );
  VHSR_AND2_2 U378 ( .A1(n337), .A2(n336), .Z(n338) );
  VHSR_MAOI222_2 U379 ( .A(n359), .B(n337), .C(n336), .ZN(n339) );
  VHSR_AOI21_2 U380 ( .A1(n338), .A2(n359), .B(n339), .ZN(n364) );
  VHSR_CLKNAND2_2 U381 ( .A1(n365), .A2(n364), .ZN(n363) );
  VHSR_IN_2 U382 ( .I(n340), .ZN(n345) );
  VHSR_NOR2_1 U383 ( .A1(n342), .A2(n341), .ZN(n344) );
  VHSR_AOI22_2 U384 ( .A1(n342), .A2(n341), .B1(n345), .B2(n344), .ZN(n343) );
  VHSR_OAI21_2 U385 ( .A1(n345), .A2(n344), .B(n343), .ZN(n361) );
  VHSR_NOR2_1 U386 ( .A1(n362), .A2(n361), .ZN(n360) );
  VHSR_AOI211_2 U387 ( .A1(n353), .A2(n359), .B(n354), .C(n355), .ZN(n407) );
  VHSR_AD1_1 U388 ( .A(n349), .B(n348), .CI(n347), .CO(n325), .S(n406) );
  VHSR_AD1_1 U389 ( .A(n352), .B(n351), .CI(n350), .CO(n347), .S(n400) );
  VHSR_IN_2 U390 ( .I(n353), .ZN(n358) );
  VHSR_NOR2_1 U391 ( .A1(n355), .A2(n354), .ZN(n357) );
  VHSR_AOI21_2 U392 ( .A1(n359), .A2(n357), .B(n358), .ZN(n356) );
  VHSR_AOI31_2 U393 ( .A1(n359), .A2(n358), .A3(n357), .B(n356), .ZN(n399) );
  VHSR_AOI21_2 U394 ( .A1(n362), .A2(n361), .B(n360), .ZN(n402) );
  VHSR_CLKNAND2_2 U395 ( .A1(a[4]), .A2(b[0]), .ZN(n418) );
  VHSR_OAI21_2 U396 ( .A1(n365), .A2(n364), .B(n363), .ZN(n422) );
  VHSR_AOI211_2 U397 ( .A1(n418), .A2(n417), .B(n416), .C(n422), .ZN(n420) );
  VHSR_AD1_1 U398 ( .A(n367), .B(n416), .CI(n366), .CO(n350), .S(n401) );
  VHSR_CLKNAND2_2 U399 ( .A1(a[6]), .A2(b[7]), .ZN(n371) );
  VHSR_AOI21_2 U400 ( .A1(a[7]), .A2(b[6]), .B(n371), .ZN(n370) );
  VHSR_AOI31_2 U401 ( .A1(a[7]), .A2(n371), .A3(b[6]), .B(n370), .ZN(n372) );
  VHSR_IN_2 U402 ( .I(n372), .ZN(n373) );
  VHSR_MAOI222_2 U403 ( .A(n376), .B(n374), .C(n373), .ZN(n383) );
  VHSR_OAI21_2 U404 ( .A1(n376), .A2(n375), .B(n383), .ZN(n380) );
  VHSR_CLKXOR2_2 U405 ( .A1(n381), .A2(n380), .Z(n377) );
  VHSR_CLKNAND2_2 U406 ( .A1(n378), .A2(n377), .ZN(n413) );
  VHSR_OAI21_2 U407 ( .A1(n378), .A2(n377), .B(n413), .ZN(n379) );
  VHSR_NOR2_1 U408 ( .A1(n381), .A2(n380), .ZN(n382) );
  VHSR_NOR2_1 U409 ( .A1(n412), .A2(n385), .ZN(product[15]) );
  VHSR_AD1_1 U410 ( .A(n407), .B(n406), .CI(n405), .CO(n395), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U411 ( .A(n410), .B(n409), .CI(n408), .CO(n378), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U412 ( .A1(n412), .A2(n411), .ZN(n415) );
  VHSR_XOR3_2 U413 ( .A1(n415), .A2(n414), .A3(n413), .Z(product[14]) );
  VHSR_AOI21_2 U414 ( .A1(n418), .A2(n417), .B(n416), .ZN(n419) );
  VHSR_IN_2 U415 ( .I(n419), .ZN(n421) );
  VHSR_AOI21_2 U416 ( .A1(n422), .A2(n421), .B(n420), .ZN(product[4]) );
  VHSR_OAI22_2 U417 ( .A1(n426), .A2(n425), .B1(n424), .B2(n423), .ZN(
        product[1]) );
  VHSR_AOI21_2 U418 ( .A1(n429), .A2(n428), .B(n427), .ZN(product[2]) );
endmodule


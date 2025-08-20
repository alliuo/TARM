
module mul8_112 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n229, n230, n231, n232, n233, n234, n235, n236,
         n237, n238, n239, n240, n241, n242, n243, n244, n245, n246, n247,
         n248, n249, n250, n251, n252, n253, n254, n255, n256, n257, n258,
         n259, n260, n261, n262, n263, n264, n265, n266, n267, n268, n269,
         n270, n271, n272, n273, n274, n275, n276, n277, n278, n279, n280,
         n281, n282, n283, n284, n285, n286, n287, n288, n289, n290, n291,
         n292, n293, n294, n295, n296, n297, n298, n299, n300, n301, n302,
         n303, n304, n305, n306, n307, n308, n309, n310, n311, n312, n313,
         n314, n315, n316, n317, n318, n319, n320, n321, n322, n323, n324,
         n325, n326, n327, n328, n329, n330, n331, n332, n333, n334, n335,
         n336, n337, n338, n339, n340, n341, n342, n343, n344, n345, n346,
         n347, n348, n349, n350, n351, n352, n353, n354, n355, n356, n357,
         n358, n359, n360, n361, n362, n363, n364, n365, n366, n367, n368,
         n369, n370, n371, n372, n373, n374, n375, n376, n377, n378, n379,
         n380, n381, n382, n383, n384, n385, n386, n387, n388, n389, n390,
         n391, n392, n393, n394, n395, n396, n397, n398, n399, n400, n401,
         n402, n403, n404, n405, n406, n407, n408, n409, n410, n411, n412,
         n413, n414, n415, n416, n417, n418, n419, n420, n421, n422, n423,
         n424, n425, n426, n427, n428;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U218 ( .A1(n235), .B1(n234), .ZN(n236) );
  VHSR_NOR2_1 U219 ( .A1(n302), .A2(n336), .ZN(n259) );
  VHSR_NOR2_1 U220 ( .A1(n303), .A2(n361), .ZN(n398) );
  VHSR_IN_2 U221 ( .I(n381), .ZN(product[13]) );
  VHSR_INOR2_1 U222 ( .A1(n385), .B1(n384), .ZN(n416) );
  VHSR_AND4_1 U223 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .Z(n262) );
  VHSR_AD1_1 U224 ( .A(n399), .B(n398), .CI(n397), .CO(n394), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n393), .B(n392), .CI(n391), .CO(n388), .S(product[10])
         );
  VHSR_AD1_1 U226 ( .A(n406), .B(n405), .CI(n425), .CO(n360), .S(product[3])
         );
  VHSR_AD1_1 U227 ( .A(n418), .B(n404), .CI(n403), .CO(n407), .S(product[5])
         );
  VHSR_AD1_1 U228 ( .A(n402), .B(n401), .CI(n400), .CO(n397), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U229 ( .A(n396), .B(n395), .CI(n394), .CO(n391), .S(product[9])
         );
  VHSR_AD1_1 U230 ( .A(n390), .B(n389), .CI(n388), .CO(n410), .S(
        \intadd_0/SUM[6] ) );
  VHSR_IN_2 U231 ( .I(b[6]), .ZN(n302) );
  VHSR_IN_2 U232 ( .I(a[2]), .ZN(n336) );
  VHSR_IN_2 U233 ( .I(b[5]), .ZN(n300) );
  VHSR_CLKNAND2_2 U234 ( .A1(b[4]), .A2(a[2]), .ZN(n285) );
  VHSR_IN_2 U235 ( .I(a[3]), .ZN(n338) );
  VHSR_NOR3_2 U236 ( .A1(n300), .A2(n285), .A3(n338), .ZN(n267) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[7]), .A2(a[3]), .ZN(n257) );
  VHSR_IN_2 U238 ( .I(n259), .ZN(n234) );
  VHSR_AOI22_2 U239 ( .A1(b[6]), .A2(a[3]), .B1(b[7]), .B2(a[2]), .ZN(n229) );
  VHSR_IAO21_2 U240 ( .A1(n257), .A2(n234), .B(n229), .ZN(n266) );
  VHSR_AOI22_2 U241 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n230) );
  VHSR_NOR2_1 U242 ( .A1(n230), .A2(n267), .ZN(n233) );
  VHSR_IN_2 U243 ( .I(a[1]), .ZN(n421) );
  VHSR_IN_2 U244 ( .I(b[7]), .ZN(n301) );
  VHSR_IN_2 U245 ( .I(a[0]), .ZN(n423) );
  VHSR_OAI22_2 U246 ( .A1(n302), .A2(n421), .B1(n301), .B2(n423), .ZN(n232) );
  VHSR_IN_2 U247 ( .I(b[4]), .ZN(n361) );
  VHSR_NOR4_2 U248 ( .A1(n361), .A2(n300), .A3(n421), .A4(n423), .ZN(n291) );
  VHSR_IN_2 U249 ( .I(n269), .ZN(n238) );
  VHSR_NOR2_1 U250 ( .A1(n301), .A2(n421), .ZN(n231) );
  VHSR_AOI211_2 U251 ( .A1(b[4]), .A2(a[2]), .B(n300), .C(n338), .ZN(n235) );
  VHSR_MAOI222_2 U252 ( .A(n231), .B(n235), .C(n259), .ZN(n237) );
  VHSR_OAI211_2 U253 ( .A1(n361), .A2(n423), .B(b[5]), .C(a[1]), .ZN(n284) );
  VHSR_CLKNAND2_2 U254 ( .A1(b[6]), .A2(a[0]), .ZN(n283) );
  VHSR_MAOI222_2 U255 ( .A(n285), .B(n284), .C(n283), .ZN(n282) );
  VHSR_AD1_1 U256 ( .A(n233), .B(n232), .CI(n291), .CO(n269), .S(n280) );
  VHSR_CLKNAND2_2 U257 ( .A1(n282), .A2(n280), .ZN(n279) );
  VHSR_AOI32_2 U258 ( .A1(a[1]), .A2(n237), .A3(b[7]), .B1(n236), .B2(n237), 
        .ZN(n268) );
  VHSR_AOI32_2 U259 ( .A1(n238), .A2(n237), .A3(n279), .B1(n268), .B2(n237), 
        .ZN(n265) );
  VHSR_IAO21_2 U260 ( .A1(n259), .A2(n258), .B(n257), .ZN(n315) );
  VHSR_AOI22_2 U261 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n264) );
  VHSR_IN_2 U262 ( .I(a[7]), .ZN(n298) );
  VHSR_IN_2 U263 ( .I(b[1]), .ZN(n424) );
  VHSR_NOR2_1 U264 ( .A1(n298), .A2(n424), .ZN(n240) );
  VHSR_IN_2 U265 ( .I(b[3]), .ZN(n348) );
  VHSR_IN_2 U266 ( .I(a[5]), .ZN(n304) );
  VHSR_AOI211_2 U267 ( .A1(b[2]), .A2(a[4]), .B(n348), .C(n304), .ZN(n241) );
  VHSR_CLKNAND2_2 U268 ( .A1(a[6]), .A2(b[2]), .ZN(n243) );
  VHSR_IN_2 U269 ( .I(n243), .ZN(n239) );
  VHSR_MAOI222_2 U270 ( .A(n240), .B(n241), .C(n239), .ZN(n253) );
  VHSR_AOI21_2 U271 ( .A1(b[1]), .A2(a[7]), .B(n241), .ZN(n244) );
  VHSR_IN_2 U272 ( .I(n253), .ZN(n242) );
  VHSR_AOI21_2 U273 ( .A1(n244), .A2(n243), .B(n242), .ZN(n274) );
  VHSR_CLKNAND2_2 U274 ( .A1(a[6]), .A2(b[1]), .ZN(n250) );
  VHSR_IN_2 U275 ( .I(n250), .ZN(n247) );
  VHSR_IN_2 U276 ( .I(a[4]), .ZN(n303) );
  VHSR_IN_2 U277 ( .I(b[0]), .ZN(n422) );
  VHSR_NOR4_2 U278 ( .A1(n304), .A2(n303), .A3(n424), .A4(n422), .ZN(n293) );
  VHSR_CLKNAND2_2 U279 ( .A1(b[2]), .A2(a[5]), .ZN(n246) );
  VHSR_CLKNAND2_2 U280 ( .A1(b[3]), .A2(a[4]), .ZN(n245) );
  VHSR_AOI21_2 U281 ( .A1(n246), .A2(n245), .B(n262), .ZN(n248) );
  VHSR_MAOI222_2 U282 ( .A(n247), .B(n293), .C(n248), .ZN(n252) );
  VHSR_CLKNAND2_2 U283 ( .A1(b[2]), .A2(a[4]), .ZN(n289) );
  VHSR_OAI21_2 U284 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n288) );
  VHSR_OAI211_2 U285 ( .A1(n303), .A2(n422), .B(a[5]), .C(b[1]), .ZN(n287) );
  VHSR_MAOI222_2 U286 ( .A(n289), .B(n288), .C(n287), .ZN(n286) );
  VHSR_NOR2_1 U287 ( .A1(n293), .A2(n248), .ZN(n251) );
  VHSR_IN_2 U288 ( .I(n252), .ZN(n249) );
  VHSR_AOI21_2 U289 ( .A1(n251), .A2(n250), .B(n249), .ZN(n277) );
  VHSR_CLKNAND2_2 U290 ( .A1(n286), .A2(n277), .ZN(n276) );
  VHSR_CLKNAND2_2 U291 ( .A1(n252), .A2(n276), .ZN(n273) );
  VHSR_CLKNAND2_2 U292 ( .A1(n274), .A2(n273), .ZN(n272) );
  VHSR_CLKNAND2_2 U293 ( .A1(n253), .A2(n272), .ZN(n261) );
  VHSR_NOR2_1 U294 ( .A1(n262), .A2(n261), .ZN(n260) );
  VHSR_NOR2_1 U295 ( .A1(n264), .A2(n260), .ZN(n255) );
  VHSR_AND3_2 U296 ( .A1(n255), .A2(b[3]), .A3(a[7]), .Z(n314) );
  VHSR_NOR2_1 U297 ( .A1(n348), .A2(n298), .ZN(n254) );
  VHSR_IAO21_2 U298 ( .A1(n255), .A2(n254), .B(n314), .ZN(n320) );
  VHSR_OAI21_2 U299 ( .A1(n259), .A2(n257), .B(n258), .ZN(n256) );
  VHSR_OAI31_2 U300 ( .A1(n259), .A2(n258), .A3(n257), .B(n256), .ZN(n319) );
  VHSR_AOI21_2 U301 ( .A1(n262), .A2(n261), .B(n260), .ZN(n263) );
  VHSR_XNOR2_2 U302 ( .A1(n264), .A2(n263), .ZN(n327) );
  VHSR_AD1_1 U303 ( .A(n267), .B(n266), .CI(n265), .CO(n258), .S(n326) );
  VHSR_NOR2_1 U304 ( .A1(n269), .A2(n268), .ZN(n271) );
  VHSR_AOI22_2 U305 ( .A1(n269), .A2(n268), .B1(n279), .B2(n271), .ZN(n270) );
  VHSR_OAI21_2 U306 ( .A1(n279), .A2(n271), .B(n270), .ZN(n332) );
  VHSR_OAI21_2 U307 ( .A1(n274), .A2(n273), .B(n272), .ZN(n275) );
  VHSR_IN_2 U308 ( .I(n275), .ZN(n331) );
  VHSR_OAI21_2 U309 ( .A1(n286), .A2(n277), .B(n276), .ZN(n278) );
  VHSR_IN_2 U310 ( .I(n278), .ZN(n351) );
  VHSR_OAI21_2 U311 ( .A1(n282), .A2(n280), .B(n279), .ZN(n281) );
  VHSR_IN_2 U312 ( .I(n281), .ZN(n350) );
  VHSR_AOI31_2 U313 ( .A1(n285), .A2(n284), .A3(n283), .B(n282), .ZN(n358) );
  VHSR_AOI31_2 U314 ( .A1(n289), .A2(n288), .A3(n287), .B(n286), .ZN(n357) );
  VHSR_CLKNAND2_2 U315 ( .A1(b[5]), .A2(a[0]), .ZN(n290) );
  VHSR_OAI32_2 U316 ( .A1(n291), .A2(n421), .A3(n361), .B1(n290), .B2(n291), 
        .ZN(n366) );
  VHSR_NOR2_1 U317 ( .A1(n422), .A2(n423), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U318 ( .A1(n398), .A2(product[0]), .ZN(n363) );
  VHSR_IN_2 U319 ( .I(n363), .ZN(n365) );
  VHSR_CLKNAND2_2 U320 ( .A1(a[4]), .A2(b[1]), .ZN(n292) );
  VHSR_OAI32_2 U321 ( .A1(n293), .A2(n422), .A3(n304), .B1(n292), .B2(n293), 
        .ZN(n364) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[6]), .A2(b[6]), .ZN(n386) );
  VHSR_IN_2 U323 ( .I(n386), .ZN(n413) );
  VHSR_CLKNAND2_2 U324 ( .A1(a[4]), .A2(b[6]), .ZN(n323) );
  VHSR_NAND3_2 U325 ( .A1(b[7]), .A2(a[5]), .A3(n323), .ZN(n295) );
  VHSR_CLKNAND2_2 U326 ( .A1(a[6]), .A2(b[4]), .ZN(n324) );
  VHSR_NAND3_2 U327 ( .A1(a[7]), .A2(b[5]), .A3(n324), .ZN(n294) );
  VHSR_CLKNAND2_2 U328 ( .A1(n295), .A2(n294), .ZN(n297) );
  VHSR_MAOI222_2 U329 ( .A(n386), .B(n295), .C(n294), .ZN(n370) );
  VHSR_IN_2 U330 ( .I(n370), .ZN(n296) );
  VHSR_OAI21_2 U331 ( .A1(n413), .A2(n297), .B(n296), .ZN(n312) );
  VHSR_NOR3_2 U332 ( .A1(n298), .A2(n324), .A3(n300), .ZN(n378) );
  VHSR_AOI22_2 U333 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n299) );
  VHSR_NOR2_1 U334 ( .A1(n378), .A2(n299), .ZN(n308) );
  VHSR_IN_2 U335 ( .I(n398), .ZN(n306) );
  VHSR_NOR3_2 U336 ( .A1(n304), .A2(n300), .A3(n306), .ZN(n329) );
  VHSR_NOR4_2 U337 ( .A1(n304), .A2(n303), .A3(n302), .A4(n301), .ZN(n376) );
  VHSR_AOI22_2 U338 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n305) );
  VHSR_NOR2_1 U339 ( .A1(n376), .A2(n305), .ZN(n307) );
  VHSR_NAND3_2 U340 ( .A1(b[5]), .A2(a[5]), .A3(n306), .ZN(n322) );
  VHSR_MAOI222_2 U341 ( .A(n324), .B(n323), .C(n322), .ZN(n321) );
  VHSR_AND2_2 U342 ( .A1(n317), .A2(n321), .Z(n316) );
  VHSR_AD1_1 U343 ( .A(n308), .B(n329), .CI(n307), .CO(n309), .S(n317) );
  VHSR_NOR2_1 U344 ( .A1(n316), .A2(n309), .ZN(n311) );
  VHSR_CLKNAND2_2 U345 ( .A1(n316), .A2(n309), .ZN(n310) );
  VHSR_NOR2_1 U346 ( .A1(n311), .A2(n312), .ZN(n371) );
  VHSR_AOI22_2 U347 ( .A1(n312), .A2(n311), .B1(n310), .B2(n371), .ZN(n411) );
  VHSR_AD1_1 U348 ( .A(n315), .B(n314), .CI(n313), .CO(n412), .S(n390) );
  VHSR_IAO21_2 U349 ( .A1(n317), .A2(n321), .B(n316), .ZN(n389) );
  VHSR_AD1_1 U350 ( .A(n320), .B(n319), .CI(n318), .CO(n313), .S(n393) );
  VHSR_AOI31_2 U351 ( .A1(n324), .A2(n323), .A3(n322), .B(n321), .ZN(n392) );
  VHSR_AD1_1 U352 ( .A(n327), .B(n326), .CI(n325), .CO(n318), .S(n396) );
  VHSR_AOI22_2 U353 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n328) );
  VHSR_NOR2_1 U354 ( .A1(n329), .A2(n328), .ZN(n395) );
  VHSR_AD1_1 U355 ( .A(n332), .B(n331), .CI(n330), .CO(n325), .S(n399) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[2]), .A2(a[2]), .ZN(n340) );
  VHSR_IN_2 U357 ( .I(n340), .ZN(n355) );
  VHSR_CLKNAND2_2 U358 ( .A1(b[3]), .A2(a[3]), .ZN(n353) );
  VHSR_AOI22_2 U359 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n333) );
  VHSR_IAO21_2 U360 ( .A1(n353), .A2(n340), .B(n333), .ZN(n369) );
  VHSR_CLKNAND2_2 U361 ( .A1(b[2]), .A2(a[0]), .ZN(n427) );
  VHSR_NOR3_2 U362 ( .A1(n348), .A2(n421), .A3(n427), .ZN(n335) );
  VHSR_AOI22_2 U363 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n334) );
  VHSR_NOR2_1 U364 ( .A1(n335), .A2(n334), .ZN(n406) );
  VHSR_NOR2_1 U365 ( .A1(n422), .A2(n336), .ZN(n339) );
  VHSR_NAND3_2 U366 ( .A1(b[1]), .A2(a[3]), .A3(n339), .ZN(n347) );
  VHSR_OAI22_2 U367 ( .A1(n424), .A2(n336), .B1(n422), .B2(n338), .ZN(n337) );
  VHSR_AND2_2 U368 ( .A1(n347), .A2(n337), .Z(n405) );
  VHSR_IN_2 U369 ( .I(n339), .ZN(n428) );
  VHSR_CLKNAND2_2 U370 ( .A1(b[1]), .A2(a[1]), .ZN(n426) );
  VHSR_MAOI222_2 U371 ( .A(n428), .B(n427), .C(n426), .ZN(n425) );
  VHSR_IN_2 U372 ( .I(n360), .ZN(n346) );
  VHSR_NOR3_2 U373 ( .A1(n339), .A2(n338), .A3(n424), .ZN(n344) );
  VHSR_NAND3_2 U374 ( .A1(a[1]), .A2(b[3]), .A3(n427), .ZN(n341) );
  VHSR_CLKNAND2_2 U375 ( .A1(n340), .A2(n341), .ZN(n343) );
  VHSR_IN_2 U376 ( .I(n341), .ZN(n342) );
  VHSR_MAOI222_2 U377 ( .A(n344), .B(n355), .C(n342), .ZN(n345) );
  VHSR_OAI21_2 U378 ( .A1(n344), .A2(n343), .B(n345), .ZN(n359) );
  VHSR_OAI21_2 U379 ( .A1(n346), .A2(n359), .B(n345), .ZN(n368) );
  VHSR_OAI31_2 U380 ( .A1(n421), .A2(n348), .A3(n427), .B(n347), .ZN(n367) );
  VHSR_IAO21_2 U381 ( .A1(n355), .A2(n354), .B(n353), .ZN(n402) );
  VHSR_AD1_1 U382 ( .A(n351), .B(n350), .CI(n349), .CO(n330), .S(n401) );
  VHSR_OAI21_2 U383 ( .A1(n355), .A2(n353), .B(n354), .ZN(n352) );
  VHSR_OAI31_2 U384 ( .A1(n355), .A2(n354), .A3(n353), .B(n352), .ZN(n409) );
  VHSR_AD1_1 U385 ( .A(n358), .B(n357), .CI(n356), .CO(n349), .S(n408) );
  VHSR_CLKXOR2_2 U386 ( .A1(n360), .A2(n359), .Z(n420) );
  VHSR_NOR2_1 U387 ( .A1(n361), .A2(n423), .ZN(n362) );
  VHSR_AOI32_2 U388 ( .A1(b[0]), .A2(n363), .A3(a[4]), .B1(n362), .B2(n363), 
        .ZN(n419) );
  VHSR_NOR2_1 U389 ( .A1(n420), .A2(n419), .ZN(n418) );
  VHSR_AD1_1 U390 ( .A(n366), .B(n365), .CI(n364), .CO(n356), .S(n404) );
  VHSR_AD1_1 U391 ( .A(n369), .B(n368), .CI(n367), .CO(n354), .S(n403) );
  VHSR_NOR2_1 U392 ( .A1(n371), .A2(n370), .ZN(n383) );
  VHSR_CLKNAND2_2 U393 ( .A1(a[7]), .A2(b[6]), .ZN(n373) );
  VHSR_AOI21_2 U394 ( .A1(a[6]), .A2(b[7]), .B(n373), .ZN(n372) );
  VHSR_AOI31_2 U395 ( .A1(a[6]), .A2(n373), .A3(b[7]), .B(n372), .ZN(n374) );
  VHSR_IN_2 U396 ( .I(n374), .ZN(n375) );
  VHSR_OR2_2 U397 ( .A1(n376), .A2(n375), .Z(n377) );
  VHSR_MAOI222_2 U398 ( .A(n378), .B(n376), .C(n375), .ZN(n385) );
  VHSR_OAI21_2 U399 ( .A1(n378), .A2(n377), .B(n385), .ZN(n382) );
  VHSR_CLKXOR2_2 U400 ( .A1(n383), .A2(n382), .Z(n379) );
  VHSR_CLKNAND2_2 U401 ( .A1(n380), .A2(n379), .ZN(n415) );
  VHSR_OAI21_2 U402 ( .A1(n380), .A2(n379), .B(n415), .ZN(n381) );
  VHSR_CLKNAND2_2 U403 ( .A1(a[7]), .A2(b[7]), .ZN(n414) );
  VHSR_NOR2_1 U404 ( .A1(n383), .A2(n382), .ZN(n384) );
  VHSR_AND3_2 U405 ( .A1(n416), .A2(n386), .A3(n415), .Z(n387) );
  VHSR_NOR2_1 U406 ( .A1(n414), .A2(n387), .ZN(product[15]) );
  VHSR_AD1_1 U407 ( .A(n409), .B(n408), .CI(n407), .CO(n400), .S(product[6])
         );
  VHSR_AD1_1 U408 ( .A(n412), .B(n411), .CI(n410), .CO(n380), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U409 ( .A1(n414), .A2(n413), .ZN(n417) );
  VHSR_XOR3_2 U410 ( .A1(n417), .A2(n416), .A3(n415), .Z(product[14]) );
  VHSR_AOI21_2 U411 ( .A1(n420), .A2(n419), .B(n418), .ZN(product[4]) );
  VHSR_OAI22_2 U412 ( .A1(n424), .A2(n423), .B1(n422), .B2(n421), .ZN(
        product[1]) );
  VHSR_AOI31_2 U413 ( .A1(n428), .A2(n427), .A3(n426), .B(n425), .ZN(
        product[2]) );
endmodule


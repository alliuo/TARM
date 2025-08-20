
module mul8_61 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n237, n238, n239, n240, n241, n242, n243, n244,
         n245, n246, n247, n248, n249, n250, n251, n252, n253, n254, n255,
         n256, n257, n258, n259, n260, n261, n262, n263, n264, n265, n266,
         n267, n268, n269, n270, n271, n272, n273, n274, n275, n276, n277,
         n278, n279, n280, n281, n282, n283, n284, n285, n286, n287, n288,
         n289, n290, n291, n292, n293, n294, n295, n296, n297, n298, n299,
         n300, n301, n302, n303, n304, n305, n306, n307, n308, n309, n310,
         n311, n312, n313, n314, n315, n316, n317, n318, n319, n320, n321,
         n322, n323, n324, n325, n326, n327, n328, n329, n330, n331, n332,
         n333, n334, n335, n336, n337, n338, n339, n340, n341, n342, n343,
         n344, n345, n346, n347, n348, n349, n350, n351, n352, n353, n354,
         n355, n356, n357, n358, n359, n360, n361, n362, n363, n364, n365,
         n366, n367, n368, n369, n370, n371, n372, n373, n374, n375, n376,
         n377, n378, n379, n380, n381, n382, n383, n384, n385, n386, n387,
         n388, n389, n390, n391, n392, n393, n394, n395, n396, n397, n398,
         n399, n400, n401, n402, n403, n404, n405, n406, n407, n408, n409,
         n410, n411, n412, n413, n414, n415, n416, n417, n418, n419, n420,
         n421, n422, n423, n424, n425, n426, n427, n428, n429, n430, n431,
         n432, n433, n434, n435, n436, n437, n438, n439, n440, n441, n442,
         n443, n444, n445, n446, n447, n448, n449, n450, n451, n452, n453,
         n454, n455, n456;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U228 ( .A1(n267), .B1(n241), .ZN(n243) );
  VHSR_NOR2_1 U229 ( .A1(n365), .A2(n341), .ZN(n293) );
  VHSR_INOR2_2 U230 ( .A1(n248), .B1(n278), .ZN(n268) );
  VHSR_INOR2_2 U231 ( .A1(n424), .B1(n311), .ZN(n315) );
  VHSR_INOR2_2 U232 ( .A1(product[0]), .B1(n352), .ZN(n362) );
  VHSR_INOR2_2 U233 ( .A1(n246), .B1(n284), .ZN(n280) );
  VHSR_INOR2_2 U234 ( .A1(n397), .B1(n396), .ZN(n409) );
  VHSR_INOR2_2 U235 ( .A1(n367), .B1(n390), .ZN(n389) );
  VHSR_NOR2_1 U236 ( .A1(n265), .A2(n264), .ZN(n325) );
  VHSR_INOR2_2 U237 ( .A1(n411), .B1(n410), .ZN(n449) );
  VHSR_IN_2 U238 ( .I(n407), .ZN(product[13]) );
  VHSR_CLKN_1 U239 ( .I(n412), .ZN(n413) );
  VHSR_INAND3_1 U240 ( .A1(n446), .B1(n449), .B2(n448), .ZN(n412) );
  VHSR_INAND2_1 U241 ( .A1(n402), .B1(n400), .ZN(n403) );
  VHSR_AD1_1 U242 ( .A(n431), .B(n430), .CI(n429), .CO(n426), .S(product[6])
         );
  VHSR_AD1_1 U243 ( .A(n425), .B(n424), .CI(n423), .CO(n420), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U244 ( .A(n419), .B(n418), .CI(n417), .CO(n414), .S(product[10])
         );
  VHSR_AD1_1 U245 ( .A(n433), .B(n454), .CI(n432), .CO(n429), .S(product[5])
         );
  VHSR_AD1_1 U246 ( .A(n428), .B(n427), .CI(n426), .CO(n423), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U247 ( .A(n422), .B(n421), .CI(n420), .CO(n417), .S(product[9])
         );
  VHSR_AD1_1 U248 ( .A(n416), .B(n415), .CI(n414), .CO(n434), .S(
        \intadd_0/SUM[6] ) );
  VHSR_IN_2 U249 ( .I(b[0]), .ZN(n438) );
  VHSR_IN_2 U250 ( .I(a[0]), .ZN(n440) );
  VHSR_NOR2_1 U251 ( .A1(n438), .A2(n440), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U252 ( .A1(b[1]), .A2(a[1]), .ZN(n352) );
  VHSR_NOR2_1 U253 ( .A1(product[0]), .A2(n352), .ZN(n237) );
  VHSR_IN_2 U254 ( .I(a[2]), .ZN(n364) );
  VHSR_NOR2_1 U255 ( .A1(n438), .A2(n364), .ZN(n370) );
  VHSR_IN_2 U256 ( .I(b[2]), .ZN(n365) );
  VHSR_NOR2_1 U257 ( .A1(n365), .A2(n440), .ZN(n371) );
  VHSR_OR2_2 U258 ( .A1(n370), .A2(n371), .Z(n350) );
  VHSR_AOI22_2 U259 ( .A1(n370), .A2(n371), .B1(n237), .B2(n350), .ZN(n445) );
  VHSR_OAI21_2 U260 ( .A1(n237), .A2(n350), .B(n445), .ZN(n238) );
  VHSR_IN_2 U261 ( .I(n238), .ZN(product[2]) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[3]), .A2(a[7]), .ZN(n265) );
  VHSR_IN_2 U263 ( .I(b[3]), .ZN(n355) );
  VHSR_IN_2 U264 ( .I(a[6]), .ZN(n299) );
  VHSR_IN_2 U265 ( .I(a[7]), .ZN(n305) );
  VHSR_OAI22_2 U266 ( .A1(n355), .A2(n299), .B1(n305), .B2(n365), .ZN(n270) );
  VHSR_IN_2 U267 ( .I(a[4]), .ZN(n341) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[3]), .A2(a[5]), .ZN(n239) );
  VHSR_IN_2 U269 ( .I(b[1]), .ZN(n442) );
  VHSR_OAI22_2 U270 ( .A1(n293), .A2(n239), .B1(n305), .B2(n442), .ZN(n247) );
  VHSR_CLKNAND2_2 U271 ( .A1(a[5]), .A2(b[1]), .ZN(n242) );
  VHSR_NOR3_2 U272 ( .A1(n293), .A2(n265), .A3(n242), .ZN(n240) );
  VHSR_AOI31_2 U273 ( .A1(b[2]), .A2(a[6]), .A3(n247), .B(n240), .ZN(n248) );
  VHSR_IN_2 U274 ( .I(a[5]), .ZN(n337) );
  VHSR_NOR4_2 U275 ( .A1(n341), .A2(n337), .A3(n442), .A4(n438), .ZN(n298) );
  VHSR_NAND3_2 U276 ( .A1(b[3]), .A2(n293), .A3(a[5]), .ZN(n267) );
  VHSR_AOI22_2 U277 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n241) );
  VHSR_OAI22_2 U278 ( .A1(n305), .A2(n438), .B1(n299), .B2(n442), .ZN(n244) );
  VHSR_MAOI222_2 U279 ( .A(n298), .B(n243), .C(n244), .ZN(n246) );
  VHSR_NOR2_1 U280 ( .A1(n299), .A2(n438), .ZN(n292) );
  VHSR_AOI21_2 U281 ( .A1(a[4]), .A2(b[0]), .B(n242), .ZN(n291) );
  VHSR_MAOI222_2 U282 ( .A(n293), .B(n292), .C(n291), .ZN(n290) );
  VHSR_OR2_2 U283 ( .A1(n298), .A2(n243), .Z(n245) );
  VHSR_OAI21_2 U284 ( .A1(n245), .A2(n244), .B(n246), .ZN(n285) );
  VHSR_NOR2_1 U285 ( .A1(n290), .A2(n285), .ZN(n284) );
  VHSR_AOI32_2 U286 ( .A1(b[2]), .A2(n248), .A3(a[6]), .B1(n247), .B2(n248), 
        .ZN(n279) );
  VHSR_NOR2_1 U287 ( .A1(n280), .A2(n279), .ZN(n278) );
  VHSR_CLKNAND2_2 U288 ( .A1(n268), .A2(n267), .ZN(n266) );
  VHSR_CLKNAND2_2 U289 ( .A1(n270), .A2(n266), .ZN(n264) );
  VHSR_CLKNAND2_2 U290 ( .A1(b[6]), .A2(a[2]), .ZN(n256) );
  VHSR_IN_2 U291 ( .I(n256), .ZN(n263) );
  VHSR_IN_2 U292 ( .I(b[5]), .ZN(n339) );
  VHSR_IN_2 U293 ( .I(a[3]), .ZN(n356) );
  VHSR_CLKNAND2_2 U294 ( .A1(b[4]), .A2(a[2]), .ZN(n289) );
  VHSR_NOR3_2 U295 ( .A1(n339), .A2(n356), .A3(n289), .ZN(n273) );
  VHSR_CLKNAND2_2 U296 ( .A1(b[7]), .A2(a[3]), .ZN(n261) );
  VHSR_AOI22_2 U297 ( .A1(b[6]), .A2(a[3]), .B1(b[7]), .B2(a[2]), .ZN(n249) );
  VHSR_IAO21_2 U298 ( .A1(n261), .A2(n256), .B(n249), .ZN(n272) );
  VHSR_IN_2 U299 ( .I(a[1]), .ZN(n437) );
  VHSR_CLKNAND2_2 U300 ( .A1(b[4]), .A2(a[0]), .ZN(n452) );
  VHSR_NOR3_2 U301 ( .A1(n339), .A2(n437), .A3(n452), .ZN(n296) );
  VHSR_AOI22_2 U302 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n250) );
  VHSR_NOR2_1 U303 ( .A1(n273), .A2(n250), .ZN(n254) );
  VHSR_IN_2 U304 ( .I(b[6]), .ZN(n308) );
  VHSR_IN_2 U305 ( .I(b[7]), .ZN(n307) );
  VHSR_OAI22_2 U306 ( .A1(n308), .A2(n437), .B1(n307), .B2(n440), .ZN(n253) );
  VHSR_IN_2 U307 ( .I(n275), .ZN(n259) );
  VHSR_NOR2_1 U308 ( .A1(n307), .A2(n437), .ZN(n252) );
  VHSR_NAND3_2 U309 ( .A1(n289), .A2(a[3]), .A3(b[5]), .ZN(n255) );
  VHSR_IN_2 U310 ( .I(n255), .ZN(n251) );
  VHSR_MAOI222_2 U311 ( .A(n252), .B(n263), .C(n251), .ZN(n258) );
  VHSR_AD1_1 U312 ( .A(n296), .B(n254), .CI(n253), .CO(n275), .S(n282) );
  VHSR_NAND3_2 U313 ( .A1(a[1]), .A2(b[5]), .A3(n452), .ZN(n288) );
  VHSR_CLKNAND2_2 U314 ( .A1(b[6]), .A2(a[0]), .ZN(n287) );
  VHSR_MAOI222_2 U315 ( .A(n289), .B(n288), .C(n287), .ZN(n286) );
  VHSR_CLKNAND2_2 U316 ( .A1(n282), .A2(n286), .ZN(n281) );
  VHSR_CLKNAND2_2 U317 ( .A1(n256), .A2(n255), .ZN(n257) );
  VHSR_AOI32_2 U318 ( .A1(a[1]), .A2(n258), .A3(b[7]), .B1(n257), .B2(n258), 
        .ZN(n274) );
  VHSR_AOI32_2 U319 ( .A1(n259), .A2(n258), .A3(n281), .B1(n274), .B2(n258), 
        .ZN(n271) );
  VHSR_IAO21_2 U320 ( .A1(n263), .A2(n262), .B(n261), .ZN(n324) );
  VHSR_OAI21_2 U321 ( .A1(n263), .A2(n261), .B(n262), .ZN(n260) );
  VHSR_OAI31_2 U322 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n332) );
  VHSR_AOI21_2 U323 ( .A1(n265), .A2(n264), .B(n325), .ZN(n331) );
  VHSR_OAI21_2 U324 ( .A1(n268), .A2(n267), .B(n266), .ZN(n269) );
  VHSR_XNOR2_2 U325 ( .A1(n270), .A2(n269), .ZN(n335) );
  VHSR_AD1_1 U326 ( .A(n273), .B(n272), .CI(n271), .CO(n262), .S(n334) );
  VHSR_NOR2_1 U327 ( .A1(n275), .A2(n274), .ZN(n277) );
  VHSR_AOI22_2 U328 ( .A1(n275), .A2(n274), .B1(n281), .B2(n277), .ZN(n276) );
  VHSR_OAI21_2 U329 ( .A1(n281), .A2(n277), .B(n276), .ZN(n344) );
  VHSR_AOI21_2 U330 ( .A1(n280), .A2(n279), .B(n278), .ZN(n343) );
  VHSR_OAI21_2 U331 ( .A1(n282), .A2(n286), .B(n281), .ZN(n283) );
  VHSR_IN_2 U332 ( .I(n283), .ZN(n347) );
  VHSR_AOI21_2 U333 ( .A1(n290), .A2(n285), .B(n284), .ZN(n346) );
  VHSR_AOI31_2 U334 ( .A1(n289), .A2(n288), .A3(n287), .B(n286), .ZN(n382) );
  VHSR_OAI31_2 U335 ( .A1(n293), .A2(n292), .A3(n291), .B(n290), .ZN(n294) );
  VHSR_IN_2 U336 ( .I(n294), .ZN(n381) );
  VHSR_AOI22_2 U337 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n295) );
  VHSR_NOR2_1 U338 ( .A1(n296), .A2(n295), .ZN(n395) );
  VHSR_CLKNAND2_2 U339 ( .A1(a[4]), .A2(b[0]), .ZN(n453) );
  VHSR_NOR2_1 U340 ( .A1(n453), .A2(n452), .ZN(n451) );
  VHSR_AOI22_2 U341 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n297) );
  VHSR_NOR2_1 U342 ( .A1(n298), .A2(n297), .ZN(n394) );
  VHSR_NOR2_1 U343 ( .A1(n299), .A2(n308), .ZN(n446) );
  VHSR_NOR2_1 U344 ( .A1(n341), .A2(n308), .ZN(n312) );
  VHSR_CLKNAND2_2 U345 ( .A1(a[5]), .A2(b[7]), .ZN(n301) );
  VHSR_CLKNAND2_2 U346 ( .A1(a[6]), .A2(b[4]), .ZN(n304) );
  VHSR_IN_2 U347 ( .I(n304), .ZN(n313) );
  VHSR_CLKNAND2_2 U348 ( .A1(a[7]), .A2(b[5]), .ZN(n300) );
  VHSR_OAI22_2 U349 ( .A1(n312), .A2(n301), .B1(n313), .B2(n300), .ZN(n303) );
  VHSR_OR2_2 U350 ( .A1(n312), .A2(n313), .Z(n327) );
  VHSR_CLKNAND2_2 U351 ( .A1(a[5]), .A2(b[5]), .ZN(n311) );
  VHSR_CLKNAND2_2 U352 ( .A1(a[7]), .A2(b[7]), .ZN(n447) );
  VHSR_NOR3_2 U353 ( .A1(n327), .A2(n311), .A3(n447), .ZN(n302) );
  VHSR_AOI31_2 U354 ( .A1(b[6]), .A2(a[6]), .A3(n303), .B(n302), .ZN(n397) );
  VHSR_OAI21_2 U355 ( .A1(n446), .A2(n303), .B(n397), .ZN(n320) );
  VHSR_NOR3_2 U356 ( .A1(n305), .A2(n304), .A3(n339), .ZN(n404) );
  VHSR_AOI22_2 U357 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n306) );
  VHSR_NOR2_1 U358 ( .A1(n404), .A2(n306), .ZN(n316) );
  VHSR_IN_2 U359 ( .I(b[4]), .ZN(n336) );
  VHSR_NOR2_1 U360 ( .A1(n341), .A2(n336), .ZN(n424) );
  VHSR_NOR4_2 U361 ( .A1(n341), .A2(n337), .A3(n308), .A4(n307), .ZN(n402) );
  VHSR_AOI22_2 U362 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n309) );
  VHSR_NOR2_1 U363 ( .A1(n402), .A2(n309), .ZN(n314) );
  VHSR_IN_2 U364 ( .I(n310), .ZN(n322) );
  VHSR_NOR2_1 U365 ( .A1(n424), .A2(n311), .ZN(n328) );
  VHSR_AOI22_2 U366 ( .A1(n313), .A2(n312), .B1(n328), .B2(n327), .ZN(n326) );
  VHSR_NOR2_1 U367 ( .A1(n322), .A2(n326), .ZN(n321) );
  VHSR_AD1_1 U368 ( .A(n316), .B(n315), .CI(n314), .CO(n317), .S(n310) );
  VHSR_NOR2_1 U369 ( .A1(n321), .A2(n317), .ZN(n319) );
  VHSR_CLKNAND2_2 U370 ( .A1(n321), .A2(n317), .ZN(n318) );
  VHSR_NOR2_1 U371 ( .A1(n319), .A2(n320), .ZN(n396) );
  VHSR_AOI22_2 U372 ( .A1(n320), .A2(n319), .B1(n318), .B2(n396), .ZN(n435) );
  VHSR_AOI21_2 U373 ( .A1(n326), .A2(n322), .B(n321), .ZN(n416) );
  VHSR_AD1_1 U374 ( .A(n325), .B(n324), .CI(n323), .CO(n436), .S(n415) );
  VHSR_OAI21_2 U375 ( .A1(n328), .A2(n327), .B(n326), .ZN(n329) );
  VHSR_IN_2 U376 ( .I(n329), .ZN(n419) );
  VHSR_AD1_1 U377 ( .A(n332), .B(n331), .CI(n330), .CO(n323), .S(n418) );
  VHSR_AD1_1 U378 ( .A(n335), .B(n334), .CI(n333), .CO(n330), .S(n422) );
  VHSR_NOR2_1 U379 ( .A1(n337), .A2(n336), .ZN(n340) );
  VHSR_OAI21_2 U380 ( .A1(n341), .A2(n339), .B(n340), .ZN(n338) );
  VHSR_OAI31_2 U381 ( .A1(n341), .A2(n340), .A3(n339), .B(n338), .ZN(n421) );
  VHSR_AD1_1 U382 ( .A(n344), .B(n343), .CI(n342), .CO(n333), .S(n425) );
  VHSR_AD1_1 U383 ( .A(n347), .B(n346), .CI(n345), .CO(n342), .S(n428) );
  VHSR_CLKNAND2_2 U384 ( .A1(b[3]), .A2(a[1]), .ZN(n349) );
  VHSR_CLKNAND2_2 U385 ( .A1(b[1]), .A2(a[3]), .ZN(n348) );
  VHSR_OAI22_2 U386 ( .A1(n371), .A2(n349), .B1(n370), .B2(n348), .ZN(n366) );
  VHSR_CLKNAND2_2 U387 ( .A1(b[3]), .A2(a[3]), .ZN(n386) );
  VHSR_NOR3_2 U388 ( .A1(n386), .A2(n352), .A3(n350), .ZN(n351) );
  VHSR_AOI31_2 U389 ( .A1(a[2]), .A2(b[2]), .A3(n366), .B(n351), .ZN(n367) );
  VHSR_NOR2_1 U390 ( .A1(n365), .A2(n437), .ZN(n354) );
  VHSR_OAI21_2 U391 ( .A1(n355), .A2(n440), .B(n354), .ZN(n353) );
  VHSR_OAI31_2 U392 ( .A1(n355), .A2(n354), .A3(n440), .B(n353), .ZN(n361) );
  VHSR_NOR2_1 U393 ( .A1(n438), .A2(n356), .ZN(n358) );
  VHSR_OAI21_2 U394 ( .A1(n442), .A2(n364), .B(n358), .ZN(n357) );
  VHSR_OAI31_2 U395 ( .A1(n442), .A2(n358), .A3(n364), .B(n357), .ZN(n360) );
  VHSR_IN_2 U396 ( .I(n359), .ZN(n444) );
  VHSR_NOR2_1 U397 ( .A1(n444), .A2(n445), .ZN(n443) );
  VHSR_AD1_1 U398 ( .A(n362), .B(n361), .CI(n360), .CO(n363), .S(n359) );
  VHSR_NOR2_1 U399 ( .A1(n443), .A2(n363), .ZN(n392) );
  VHSR_NOR2_1 U400 ( .A1(n365), .A2(n364), .ZN(n385) );
  VHSR_OAI21_2 U401 ( .A1(n385), .A2(n366), .B(n367), .ZN(n391) );
  VHSR_NOR2_1 U402 ( .A1(n392), .A2(n391), .ZN(n390) );
  VHSR_CLKNAND2_2 U403 ( .A1(b[2]), .A2(a[3]), .ZN(n369) );
  VHSR_AOI21_2 U404 ( .A1(b[3]), .A2(a[2]), .B(n369), .ZN(n368) );
  VHSR_AOI31_2 U405 ( .A1(b[3]), .A2(n369), .A3(a[2]), .B(n368), .ZN(n378) );
  VHSR_NAND3_2 U406 ( .A1(b[1]), .A2(a[3]), .A3(n370), .ZN(n377) );
  VHSR_IN_2 U407 ( .I(n377), .ZN(n373) );
  VHSR_NAND3_2 U408 ( .A1(b[3]), .A2(n371), .A3(a[1]), .ZN(n376) );
  VHSR_IN_2 U409 ( .I(n376), .ZN(n372) );
  VHSR_NOR2_1 U410 ( .A1(n373), .A2(n372), .ZN(n375) );
  VHSR_AOI22_2 U411 ( .A1(n373), .A2(n372), .B1(n378), .B2(n375), .ZN(n374) );
  VHSR_OAI21_2 U412 ( .A1(n378), .A2(n375), .B(n374), .ZN(n388) );
  VHSR_NOR2_1 U413 ( .A1(n389), .A2(n388), .ZN(n387) );
  VHSR_MAOI222_2 U414 ( .A(n378), .B(n377), .C(n376), .ZN(n379) );
  VHSR_OR2_2 U415 ( .A1(n387), .A2(n379), .Z(n384) );
  VHSR_IAO21_2 U416 ( .A1(n384), .A2(n385), .B(n386), .ZN(n427) );
  VHSR_AD1_1 U417 ( .A(n382), .B(n381), .CI(n380), .CO(n345), .S(n431) );
  VHSR_OAI21_2 U418 ( .A1(n386), .A2(n385), .B(n384), .ZN(n383) );
  VHSR_OAI31_2 U419 ( .A1(n386), .A2(n385), .A3(n384), .B(n383), .ZN(n430) );
  VHSR_AOI21_2 U420 ( .A1(n389), .A2(n388), .B(n387), .ZN(n433) );
  VHSR_AOI21_2 U421 ( .A1(n392), .A2(n391), .B(n390), .ZN(n456) );
  VHSR_IN_2 U422 ( .I(n456), .ZN(n393) );
  VHSR_AOI211_2 U423 ( .A1(n453), .A2(n452), .B(n451), .C(n393), .ZN(n454) );
  VHSR_AD1_1 U424 ( .A(n395), .B(n451), .CI(n394), .CO(n380), .S(n432) );
  VHSR_CLKNAND2_2 U425 ( .A1(a[6]), .A2(b[7]), .ZN(n399) );
  VHSR_AOI21_2 U426 ( .A1(a[7]), .A2(b[6]), .B(n399), .ZN(n398) );
  VHSR_AOI31_2 U427 ( .A1(a[7]), .A2(n399), .A3(b[6]), .B(n398), .ZN(n400) );
  VHSR_IN_2 U428 ( .I(n400), .ZN(n401) );
  VHSR_MAOI222_2 U429 ( .A(n404), .B(n402), .C(n401), .ZN(n411) );
  VHSR_OAI21_2 U430 ( .A1(n404), .A2(n403), .B(n411), .ZN(n408) );
  VHSR_CLKXOR2_2 U431 ( .A1(n409), .A2(n408), .Z(n405) );
  VHSR_CLKNAND2_2 U432 ( .A1(n406), .A2(n405), .ZN(n448) );
  VHSR_OAI21_2 U433 ( .A1(n406), .A2(n405), .B(n448), .ZN(n407) );
  VHSR_NOR2_1 U434 ( .A1(n409), .A2(n408), .ZN(n410) );
  VHSR_NOR2_1 U435 ( .A1(n447), .A2(n413), .ZN(product[15]) );
  VHSR_AD1_1 U436 ( .A(n436), .B(n435), .CI(n434), .CO(n406), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U437 ( .A1(n438), .A2(n437), .ZN(n441) );
  VHSR_OAI21_2 U438 ( .A1(n442), .A2(n440), .B(n441), .ZN(n439) );
  VHSR_OAI31_2 U439 ( .A1(n442), .A2(n441), .A3(n440), .B(n439), .ZN(
        product[1]) );
  VHSR_AOI21_2 U440 ( .A1(n445), .A2(n444), .B(n443), .ZN(product[3]) );
  VHSR_NOR2_1 U441 ( .A1(n447), .A2(n446), .ZN(n450) );
  VHSR_XOR3_2 U442 ( .A1(n450), .A2(n449), .A3(n448), .Z(product[14]) );
  VHSR_AOI21_2 U443 ( .A1(n453), .A2(n452), .B(n451), .ZN(n455) );
  VHSR_IAO21_2 U444 ( .A1(n456), .A2(n455), .B(n454), .ZN(product[4]) );
endmodule


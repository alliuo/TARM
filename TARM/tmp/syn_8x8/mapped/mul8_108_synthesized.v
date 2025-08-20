
module mul8_108 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n223, n224, n225, n226, n227, n228, n229, n230,
         n231, n232, n233, n234, n235, n236, n237, n238, n239, n240, n241,
         n242, n243, n244, n245, n246, n247, n248, n249, n250, n251, n252,
         n253, n254, n255, n256, n257, n258, n259, n260, n261, n262, n263,
         n264, n265, n266, n267, n268, n269, n270, n271, n272, n273, n274,
         n275, n276, n277, n278, n279, n280, n281, n282, n283, n284, n285,
         n286, n287, n288, n289, n290, n291, n292, n293, n294, n295, n296,
         n297, n298, n299, n300, n301, n302, n303, n304, n305, n306, n307,
         n308, n309, n310, n311, n312, n313, n314, n315, n316, n317, n318,
         n319, n320, n321, n322, n323, n324, n325, n326, n327, n328, n329,
         n330, n331, n332, n333, n334, n335, n336, n337, n338, n339, n340,
         n341, n342, n343, n344, n345, n346, n347, n348, n349, n350, n351,
         n352, n353, n354, n355, n356, n357, n358, n359, n360, n361, n362,
         n363, n364, n365, n366, n367, n368, n369, n370, n371, n372, n373,
         n374, n375, n376, n377, n378, n379, n380, n381, n382, n383, n384,
         n385, n386, n387, n388, n389, n390, n391, n392, n393, n394, n395,
         n396, n397, n398, n399, n400, n401, n402, n403, n404, n405, n406,
         n407, n408, n409, n410, n411, n412, n413, n414, n415, n416, n417,
         n418, n419, n420, n421;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U212 ( .A1(n388), .B1(n298), .ZN(n302) );
  VHSR_INOR2_2 U213 ( .A1(n361), .B1(n360), .ZN(n373) );
  VHSR_NOR2_1 U214 ( .A1(n309), .A2(n313), .ZN(n308) );
  VHSR_INOR3_2 U215 ( .A1(n251), .B1(n335), .B2(n292), .ZN(n312) );
  VHSR_IOA21_2 U216 ( .A1(n410), .A2(n409), .B(n408), .ZN(n412) );
  VHSR_NOR2_1 U217 ( .A1(n328), .A2(n323), .ZN(n388) );
  VHSR_IN_2 U218 ( .I(n371), .ZN(product[13]) );
  VHSR_INOR2_1 U219 ( .A1(n375), .B1(n374), .ZN(n406) );
  VHSR_INAND2_1 U220 ( .A1(n366), .B1(n364), .ZN(n367) );
  VHSR_MOAI22_1 U221 ( .A1(n415), .A2(n334), .B1(b[1]), .B2(a[2]), .ZN(n398)
         );
  VHSR_AD1_1 U222 ( .A(n395), .B(n394), .CI(n393), .CO(n390), .S(product[6])
         );
  VHSR_AD1_1 U223 ( .A(n389), .B(n388), .CI(n387), .CO(n384), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n383), .B(n382), .CI(n381), .CO(n378), .S(product[10])
         );
  VHSR_AD1_1 U225 ( .A(n399), .B(n418), .CI(n398), .CO(n333), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n397), .B(n396), .CI(n411), .CO(n393), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n392), .B(n391), .CI(n390), .CO(n387), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U228 ( .A(n386), .B(n385), .CI(n384), .CO(n381), .S(product[9])
         );
  VHSR_AD1_1 U229 ( .A(n380), .B(n379), .CI(n378), .CO(n400), .S(
        \intadd_0/SUM[6] ) );
  VHSR_AOI22_2 U230 ( .A1(b[3]), .A2(a[6]), .B1(a[7]), .B2(b[2]), .ZN(n257) );
  VHSR_IN_2 U231 ( .I(b[3]), .ZN(n335) );
  VHSR_CLKNAND2_2 U232 ( .A1(b[2]), .A2(a[4]), .ZN(n282) );
  VHSR_IN_2 U233 ( .I(a[5]), .ZN(n324) );
  VHSR_NOR3_2 U234 ( .A1(n335), .A2(n282), .A3(n324), .ZN(n255) );
  VHSR_IN_2 U235 ( .I(a[7]), .ZN(n292) );
  VHSR_IN_2 U236 ( .I(b[1]), .ZN(n417) );
  VHSR_NOR2_1 U237 ( .A1(n292), .A2(n417), .ZN(n224) );
  VHSR_AOI211_2 U238 ( .A1(a[4]), .A2(b[2]), .B(n335), .C(n324), .ZN(n225) );
  VHSR_CLKNAND2_2 U239 ( .A1(a[6]), .A2(b[2]), .ZN(n227) );
  VHSR_IN_2 U240 ( .I(n227), .ZN(n223) );
  VHSR_MAOI222_2 U241 ( .A(n224), .B(n225), .C(n223), .ZN(n236) );
  VHSR_AOI21_2 U242 ( .A1(b[1]), .A2(a[7]), .B(n225), .ZN(n228) );
  VHSR_IN_2 U243 ( .I(n236), .ZN(n226) );
  VHSR_AOI21_2 U244 ( .A1(n228), .A2(n227), .B(n226), .ZN(n263) );
  VHSR_IN_2 U245 ( .I(a[4]), .ZN(n328) );
  VHSR_IN_2 U246 ( .I(b[0]), .ZN(n415) );
  VHSR_NOR4_2 U247 ( .A1(n328), .A2(n324), .A3(n417), .A4(n415), .ZN(n286) );
  VHSR_AOI22_2 U248 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n229) );
  VHSR_NOR2_1 U249 ( .A1(n255), .A2(n229), .ZN(n231) );
  VHSR_AOI22_2 U250 ( .A1(a[7]), .A2(b[0]), .B1(a[6]), .B2(b[1]), .ZN(n233) );
  VHSR_IN_2 U251 ( .I(n233), .ZN(n230) );
  VHSR_MAOI222_2 U252 ( .A(n286), .B(n231), .C(n230), .ZN(n235) );
  VHSR_OAI211_2 U253 ( .A1(n328), .A2(n415), .B(a[5]), .C(b[1]), .ZN(n281) );
  VHSR_CLKNAND2_2 U254 ( .A1(a[6]), .A2(b[0]), .ZN(n280) );
  VHSR_MAOI222_2 U255 ( .A(n282), .B(n281), .C(n280), .ZN(n279) );
  VHSR_NOR2_1 U256 ( .A1(n286), .A2(n231), .ZN(n234) );
  VHSR_IN_2 U257 ( .I(n235), .ZN(n232) );
  VHSR_AOI21_2 U258 ( .A1(n234), .A2(n233), .B(n232), .ZN(n273) );
  VHSR_CLKNAND2_2 U259 ( .A1(n279), .A2(n273), .ZN(n272) );
  VHSR_CLKNAND2_2 U260 ( .A1(n235), .A2(n272), .ZN(n262) );
  VHSR_CLKNAND2_2 U261 ( .A1(n263), .A2(n262), .ZN(n261) );
  VHSR_CLKNAND2_2 U262 ( .A1(n236), .A2(n261), .ZN(n254) );
  VHSR_NOR2_1 U263 ( .A1(n255), .A2(n254), .ZN(n253) );
  VHSR_NOR2_1 U264 ( .A1(n257), .A2(n253), .ZN(n251) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[2]), .ZN(n238) );
  VHSR_IN_2 U266 ( .I(n238), .ZN(n250) );
  VHSR_IN_2 U267 ( .I(b[5]), .ZN(n326) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[4]), .A2(a[2]), .ZN(n276) );
  VHSR_IN_2 U269 ( .I(a[3]), .ZN(n334) );
  VHSR_NOR3_2 U270 ( .A1(n326), .A2(n276), .A3(n334), .ZN(n260) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[7]), .A2(a[3]), .ZN(n248) );
  VHSR_AOI22_2 U272 ( .A1(b[6]), .A2(a[3]), .B1(b[7]), .B2(a[2]), .ZN(n237) );
  VHSR_IAO21_2 U273 ( .A1(n248), .A2(n238), .B(n237), .ZN(n259) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[4]), .A2(a[0]), .ZN(n409) );
  VHSR_NAND3_2 U275 ( .A1(b[5]), .A2(a[1]), .A3(n409), .ZN(n278) );
  VHSR_CLKNAND2_2 U276 ( .A1(b[6]), .A2(a[0]), .ZN(n277) );
  VHSR_MAOI222_2 U277 ( .A(n278), .B(n277), .C(n276), .ZN(n275) );
  VHSR_IN_2 U278 ( .I(a[1]), .ZN(n414) );
  VHSR_NOR3_2 U279 ( .A1(n326), .A2(n409), .A3(n414), .ZN(n283) );
  VHSR_AOI22_2 U280 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n239) );
  VHSR_NOR2_1 U281 ( .A1(n239), .A2(n260), .ZN(n242) );
  VHSR_IN_2 U282 ( .I(b[6]), .ZN(n295) );
  VHSR_IN_2 U283 ( .I(b[7]), .ZN(n294) );
  VHSR_IN_2 U284 ( .I(a[0]), .ZN(n416) );
  VHSR_OAI22_2 U285 ( .A1(n295), .A2(n414), .B1(n294), .B2(n416), .ZN(n241) );
  VHSR_CLKNAND2_2 U286 ( .A1(n275), .A2(n270), .ZN(n269) );
  VHSR_NOR2_1 U287 ( .A1(n294), .A2(n414), .ZN(n240) );
  VHSR_AOI211_2 U288 ( .A1(b[4]), .A2(a[2]), .B(n326), .C(n334), .ZN(n243) );
  VHSR_MAOI222_2 U289 ( .A(n240), .B(n250), .C(n243), .ZN(n246) );
  VHSR_AD1_1 U290 ( .A(n283), .B(n242), .CI(n241), .CO(n266), .S(n270) );
  VHSR_IN_2 U291 ( .I(n266), .ZN(n245) );
  VHSR_OR2_2 U292 ( .A1(n243), .A2(n250), .Z(n244) );
  VHSR_AOI32_2 U293 ( .A1(a[1]), .A2(n246), .A3(b[7]), .B1(n244), .B2(n246), 
        .ZN(n265) );
  VHSR_AOI32_2 U294 ( .A1(n269), .A2(n246), .A3(n245), .B1(n265), .B2(n246), 
        .ZN(n258) );
  VHSR_IAO21_2 U295 ( .A1(n250), .A2(n249), .B(n248), .ZN(n311) );
  VHSR_OAI21_2 U296 ( .A1(n250), .A2(n248), .B(n249), .ZN(n247) );
  VHSR_OAI31_2 U297 ( .A1(n250), .A2(n249), .A3(n248), .B(n247), .ZN(n319) );
  VHSR_NOR2_1 U298 ( .A1(n335), .A2(n292), .ZN(n252) );
  VHSR_IAO21_2 U299 ( .A1(n252), .A2(n251), .B(n312), .ZN(n318) );
  VHSR_AOI21_2 U300 ( .A1(n255), .A2(n254), .B(n253), .ZN(n256) );
  VHSR_XNOR2_2 U301 ( .A1(n257), .A2(n256), .ZN(n322) );
  VHSR_AD1_1 U302 ( .A(n260), .B(n259), .CI(n258), .CO(n249), .S(n321) );
  VHSR_OAI21_2 U303 ( .A1(n263), .A2(n262), .B(n261), .ZN(n264) );
  VHSR_IN_2 U304 ( .I(n264), .ZN(n331) );
  VHSR_NOR2_1 U305 ( .A1(n266), .A2(n265), .ZN(n268) );
  VHSR_AOI22_2 U306 ( .A1(n266), .A2(n265), .B1(n269), .B2(n268), .ZN(n267) );
  VHSR_OAI21_2 U307 ( .A1(n269), .A2(n268), .B(n267), .ZN(n330) );
  VHSR_OAI21_2 U308 ( .A1(n275), .A2(n270), .B(n269), .ZN(n271) );
  VHSR_IN_2 U309 ( .I(n271), .ZN(n344) );
  VHSR_OAI21_2 U310 ( .A1(n279), .A2(n273), .B(n272), .ZN(n274) );
  VHSR_IN_2 U311 ( .I(n274), .ZN(n343) );
  VHSR_AOI31_2 U312 ( .A1(n278), .A2(n277), .A3(n276), .B(n275), .ZN(n351) );
  VHSR_AOI31_2 U313 ( .A1(n282), .A2(n281), .A3(n280), .B(n279), .ZN(n350) );
  VHSR_AOI22_2 U314 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n284) );
  VHSR_NOR2_1 U315 ( .A1(n284), .A2(n283), .ZN(n353) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[5]), .A2(b[0]), .ZN(n285) );
  VHSR_OAI32_2 U317 ( .A1(n286), .A2(n417), .A3(n328), .B1(n285), .B2(n286), 
        .ZN(n352) );
  VHSR_IN_2 U318 ( .I(b[4]), .ZN(n323) );
  VHSR_NOR2_1 U319 ( .A1(n415), .A2(n416), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U320 ( .A1(n388), .A2(product[0]), .ZN(n408) );
  VHSR_IN_2 U321 ( .I(n408), .ZN(n359) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[6]), .A2(b[6]), .ZN(n376) );
  VHSR_IN_2 U323 ( .I(n376), .ZN(n403) );
  VHSR_NOR2_1 U324 ( .A1(n328), .A2(n295), .ZN(n299) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[5]), .A2(b[7]), .ZN(n288) );
  VHSR_CLKNAND2_2 U326 ( .A1(a[6]), .A2(b[4]), .ZN(n291) );
  VHSR_IN_2 U327 ( .I(n291), .ZN(n300) );
  VHSR_CLKNAND2_2 U328 ( .A1(a[7]), .A2(b[5]), .ZN(n287) );
  VHSR_OAI22_2 U329 ( .A1(n299), .A2(n288), .B1(n300), .B2(n287), .ZN(n290) );
  VHSR_OR2_2 U330 ( .A1(n299), .A2(n300), .Z(n314) );
  VHSR_CLKNAND2_2 U331 ( .A1(a[5]), .A2(b[5]), .ZN(n298) );
  VHSR_CLKNAND2_2 U332 ( .A1(a[7]), .A2(b[7]), .ZN(n404) );
  VHSR_NOR3_2 U333 ( .A1(n314), .A2(n298), .A3(n404), .ZN(n289) );
  VHSR_AOI31_2 U334 ( .A1(b[6]), .A2(a[6]), .A3(n290), .B(n289), .ZN(n361) );
  VHSR_OAI21_2 U335 ( .A1(n403), .A2(n290), .B(n361), .ZN(n307) );
  VHSR_NOR3_2 U336 ( .A1(n292), .A2(n291), .A3(n326), .ZN(n368) );
  VHSR_AOI22_2 U337 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n293) );
  VHSR_NOR2_1 U338 ( .A1(n368), .A2(n293), .ZN(n303) );
  VHSR_NOR4_2 U339 ( .A1(n328), .A2(n324), .A3(n295), .A4(n294), .ZN(n366) );
  VHSR_AOI22_2 U340 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n296) );
  VHSR_NOR2_1 U341 ( .A1(n366), .A2(n296), .ZN(n301) );
  VHSR_IN_2 U342 ( .I(n297), .ZN(n309) );
  VHSR_NOR2_1 U343 ( .A1(n388), .A2(n298), .ZN(n315) );
  VHSR_AOI22_2 U344 ( .A1(n300), .A2(n299), .B1(n315), .B2(n314), .ZN(n313) );
  VHSR_AD1_1 U345 ( .A(n303), .B(n302), .CI(n301), .CO(n304), .S(n297) );
  VHSR_NOR2_1 U346 ( .A1(n308), .A2(n304), .ZN(n306) );
  VHSR_CLKNAND2_2 U347 ( .A1(n308), .A2(n304), .ZN(n305) );
  VHSR_NOR2_1 U348 ( .A1(n306), .A2(n307), .ZN(n360) );
  VHSR_AOI22_2 U349 ( .A1(n307), .A2(n306), .B1(n305), .B2(n360), .ZN(n401) );
  VHSR_AOI21_2 U350 ( .A1(n313), .A2(n309), .B(n308), .ZN(n380) );
  VHSR_AD1_1 U351 ( .A(n312), .B(n311), .CI(n310), .CO(n402), .S(n379) );
  VHSR_OAI21_2 U352 ( .A1(n315), .A2(n314), .B(n313), .ZN(n316) );
  VHSR_IN_2 U353 ( .I(n316), .ZN(n383) );
  VHSR_AD1_1 U354 ( .A(n319), .B(n318), .CI(n317), .CO(n310), .S(n382) );
  VHSR_AD1_1 U355 ( .A(n322), .B(n321), .CI(n320), .CO(n317), .S(n386) );
  VHSR_NOR2_1 U356 ( .A1(n324), .A2(n323), .ZN(n327) );
  VHSR_OAI21_2 U357 ( .A1(n328), .A2(n326), .B(n327), .ZN(n325) );
  VHSR_OAI31_2 U358 ( .A1(n328), .A2(n327), .A3(n326), .B(n325), .ZN(n385) );
  VHSR_AD1_1 U359 ( .A(n331), .B(n330), .CI(n329), .CO(n320), .S(n389) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[2]), .A2(a[2]), .ZN(n341) );
  VHSR_IN_2 U361 ( .I(n341), .ZN(n348) );
  VHSR_CLKNAND2_2 U362 ( .A1(b[2]), .A2(a[0]), .ZN(n421) );
  VHSR_NOR3_2 U363 ( .A1(n335), .A2(n414), .A3(n421), .ZN(n356) );
  VHSR_AOI22_2 U364 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n332) );
  VHSR_NOR2_1 U365 ( .A1(n356), .A2(n332), .ZN(n399) );
  VHSR_CLKNAND2_2 U366 ( .A1(b[1]), .A2(a[1]), .ZN(n420) );
  VHSR_CLKNAND2_2 U367 ( .A1(b[0]), .A2(a[2]), .ZN(n419) );
  VHSR_MAOI222_2 U368 ( .A(n421), .B(n420), .C(n419), .ZN(n418) );
  VHSR_IN_2 U369 ( .I(n333), .ZN(n358) );
  VHSR_NOR2_1 U370 ( .A1(n417), .A2(n334), .ZN(n336) );
  VHSR_AOI211_2 U371 ( .A1(a[0]), .A2(b[2]), .B(n335), .C(n414), .ZN(n337) );
  VHSR_MAOI222_2 U372 ( .A(n336), .B(n337), .C(n348), .ZN(n339) );
  VHSR_OR2_2 U373 ( .A1(n337), .A2(n348), .Z(n338) );
  VHSR_AOI32_2 U374 ( .A1(a[3]), .A2(n339), .A3(b[1]), .B1(n338), .B2(n339), 
        .ZN(n357) );
  VHSR_OAI21_2 U375 ( .A1(n358), .A2(n357), .B(n339), .ZN(n355) );
  VHSR_CLKNAND2_2 U376 ( .A1(b[3]), .A2(a[3]), .ZN(n346) );
  VHSR_AOI22_2 U377 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n340) );
  VHSR_IAO21_2 U378 ( .A1(n346), .A2(n341), .B(n340), .ZN(n354) );
  VHSR_IAO21_2 U379 ( .A1(n348), .A2(n347), .B(n346), .ZN(n392) );
  VHSR_AD1_1 U380 ( .A(n344), .B(n343), .CI(n342), .CO(n329), .S(n391) );
  VHSR_OAI21_2 U381 ( .A1(n348), .A2(n346), .B(n347), .ZN(n345) );
  VHSR_OAI31_2 U382 ( .A1(n348), .A2(n347), .A3(n346), .B(n345), .ZN(n395) );
  VHSR_AD1_1 U383 ( .A(n351), .B(n350), .CI(n349), .CO(n342), .S(n394) );
  VHSR_AD1_1 U384 ( .A(n353), .B(n352), .CI(n359), .CO(n349), .S(n397) );
  VHSR_AD1_1 U385 ( .A(n356), .B(n355), .CI(n354), .CO(n347), .S(n396) );
  VHSR_CLKNAND2_2 U386 ( .A1(a[4]), .A2(b[0]), .ZN(n410) );
  VHSR_XNOR2_2 U387 ( .A1(n358), .A2(n357), .ZN(n413) );
  VHSR_AOI211_2 U388 ( .A1(n410), .A2(n409), .B(n359), .C(n413), .ZN(n411) );
  VHSR_CLKNAND2_2 U389 ( .A1(a[6]), .A2(b[7]), .ZN(n363) );
  VHSR_AOI21_2 U390 ( .A1(a[7]), .A2(b[6]), .B(n363), .ZN(n362) );
  VHSR_AOI31_2 U391 ( .A1(a[7]), .A2(n363), .A3(b[6]), .B(n362), .ZN(n364) );
  VHSR_IN_2 U392 ( .I(n364), .ZN(n365) );
  VHSR_MAOI222_2 U393 ( .A(n368), .B(n366), .C(n365), .ZN(n375) );
  VHSR_OAI21_2 U394 ( .A1(n368), .A2(n367), .B(n375), .ZN(n372) );
  VHSR_CLKXOR2_2 U395 ( .A1(n373), .A2(n372), .Z(n369) );
  VHSR_CLKNAND2_2 U396 ( .A1(n370), .A2(n369), .ZN(n405) );
  VHSR_OAI21_2 U397 ( .A1(n370), .A2(n369), .B(n405), .ZN(n371) );
  VHSR_NOR2_1 U398 ( .A1(n373), .A2(n372), .ZN(n374) );
  VHSR_AND3_2 U399 ( .A1(n406), .A2(n376), .A3(n405), .Z(n377) );
  VHSR_NOR2_1 U400 ( .A1(n404), .A2(n377), .ZN(product[15]) );
  VHSR_AD1_1 U401 ( .A(n402), .B(n401), .CI(n400), .CO(n370), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U402 ( .A1(n404), .A2(n403), .ZN(n407) );
  VHSR_XOR3_2 U403 ( .A1(n407), .A2(n406), .A3(n405), .Z(product[14]) );
  VHSR_AOI21_2 U404 ( .A1(n413), .A2(n412), .B(n411), .ZN(product[4]) );
  VHSR_OAI22_2 U405 ( .A1(n417), .A2(n416), .B1(n415), .B2(n414), .ZN(
        product[1]) );
  VHSR_AOI31_2 U406 ( .A1(n421), .A2(n420), .A3(n419), .B(n418), .ZN(
        product[2]) );
endmodule


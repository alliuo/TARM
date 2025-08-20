
module mul8_78 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n226, n227, n228, n229, n230, n231, n232, n233,
         n234, n235, n236, n237, n238, n239, n240, n241, n242, n243, n244,
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
         n410, n411, n412, n413, n414, n415, n416, n417, n418, n419;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[11] = \intadd_0/SUM[6] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U215 ( .A1(n279), .B1(a[5]), .B2(b[3]), .ZN(n236) );
  VHSR_INOR2_2 U216 ( .A1(n245), .B1(n264), .ZN(n254) );
  VHSR_INOR2_2 U217 ( .A1(n243), .B1(n267), .ZN(n266) );
  VHSR_NOR2_1 U218 ( .A1(n294), .A2(n330), .ZN(n251) );
  VHSR_NOR2_1 U219 ( .A1(n247), .A2(n246), .ZN(n306) );
  VHSR_NOR2_1 U220 ( .A1(n295), .A2(n353), .ZN(n387) );
  VHSR_IN_2 U221 ( .I(n373), .ZN(product[13]) );
  VHSR_INOR2_1 U222 ( .A1(n377), .B1(n376), .ZN(n408) );
  VHSR_INAND2_1 U223 ( .A1(n232), .B1(n231), .ZN(n233) );
  VHSR_AD1_1 U224 ( .A(n388), .B(n387), .CI(n386), .CO(n383), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n392), .B(n391), .CI(n417), .CO(n352), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n410), .B(n390), .CI(n389), .CO(n393), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n385), .B(n384), .CI(n383), .CO(n399), .S(product[9])
         );
  VHSR_AD1_1 U228 ( .A(n382), .B(n381), .CI(n380), .CO(n402), .S(
        \intadd_0/SUM[6] ) );
  VHSR_IN_2 U229 ( .I(b[6]), .ZN(n294) );
  VHSR_IN_2 U230 ( .I(a[2]), .ZN(n330) );
  VHSR_IN_2 U231 ( .I(b[5]), .ZN(n292) );
  VHSR_CLKNAND2_2 U232 ( .A1(b[4]), .A2(a[2]), .ZN(n275) );
  VHSR_IN_2 U233 ( .I(a[3]), .ZN(n332) );
  VHSR_NOR3_2 U234 ( .A1(n292), .A2(n275), .A3(n332), .ZN(n259) );
  VHSR_CLKNAND2_2 U235 ( .A1(b[7]), .A2(a[3]), .ZN(n249) );
  VHSR_IN_2 U236 ( .I(n251), .ZN(n231) );
  VHSR_AOI22_2 U237 ( .A1(b[6]), .A2(a[3]), .B1(b[7]), .B2(a[2]), .ZN(n226) );
  VHSR_IAO21_2 U238 ( .A1(n249), .A2(n231), .B(n226), .ZN(n258) );
  VHSR_AOI22_2 U239 ( .A1(b[4]), .A2(a[3]), .B1(b[5]), .B2(a[2]), .ZN(n227) );
  VHSR_NOR2_1 U240 ( .A1(n227), .A2(n259), .ZN(n230) );
  VHSR_IN_2 U241 ( .I(a[1]), .ZN(n413) );
  VHSR_IN_2 U242 ( .I(b[7]), .ZN(n293) );
  VHSR_IN_2 U243 ( .I(a[0]), .ZN(n415) );
  VHSR_OAI22_2 U244 ( .A1(n294), .A2(n413), .B1(n293), .B2(n415), .ZN(n229) );
  VHSR_IN_2 U245 ( .I(b[4]), .ZN(n353) );
  VHSR_NOR4_2 U246 ( .A1(n353), .A2(n292), .A3(n413), .A4(n415), .ZN(n282) );
  VHSR_IN_2 U247 ( .I(n261), .ZN(n235) );
  VHSR_NOR2_1 U248 ( .A1(n293), .A2(n413), .ZN(n228) );
  VHSR_AOI211_2 U249 ( .A1(b[4]), .A2(a[2]), .B(n292), .C(n332), .ZN(n232) );
  VHSR_MAOI222_2 U250 ( .A(n228), .B(n251), .C(n232), .ZN(n234) );
  VHSR_OAI211_2 U251 ( .A1(n353), .A2(n415), .B(b[5]), .C(a[1]), .ZN(n274) );
  VHSR_CLKNAND2_2 U252 ( .A1(b[6]), .A2(a[0]), .ZN(n273) );
  VHSR_MAOI222_2 U253 ( .A(n275), .B(n274), .C(n273), .ZN(n272) );
  VHSR_AD1_1 U254 ( .A(n230), .B(n229), .CI(n282), .CO(n261), .S(n270) );
  VHSR_CLKNAND2_2 U255 ( .A1(n272), .A2(n270), .ZN(n269) );
  VHSR_AOI32_2 U256 ( .A1(a[1]), .A2(n234), .A3(b[7]), .B1(n233), .B2(n234), 
        .ZN(n260) );
  VHSR_AOI32_2 U257 ( .A1(n235), .A2(n234), .A3(n269), .B1(n260), .B2(n234), 
        .ZN(n257) );
  VHSR_IAO21_2 U258 ( .A1(n251), .A2(n250), .B(n249), .ZN(n307) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[3]), .A2(a[7]), .ZN(n247) );
  VHSR_IN_2 U260 ( .I(b[3]), .ZN(n331) );
  VHSR_IN_2 U261 ( .I(a[6]), .ZN(n285) );
  VHSR_IN_2 U262 ( .I(a[7]), .ZN(n290) );
  VHSR_IN_2 U263 ( .I(b[2]), .ZN(n327) );
  VHSR_OAI22_2 U264 ( .A1(n331), .A2(n285), .B1(n290), .B2(n327), .ZN(n256) );
  VHSR_IN_2 U265 ( .I(b[1]), .ZN(n416) );
  VHSR_IN_2 U266 ( .I(a[4]), .ZN(n295) );
  VHSR_NOR2_1 U267 ( .A1(n327), .A2(n295), .ZN(n279) );
  VHSR_OAI21_2 U268 ( .A1(n416), .A2(n290), .B(n236), .ZN(n244) );
  VHSR_IN_2 U269 ( .I(a[5]), .ZN(n296) );
  VHSR_NOR4_2 U270 ( .A1(n279), .A2(n296), .A3(n247), .A4(n416), .ZN(n237) );
  VHSR_AOI31_2 U271 ( .A1(b[2]), .A2(a[6]), .A3(n244), .B(n237), .ZN(n245) );
  VHSR_IN_2 U272 ( .I(b[0]), .ZN(n414) );
  VHSR_NOR4_2 U273 ( .A1(n296), .A2(n295), .A3(n416), .A4(n414), .ZN(n284) );
  VHSR_NAND4_2 U274 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n253) );
  VHSR_NOR2_1 U275 ( .A1(n327), .A2(n296), .ZN(n238) );
  VHSR_AOI32_2 U276 ( .A1(b[3]), .A2(n253), .A3(a[4]), .B1(n238), .B2(n253), 
        .ZN(n239) );
  VHSR_IN_2 U277 ( .I(n239), .ZN(n240) );
  VHSR_OAI22_2 U278 ( .A1(n290), .A2(n414), .B1(n285), .B2(n416), .ZN(n241) );
  VHSR_MAOI222_2 U279 ( .A(n284), .B(n240), .C(n241), .ZN(n243) );
  VHSR_NOR2_1 U280 ( .A1(n285), .A2(n414), .ZN(n278) );
  VHSR_AOI211_2 U281 ( .A1(a[4]), .A2(b[0]), .B(n296), .C(n416), .ZN(n277) );
  VHSR_MAOI222_2 U282 ( .A(n279), .B(n278), .C(n277), .ZN(n276) );
  VHSR_OR2_2 U283 ( .A1(n284), .A2(n240), .Z(n242) );
  VHSR_OAI21_2 U284 ( .A1(n242), .A2(n241), .B(n243), .ZN(n268) );
  VHSR_NOR2_1 U285 ( .A1(n276), .A2(n268), .ZN(n267) );
  VHSR_AOI32_2 U286 ( .A1(b[2]), .A2(n245), .A3(a[6]), .B1(n244), .B2(n245), 
        .ZN(n265) );
  VHSR_NOR2_1 U287 ( .A1(n266), .A2(n265), .ZN(n264) );
  VHSR_CLKNAND2_2 U288 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U289 ( .A1(n256), .A2(n252), .ZN(n246) );
  VHSR_AOI21_2 U290 ( .A1(n247), .A2(n246), .B(n306), .ZN(n312) );
  VHSR_OAI21_2 U291 ( .A1(n251), .A2(n249), .B(n250), .ZN(n248) );
  VHSR_OAI31_2 U292 ( .A1(n251), .A2(n250), .A3(n249), .B(n248), .ZN(n311) );
  VHSR_OAI21_2 U293 ( .A1(n254), .A2(n253), .B(n252), .ZN(n255) );
  VHSR_XNOR2_2 U294 ( .A1(n256), .A2(n255), .ZN(n319) );
  VHSR_AD1_1 U295 ( .A(n259), .B(n258), .CI(n257), .CO(n250), .S(n318) );
  VHSR_NOR2_1 U296 ( .A1(n261), .A2(n260), .ZN(n263) );
  VHSR_AOI22_2 U297 ( .A1(n261), .A2(n260), .B1(n269), .B2(n263), .ZN(n262) );
  VHSR_OAI21_2 U298 ( .A1(n269), .A2(n263), .B(n262), .ZN(n324) );
  VHSR_AOI21_2 U299 ( .A1(n266), .A2(n265), .B(n264), .ZN(n323) );
  VHSR_AOI21_2 U300 ( .A1(n276), .A2(n268), .B(n267), .ZN(n343) );
  VHSR_OAI21_2 U301 ( .A1(n272), .A2(n270), .B(n269), .ZN(n271) );
  VHSR_IN_2 U302 ( .I(n271), .ZN(n342) );
  VHSR_AOI31_2 U303 ( .A1(n275), .A2(n274), .A3(n273), .B(n272), .ZN(n350) );
  VHSR_OAI31_2 U304 ( .A1(n279), .A2(n278), .A3(n277), .B(n276), .ZN(n280) );
  VHSR_IN_2 U305 ( .I(n280), .ZN(n349) );
  VHSR_CLKNAND2_2 U306 ( .A1(b[5]), .A2(a[0]), .ZN(n281) );
  VHSR_OAI32_2 U307 ( .A1(n282), .A2(n413), .A3(n353), .B1(n281), .B2(n282), 
        .ZN(n358) );
  VHSR_NOR2_1 U308 ( .A1(n414), .A2(n415), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U309 ( .A1(n387), .A2(product[0]), .ZN(n355) );
  VHSR_IN_2 U310 ( .I(n355), .ZN(n357) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[4]), .A2(b[1]), .ZN(n283) );
  VHSR_OAI32_2 U312 ( .A1(n284), .A2(n414), .A3(n296), .B1(n283), .B2(n284), 
        .ZN(n356) );
  VHSR_NOR2_1 U313 ( .A1(n285), .A2(n294), .ZN(n405) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[6]), .A2(b[4]), .ZN(n316) );
  VHSR_NAND3_2 U315 ( .A1(a[7]), .A2(b[5]), .A3(n316), .ZN(n287) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[4]), .A2(b[6]), .ZN(n315) );
  VHSR_NAND3_2 U317 ( .A1(b[7]), .A2(a[5]), .A3(n315), .ZN(n286) );
  VHSR_CLKNAND2_2 U318 ( .A1(n287), .A2(n286), .ZN(n289) );
  VHSR_IN_2 U319 ( .I(n405), .ZN(n378) );
  VHSR_MAOI222_2 U320 ( .A(n378), .B(n287), .C(n286), .ZN(n362) );
  VHSR_IN_2 U321 ( .I(n362), .ZN(n288) );
  VHSR_OAI21_2 U322 ( .A1(n405), .A2(n289), .B(n288), .ZN(n304) );
  VHSR_NOR3_2 U323 ( .A1(n290), .A2(n316), .A3(n292), .ZN(n370) );
  VHSR_AOI22_2 U324 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n291) );
  VHSR_NOR2_1 U325 ( .A1(n370), .A2(n291), .ZN(n300) );
  VHSR_IN_2 U326 ( .I(n387), .ZN(n298) );
  VHSR_NOR3_2 U327 ( .A1(n296), .A2(n292), .A3(n298), .ZN(n321) );
  VHSR_NOR4_2 U328 ( .A1(n296), .A2(n295), .A3(n294), .A4(n293), .ZN(n368) );
  VHSR_AOI22_2 U329 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n297) );
  VHSR_NOR2_1 U330 ( .A1(n368), .A2(n297), .ZN(n299) );
  VHSR_NAND3_2 U331 ( .A1(b[5]), .A2(a[5]), .A3(n298), .ZN(n314) );
  VHSR_MAOI222_2 U332 ( .A(n316), .B(n315), .C(n314), .ZN(n313) );
  VHSR_AND2_2 U333 ( .A1(n309), .A2(n313), .Z(n308) );
  VHSR_AD1_1 U334 ( .A(n300), .B(n321), .CI(n299), .CO(n301), .S(n309) );
  VHSR_NOR2_1 U335 ( .A1(n308), .A2(n301), .ZN(n303) );
  VHSR_CLKNAND2_2 U336 ( .A1(n308), .A2(n301), .ZN(n302) );
  VHSR_NOR2_1 U337 ( .A1(n303), .A2(n304), .ZN(n363) );
  VHSR_AOI22_2 U338 ( .A1(n304), .A2(n303), .B1(n302), .B2(n363), .ZN(n403) );
  VHSR_AD1_1 U339 ( .A(n307), .B(n306), .CI(n305), .CO(n404), .S(n382) );
  VHSR_IAO21_2 U340 ( .A1(n309), .A2(n313), .B(n308), .ZN(n381) );
  VHSR_AD1_1 U341 ( .A(n312), .B(n311), .CI(n310), .CO(n305), .S(n401) );
  VHSR_AOI31_2 U342 ( .A1(n316), .A2(n315), .A3(n314), .B(n313), .ZN(n400) );
  VHSR_AD1_1 U343 ( .A(n319), .B(n318), .CI(n317), .CO(n310), .S(n385) );
  VHSR_AOI22_2 U344 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n320) );
  VHSR_NOR2_1 U345 ( .A1(n321), .A2(n320), .ZN(n384) );
  VHSR_AD1_1 U346 ( .A(n324), .B(n323), .CI(n322), .CO(n317), .S(n388) );
  VHSR_NOR2_1 U347 ( .A1(n327), .A2(n330), .ZN(n347) );
  VHSR_NOR2_1 U348 ( .A1(n327), .A2(n332), .ZN(n326) );
  VHSR_OAI21_2 U349 ( .A1(n331), .A2(n330), .B(n326), .ZN(n325) );
  VHSR_OAI31_2 U350 ( .A1(n331), .A2(n326), .A3(n330), .B(n325), .ZN(n361) );
  VHSR_NOR2_1 U351 ( .A1(n327), .A2(n415), .ZN(n339) );
  VHSR_AOI22_2 U352 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n328) );
  VHSR_AOI31_2 U353 ( .A1(a[1]), .A2(b[3]), .A3(n339), .B(n328), .ZN(n392) );
  VHSR_NOR4_2 U354 ( .A1(n416), .A2(n414), .A3(n330), .A4(n332), .ZN(n338) );
  VHSR_CLKNAND2_2 U355 ( .A1(b[0]), .A2(a[3]), .ZN(n329) );
  VHSR_OAI32_2 U356 ( .A1(n338), .A2(n330), .A3(n416), .B1(n329), .B2(n338), 
        .ZN(n391) );
  VHSR_AOI22_2 U357 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n419) );
  VHSR_IN_2 U358 ( .I(n339), .ZN(n418) );
  VHSR_NOR2_1 U359 ( .A1(n419), .A2(n418), .ZN(n417) );
  VHSR_IN_2 U360 ( .I(n352), .ZN(n337) );
  VHSR_NOR3_2 U361 ( .A1(n339), .A2(n413), .A3(n331), .ZN(n335) );
  VHSR_AOI211_2 U362 ( .A1(a[2]), .A2(b[0]), .B(n416), .C(n332), .ZN(n333) );
  VHSR_OR2_2 U363 ( .A1(n347), .A2(n333), .Z(n334) );
  VHSR_MAOI222_2 U364 ( .A(n335), .B(n347), .C(n333), .ZN(n336) );
  VHSR_OAI21_2 U365 ( .A1(n335), .A2(n334), .B(n336), .ZN(n351) );
  VHSR_OAI21_2 U366 ( .A1(n337), .A2(n351), .B(n336), .ZN(n360) );
  VHSR_AOI31_2 U367 ( .A1(a[1]), .A2(b[3]), .A3(n339), .B(n338), .ZN(n340) );
  VHSR_IN_2 U368 ( .I(n340), .ZN(n359) );
  VHSR_CLKNAND2_2 U369 ( .A1(b[3]), .A2(a[3]), .ZN(n345) );
  VHSR_IAO21_2 U370 ( .A1(n347), .A2(n346), .B(n345), .ZN(n398) );
  VHSR_AD1_1 U371 ( .A(n343), .B(n342), .CI(n341), .CO(n322), .S(n397) );
  VHSR_OAI21_2 U372 ( .A1(n347), .A2(n345), .B(n346), .ZN(n344) );
  VHSR_OAI31_2 U373 ( .A1(n347), .A2(n346), .A3(n345), .B(n344), .ZN(n395) );
  VHSR_AD1_1 U374 ( .A(n350), .B(n349), .CI(n348), .CO(n341), .S(n394) );
  VHSR_CLKXOR2_2 U375 ( .A1(n352), .A2(n351), .Z(n412) );
  VHSR_NOR2_1 U376 ( .A1(n353), .A2(n415), .ZN(n354) );
  VHSR_AOI32_2 U377 ( .A1(b[0]), .A2(n355), .A3(a[4]), .B1(n354), .B2(n355), 
        .ZN(n411) );
  VHSR_NOR2_1 U378 ( .A1(n412), .A2(n411), .ZN(n410) );
  VHSR_AD1_1 U379 ( .A(n358), .B(n357), .CI(n356), .CO(n348), .S(n390) );
  VHSR_AD1_1 U380 ( .A(n361), .B(n360), .CI(n359), .CO(n346), .S(n389) );
  VHSR_NOR2_1 U381 ( .A1(n363), .A2(n362), .ZN(n375) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[6]), .A2(b[7]), .ZN(n365) );
  VHSR_AOI21_2 U383 ( .A1(a[7]), .A2(b[6]), .B(n365), .ZN(n364) );
  VHSR_AOI31_2 U384 ( .A1(a[7]), .A2(n365), .A3(b[6]), .B(n364), .ZN(n366) );
  VHSR_IN_2 U385 ( .I(n366), .ZN(n367) );
  VHSR_OR2_2 U386 ( .A1(n368), .A2(n367), .Z(n369) );
  VHSR_MAOI222_2 U387 ( .A(n370), .B(n368), .C(n367), .ZN(n377) );
  VHSR_OAI21_2 U388 ( .A1(n370), .A2(n369), .B(n377), .ZN(n374) );
  VHSR_CLKXOR2_2 U389 ( .A1(n375), .A2(n374), .Z(n371) );
  VHSR_CLKNAND2_2 U390 ( .A1(n372), .A2(n371), .ZN(n407) );
  VHSR_OAI21_2 U391 ( .A1(n372), .A2(n371), .B(n407), .ZN(n373) );
  VHSR_CLKNAND2_2 U392 ( .A1(a[7]), .A2(b[7]), .ZN(n406) );
  VHSR_NOR2_1 U393 ( .A1(n375), .A2(n374), .ZN(n376) );
  VHSR_AND3_2 U394 ( .A1(n408), .A2(n378), .A3(n407), .Z(n379) );
  VHSR_NOR2_1 U395 ( .A1(n406), .A2(n379), .ZN(product[15]) );
  VHSR_AD1_1 U396 ( .A(n395), .B(n394), .CI(n393), .CO(n396), .S(product[6])
         );
  VHSR_AD1_1 U397 ( .A(n398), .B(n397), .CI(n396), .CO(n386), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U398 ( .A(n401), .B(n400), .CI(n399), .CO(n380), .S(product[10])
         );
  VHSR_AD1_1 U399 ( .A(n404), .B(n403), .CI(n402), .CO(n372), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U400 ( .A1(n406), .A2(n405), .ZN(n409) );
  VHSR_XOR3_2 U401 ( .A1(n409), .A2(n408), .A3(n407), .Z(product[14]) );
  VHSR_AOI21_2 U402 ( .A1(n412), .A2(n411), .B(n410), .ZN(product[4]) );
  VHSR_OAI22_2 U403 ( .A1(n416), .A2(n415), .B1(n414), .B2(n413), .ZN(
        product[1]) );
  VHSR_AOI21_2 U404 ( .A1(n419), .A2(n418), .B(n417), .ZN(product[2]) );
endmodule


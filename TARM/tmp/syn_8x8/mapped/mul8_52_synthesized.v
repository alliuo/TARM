
module mul8_52 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n221, n222,
         n223, n224, n225, n226, n227, n228, n229, n230, n231, n232, n233,
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
         n410, n411;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U211 ( .A1(n270), .B1(a[5]), .B2(b[3]), .ZN(n221) );
  VHSR_INOR2_2 U212 ( .A1(n230), .B1(n256), .ZN(n249) );
  VHSR_INOR2_2 U213 ( .A1(n325), .B1(n339), .ZN(n328) );
  VHSR_INOR2_2 U214 ( .A1(n228), .B1(n259), .ZN(n258) );
  VHSR_NOR2_1 U215 ( .A1(n240), .A2(n239), .ZN(n300) );
  VHSR_NOR2_1 U216 ( .A1(n286), .A2(n345), .ZN(n379) );
  VHSR_IN_2 U217 ( .I(n365), .ZN(product[13]) );
  VHSR_INOR2_1 U218 ( .A1(n369), .B1(n368), .ZN(n400) );
  VHSR_AD1_1 U219 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(product[9])
         );
  VHSR_AD1_1 U220 ( .A(n384), .B(n383), .CI(n409), .CO(n344), .S(product[3])
         );
  VHSR_AD1_1 U221 ( .A(n402), .B(n382), .CI(n381), .CO(n385), .S(product[5])
         );
  VHSR_AD1_1 U222 ( .A(n380), .B(n379), .CI(n378), .CO(n375), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U223 ( .A(n374), .B(n373), .CI(n372), .CO(n391), .S(product[10])
         );
  VHSR_CLKNAND2_2 U224 ( .A1(b[3]), .A2(a[7]), .ZN(n240) );
  VHSR_IN_2 U225 ( .I(b[3]), .ZN(n332) );
  VHSR_IN_2 U226 ( .I(a[6]), .ZN(n276) );
  VHSR_IN_2 U227 ( .I(a[7]), .ZN(n282) );
  VHSR_IN_2 U228 ( .I(b[2]), .ZN(n317) );
  VHSR_OAI22_2 U229 ( .A1(n332), .A2(n276), .B1(n282), .B2(n317), .ZN(n251) );
  VHSR_IN_2 U230 ( .I(b[1]), .ZN(n408) );
  VHSR_IN_2 U231 ( .I(a[4]), .ZN(n286) );
  VHSR_NOR2_1 U232 ( .A1(n317), .A2(n286), .ZN(n270) );
  VHSR_OAI21_2 U233 ( .A1(n408), .A2(n282), .B(n221), .ZN(n229) );
  VHSR_IN_2 U234 ( .I(a[5]), .ZN(n287) );
  VHSR_NOR4_2 U235 ( .A1(n270), .A2(n287), .A3(n240), .A4(n408), .ZN(n222) );
  VHSR_AOI31_2 U236 ( .A1(b[2]), .A2(a[6]), .A3(n229), .B(n222), .ZN(n230) );
  VHSR_IN_2 U237 ( .I(b[0]), .ZN(n406) );
  VHSR_NOR4_2 U238 ( .A1(n287), .A2(n286), .A3(n408), .A4(n406), .ZN(n275) );
  VHSR_NAND4_2 U239 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n248) );
  VHSR_NOR2_1 U240 ( .A1(n317), .A2(n287), .ZN(n223) );
  VHSR_AOI32_2 U241 ( .A1(b[3]), .A2(n248), .A3(a[4]), .B1(n223), .B2(n248), 
        .ZN(n224) );
  VHSR_IN_2 U242 ( .I(n224), .ZN(n225) );
  VHSR_OAI22_2 U243 ( .A1(n282), .A2(n406), .B1(n276), .B2(n408), .ZN(n226) );
  VHSR_MAOI222_2 U244 ( .A(n275), .B(n225), .C(n226), .ZN(n228) );
  VHSR_NOR2_1 U245 ( .A1(n276), .A2(n406), .ZN(n269) );
  VHSR_AOI211_2 U246 ( .A1(a[4]), .A2(b[0]), .B(n287), .C(n408), .ZN(n268) );
  VHSR_MAOI222_2 U247 ( .A(n270), .B(n269), .C(n268), .ZN(n267) );
  VHSR_OR2_2 U248 ( .A1(n275), .A2(n225), .Z(n227) );
  VHSR_OAI21_2 U249 ( .A1(n227), .A2(n226), .B(n228), .ZN(n260) );
  VHSR_NOR2_1 U250 ( .A1(n267), .A2(n260), .ZN(n259) );
  VHSR_AOI32_2 U251 ( .A1(b[2]), .A2(n230), .A3(a[6]), .B1(n229), .B2(n230), 
        .ZN(n257) );
  VHSR_NOR2_1 U252 ( .A1(n258), .A2(n257), .ZN(n256) );
  VHSR_CLKNAND2_2 U253 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U254 ( .A1(n251), .A2(n247), .ZN(n239) );
  VHSR_IN_2 U255 ( .I(b[7]), .ZN(n284) );
  VHSR_IN_2 U256 ( .I(a[3]), .ZN(n316) );
  VHSR_IN_2 U257 ( .I(b[6]), .ZN(n285) );
  VHSR_IN_2 U258 ( .I(a[2]), .ZN(n324) );
  VHSR_OAI22_2 U259 ( .A1(n285), .A2(n316), .B1(n284), .B2(n324), .ZN(n246) );
  VHSR_AOI22_2 U260 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n237) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[4]), .A2(a[2]), .ZN(n266) );
  VHSR_NAND3_2 U262 ( .A1(a[3]), .A2(b[5]), .A3(n266), .ZN(n236) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[7]), .A2(a[2]), .ZN(n231) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[6]), .A2(a[1]), .ZN(n233) );
  VHSR_OAI22_2 U265 ( .A1(n237), .A2(n236), .B1(n231), .B2(n233), .ZN(n238) );
  VHSR_IN_2 U266 ( .I(b[4]), .ZN(n345) );
  VHSR_IN_2 U267 ( .I(a[0]), .ZN(n407) );
  VHSR_OAI211_2 U268 ( .A1(n345), .A2(n407), .B(b[5]), .C(a[1]), .ZN(n265) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[0]), .ZN(n264) );
  VHSR_MAOI222_2 U270 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_NAND4_2 U271 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n243) );
  VHSR_IN_2 U272 ( .I(b[5]), .ZN(n281) );
  VHSR_OAI22_2 U273 ( .A1(n345), .A2(n316), .B1(n281), .B2(n324), .ZN(n232) );
  VHSR_AND2_2 U274 ( .A1(n243), .A2(n232), .Z(n235) );
  VHSR_OAI21_2 U275 ( .A1(n284), .A2(n407), .B(n233), .ZN(n234) );
  VHSR_IN_2 U276 ( .I(a[1]), .ZN(n405) );
  VHSR_NOR4_2 U277 ( .A1(n345), .A2(n281), .A3(n405), .A4(n407), .ZN(n273) );
  VHSR_AND2_2 U278 ( .A1(n263), .A2(n262), .Z(n261) );
  VHSR_AD1_1 U279 ( .A(n235), .B(n234), .CI(n273), .CO(n252), .S(n262) );
  VHSR_AOI21_2 U280 ( .A1(n237), .A2(n236), .B(n238), .ZN(n255) );
  VHSR_OAI32_2 U281 ( .A1(n238), .A2(n261), .A3(n252), .B1(n255), .B2(n238), 
        .ZN(n244) );
  VHSR_CLKNAND2_2 U282 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_CLKNAND2_2 U283 ( .A1(n246), .A2(n242), .ZN(n241) );
  VHSR_NOR3_2 U284 ( .A1(n284), .A2(n316), .A3(n241), .ZN(n299) );
  VHSR_AOI21_2 U285 ( .A1(n240), .A2(n239), .B(n300), .ZN(n303) );
  VHSR_OAI32_2 U286 ( .A1(n299), .A2(n316), .A3(n284), .B1(n241), .B2(n299), 
        .ZN(n302) );
  VHSR_OAI21_2 U287 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U288 ( .A1(n246), .A2(n245), .ZN(n310) );
  VHSR_OAI21_2 U289 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_XNOR2_2 U290 ( .A1(n251), .A2(n250), .ZN(n309) );
  VHSR_NOR2_1 U291 ( .A1(n261), .A2(n252), .ZN(n254) );
  VHSR_AOI22_2 U292 ( .A1(n261), .A2(n252), .B1(n255), .B2(n254), .ZN(n253) );
  VHSR_OAI21_2 U293 ( .A1(n255), .A2(n254), .B(n253), .ZN(n315) );
  VHSR_AOI21_2 U294 ( .A1(n258), .A2(n257), .B(n256), .ZN(n314) );
  VHSR_AOI21_2 U295 ( .A1(n267), .A2(n260), .B(n259), .ZN(n335) );
  VHSR_IAO21_2 U296 ( .A1(n263), .A2(n262), .B(n261), .ZN(n334) );
  VHSR_AOI31_2 U297 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n342) );
  VHSR_OAI31_2 U298 ( .A1(n270), .A2(n269), .A3(n268), .B(n267), .ZN(n271) );
  VHSR_IN_2 U299 ( .I(n271), .ZN(n341) );
  VHSR_CLKNAND2_2 U300 ( .A1(b[5]), .A2(a[0]), .ZN(n272) );
  VHSR_OAI32_2 U301 ( .A1(n273), .A2(n405), .A3(n345), .B1(n272), .B2(n273), 
        .ZN(n350) );
  VHSR_NOR2_1 U302 ( .A1(n406), .A2(n407), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U303 ( .A1(n379), .A2(product[0]), .ZN(n347) );
  VHSR_IN_2 U304 ( .I(n347), .ZN(n349) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[1]), .ZN(n274) );
  VHSR_OAI32_2 U306 ( .A1(n275), .A2(n406), .A3(n287), .B1(n274), .B2(n275), 
        .ZN(n348) );
  VHSR_NOR2_1 U307 ( .A1(n276), .A2(n285), .ZN(n397) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[6]), .A2(b[4]), .ZN(n307) );
  VHSR_NAND3_2 U309 ( .A1(a[7]), .A2(b[5]), .A3(n307), .ZN(n278) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[4]), .A2(b[6]), .ZN(n306) );
  VHSR_NAND3_2 U311 ( .A1(b[7]), .A2(a[5]), .A3(n306), .ZN(n277) );
  VHSR_CLKNAND2_2 U312 ( .A1(n278), .A2(n277), .ZN(n280) );
  VHSR_IN_2 U313 ( .I(n397), .ZN(n370) );
  VHSR_MAOI222_2 U314 ( .A(n370), .B(n278), .C(n277), .ZN(n354) );
  VHSR_IN_2 U315 ( .I(n354), .ZN(n279) );
  VHSR_OAI21_2 U316 ( .A1(n397), .A2(n280), .B(n279), .ZN(n295) );
  VHSR_IN_2 U317 ( .I(n379), .ZN(n289) );
  VHSR_NOR3_2 U318 ( .A1(n287), .A2(n281), .A3(n289), .ZN(n311) );
  VHSR_NOR3_2 U319 ( .A1(n282), .A2(n307), .A3(n281), .ZN(n362) );
  VHSR_AOI22_2 U320 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n283) );
  VHSR_NOR2_1 U321 ( .A1(n362), .A2(n283), .ZN(n291) );
  VHSR_NOR4_2 U322 ( .A1(n287), .A2(n286), .A3(n285), .A4(n284), .ZN(n360) );
  VHSR_AOI22_2 U323 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n288) );
  VHSR_NOR2_1 U324 ( .A1(n360), .A2(n288), .ZN(n290) );
  VHSR_NAND3_2 U325 ( .A1(b[5]), .A2(a[5]), .A3(n289), .ZN(n305) );
  VHSR_MAOI222_2 U326 ( .A(n307), .B(n306), .C(n305), .ZN(n304) );
  VHSR_AND2_2 U327 ( .A1(n297), .A2(n304), .Z(n296) );
  VHSR_AD1_1 U328 ( .A(n311), .B(n291), .CI(n290), .CO(n292), .S(n297) );
  VHSR_NOR2_1 U329 ( .A1(n296), .A2(n292), .ZN(n294) );
  VHSR_CLKNAND2_2 U330 ( .A1(n296), .A2(n292), .ZN(n293) );
  VHSR_NOR2_1 U331 ( .A1(n294), .A2(n295), .ZN(n355) );
  VHSR_AOI22_2 U332 ( .A1(n295), .A2(n294), .B1(n293), .B2(n355), .ZN(n395) );
  VHSR_IAO21_2 U333 ( .A1(n297), .A2(n304), .B(n296), .ZN(n393) );
  VHSR_AD1_1 U334 ( .A(n300), .B(n299), .CI(n298), .CO(n396), .S(n392) );
  VHSR_AD1_1 U335 ( .A(n303), .B(n302), .CI(n301), .CO(n298), .S(n374) );
  VHSR_AOI31_2 U336 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n373) );
  VHSR_AD1_1 U337 ( .A(n310), .B(n309), .CI(n308), .CO(n301), .S(n377) );
  VHSR_AOI22_2 U338 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n312) );
  VHSR_NOR2_1 U339 ( .A1(n312), .A2(n311), .ZN(n376) );
  VHSR_AD1_1 U340 ( .A(n315), .B(n314), .CI(n313), .CO(n308), .S(n380) );
  VHSR_NOR2_1 U341 ( .A1(n317), .A2(n324), .ZN(n339) );
  VHSR_NOR2_1 U342 ( .A1(n317), .A2(n316), .ZN(n319) );
  VHSR_OAI21_2 U343 ( .A1(n332), .A2(n324), .B(n319), .ZN(n318) );
  VHSR_OAI31_2 U344 ( .A1(n332), .A2(n319), .A3(n324), .B(n318), .ZN(n353) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[2]), .A2(a[0]), .ZN(n410) );
  VHSR_NOR3_2 U346 ( .A1(n332), .A2(n405), .A3(n410), .ZN(n321) );
  VHSR_AOI22_2 U347 ( .A1(b[3]), .A2(a[0]), .B1(b[2]), .B2(a[1]), .ZN(n320) );
  VHSR_NOR2_1 U348 ( .A1(n321), .A2(n320), .ZN(n384) );
  VHSR_NAND4_2 U349 ( .A1(b[1]), .A2(b[0]), .A3(a[2]), .A4(a[3]), .ZN(n331) );
  VHSR_IN_2 U350 ( .I(n331), .ZN(n323) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[0]), .A2(a[3]), .ZN(n322) );
  VHSR_OAI32_2 U352 ( .A1(n323), .A2(n324), .A3(n408), .B1(n322), .B2(n323), 
        .ZN(n383) );
  VHSR_AOI22_2 U353 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n411) );
  VHSR_NOR2_1 U354 ( .A1(n411), .A2(n410), .ZN(n409) );
  VHSR_OAI211_2 U355 ( .A1(n324), .A2(n406), .B(b[1]), .C(a[3]), .ZN(n325) );
  VHSR_NAND3_2 U356 ( .A1(a[1]), .A2(b[3]), .A3(n410), .ZN(n327) );
  VHSR_IN_2 U357 ( .I(n339), .ZN(n326) );
  VHSR_MAOI222_2 U358 ( .A(n327), .B(n326), .C(n325), .ZN(n329) );
  VHSR_AOI21_2 U359 ( .A1(n328), .A2(n327), .B(n329), .ZN(n343) );
  VHSR_AOI21_2 U360 ( .A1(n344), .A2(n343), .B(n329), .ZN(n330) );
  VHSR_IN_2 U361 ( .I(n330), .ZN(n352) );
  VHSR_OAI31_2 U362 ( .A1(n405), .A2(n332), .A3(n410), .B(n331), .ZN(n351) );
  VHSR_CLKNAND2_2 U363 ( .A1(b[3]), .A2(a[3]), .ZN(n337) );
  VHSR_IAO21_2 U364 ( .A1(n339), .A2(n338), .B(n337), .ZN(n390) );
  VHSR_AD1_1 U365 ( .A(n335), .B(n334), .CI(n333), .CO(n313), .S(n389) );
  VHSR_OAI21_2 U366 ( .A1(n339), .A2(n337), .B(n338), .ZN(n336) );
  VHSR_OAI31_2 U367 ( .A1(n339), .A2(n338), .A3(n337), .B(n336), .ZN(n387) );
  VHSR_AD1_1 U368 ( .A(n342), .B(n341), .CI(n340), .CO(n333), .S(n386) );
  VHSR_XNOR2_2 U369 ( .A1(n344), .A2(n343), .ZN(n404) );
  VHSR_NOR2_1 U370 ( .A1(n345), .A2(n407), .ZN(n346) );
  VHSR_AOI32_2 U371 ( .A1(b[0]), .A2(n347), .A3(a[4]), .B1(n346), .B2(n347), 
        .ZN(n403) );
  VHSR_NOR2_1 U372 ( .A1(n404), .A2(n403), .ZN(n402) );
  VHSR_AD1_1 U373 ( .A(n350), .B(n349), .CI(n348), .CO(n340), .S(n382) );
  VHSR_AD1_1 U374 ( .A(n353), .B(n352), .CI(n351), .CO(n338), .S(n381) );
  VHSR_NOR2_1 U375 ( .A1(n355), .A2(n354), .ZN(n367) );
  VHSR_CLKNAND2_2 U376 ( .A1(a[6]), .A2(b[7]), .ZN(n357) );
  VHSR_AOI21_2 U377 ( .A1(a[7]), .A2(b[6]), .B(n357), .ZN(n356) );
  VHSR_AOI31_2 U378 ( .A1(a[7]), .A2(n357), .A3(b[6]), .B(n356), .ZN(n358) );
  VHSR_IN_2 U379 ( .I(n358), .ZN(n359) );
  VHSR_OR2_2 U380 ( .A1(n360), .A2(n359), .Z(n361) );
  VHSR_MAOI222_2 U381 ( .A(n362), .B(n360), .C(n359), .ZN(n369) );
  VHSR_OAI21_2 U382 ( .A1(n362), .A2(n361), .B(n369), .ZN(n366) );
  VHSR_CLKXOR2_2 U383 ( .A1(n367), .A2(n366), .Z(n363) );
  VHSR_CLKNAND2_2 U384 ( .A1(n364), .A2(n363), .ZN(n399) );
  VHSR_OAI21_2 U385 ( .A1(n364), .A2(n363), .B(n399), .ZN(n365) );
  VHSR_CLKNAND2_2 U386 ( .A1(a[7]), .A2(b[7]), .ZN(n398) );
  VHSR_NOR2_1 U387 ( .A1(n367), .A2(n366), .ZN(n368) );
  VHSR_AND3_2 U388 ( .A1(n400), .A2(n370), .A3(n399), .Z(n371) );
  VHSR_NOR2_1 U389 ( .A1(n398), .A2(n371), .ZN(product[15]) );
  VHSR_AD1_1 U390 ( .A(n387), .B(n386), .CI(n385), .CO(n388), .S(product[6])
         );
  VHSR_AD1_1 U391 ( .A(n390), .B(n389), .CI(n388), .CO(n378), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U392 ( .A(n393), .B(n392), .CI(n391), .CO(n394), .S(product[11])
         );
  VHSR_AD1_1 U393 ( .A(n396), .B(n395), .CI(n394), .CO(n364), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U394 ( .A1(n398), .A2(n397), .ZN(n401) );
  VHSR_XOR3_2 U395 ( .A1(n401), .A2(n400), .A3(n399), .Z(product[14]) );
  VHSR_AOI21_2 U396 ( .A1(n404), .A2(n403), .B(n402), .ZN(product[4]) );
  VHSR_OAI22_2 U397 ( .A1(n408), .A2(n407), .B1(n406), .B2(n405), .ZN(
        product[1]) );
  VHSR_AOI21_2 U398 ( .A1(n411), .A2(n410), .B(n409), .ZN(product[2]) );
endmodule


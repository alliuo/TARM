
module mul8_96 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236, n237, n238,
         n239, n240, n241, n242, n243, n244, n245, n246, n247, n248, n249,
         n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, n260,
         n261, n262, n263, n264, n265, n266, n267, n268, n269, n270, n271,
         n272, n273, n274, n275, n276, n277, n278, n279, n280, n281, n282,
         n283, n284, n285, n286, n287, n288, n289, n290, n291, n292, n293,
         n294, n295, n296, n297, n298, n299, n300, n301, n302, n303, n304,
         n305, n306, n307, n308, n309, n310, n311, n312, n313, n314, n315,
         n316, n317, n318, n319, n320, n321, n322, n323, n324, n325, n326,
         n327, n328, n329, n330, n331, n332, n333, n334, n335, n336, n337,
         n338, n339, n340, n341, n342, n343, n344, n345, n346, n347, n348,
         n349, n350, n351, n352, n353, n354, n355, n356, n357, n358, n359,
         n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, n370,
         n371, n372, n373, n374, n375, n376, n377, n378, n379, n380, n381,
         n382, n383, n384, n385, n386, n387, n388, n389, n390, n391, n392,
         n393, n394, n395, n396, n397, n398, n399;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U206 ( .A1(n244), .B1(n217), .ZN(n221) );
  VHSR_NOR2_1 U207 ( .A1(n315), .A2(n277), .ZN(n266) );
  VHSR_INOR2_2 U208 ( .A1(n225), .B1(n252), .ZN(n245) );
  VHSR_NOR2_1 U209 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_NOR2_1 U210 ( .A1(n343), .A2(n342), .ZN(n355) );
  VHSR_NOR2_1 U211 ( .A1(n392), .A2(n391), .ZN(n390) );
  VHSR_INAND2_2 U212 ( .A1(n320), .B1(n339), .ZN(n335) );
  VHSR_INAND3_2 U213 ( .A1(n370), .B1(b[5]), .B2(a[5]), .ZN(n299) );
  VHSR_NOR2_1 U214 ( .A1(n235), .A2(n234), .ZN(n294) );
  VHSR_NOR2_1 U215 ( .A1(n277), .A2(n276), .ZN(n370) );
  VHSR_IN_2 U216 ( .I(n353), .ZN(product[13]) );
  VHSR_INOR2_1 U217 ( .A1(n357), .B1(n356), .ZN(n388) );
  VHSR_INOR2_1 U218 ( .A1(n223), .B1(n255), .ZN(n254) );
  VHSR_NOR2_2 U219 ( .A1(n291), .A2(n290), .ZN(n289) );
  VHSR_MOAI22_1 U220 ( .A1(n237), .A2(n317), .B1(b[6]), .B2(a[3]), .ZN(n242)
         );
  VHSR_AND4_1 U221 ( .A1(a[4]), .A2(a[5]), .A3(b[6]), .A4(b[7]), .Z(n348) );
  VHSR_AD1_1 U222 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(product[6])
         );
  VHSR_AD1_1 U223 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(product[10])
         );
  VHSR_AD1_1 U225 ( .A(n381), .B(n397), .CI(n380), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n379), .B(n378), .CI(n394), .CO(n375), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U228 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[9])
         );
  VHSR_AD1_1 U229 ( .A(n362), .B(n361), .CI(n360), .CO(n382), .S(product[11])
         );
  VHSR_IN_2 U230 ( .I(b[0]), .ZN(n316) );
  VHSR_IN_2 U231 ( .I(a[1]), .ZN(n312) );
  VHSR_NOR2_1 U232 ( .A1(n316), .A2(n312), .ZN(product[1]) );
  VHSR_CLKNAND2_2 U233 ( .A1(b[3]), .A2(a[7]), .ZN(n235) );
  VHSR_IN_2 U234 ( .I(b[3]), .ZN(n321) );
  VHSR_IN_2 U235 ( .I(a[6]), .ZN(n220) );
  VHSR_IN_2 U236 ( .I(a[7]), .ZN(n279) );
  VHSR_IN_2 U237 ( .I(b[2]), .ZN(n315) );
  VHSR_OAI22_2 U238 ( .A1(n321), .A2(n220), .B1(n279), .B2(n315), .ZN(n247) );
  VHSR_IN_2 U239 ( .I(a[4]), .ZN(n277) );
  VHSR_CLKNAND2_2 U240 ( .A1(b[3]), .A2(a[5]), .ZN(n215) );
  VHSR_IN_2 U241 ( .I(b[1]), .ZN(n318) );
  VHSR_OAI22_2 U242 ( .A1(n266), .A2(n215), .B1(n279), .B2(n318), .ZN(n224) );
  VHSR_CLKNAND2_2 U243 ( .A1(a[5]), .A2(b[1]), .ZN(n219) );
  VHSR_NOR3_2 U244 ( .A1(n266), .A2(n235), .A3(n219), .ZN(n216) );
  VHSR_AOI31_2 U245 ( .A1(a[6]), .A2(b[2]), .A3(n224), .B(n216), .ZN(n225) );
  VHSR_NOR2_1 U246 ( .A1(n220), .A2(n318), .ZN(n218) );
  VHSR_CLKNAND2_2 U247 ( .A1(a[4]), .A2(b[0]), .ZN(n392) );
  VHSR_NOR2_1 U248 ( .A1(n219), .A2(n392), .ZN(n271) );
  VHSR_NAND3_2 U249 ( .A1(b[3]), .A2(n266), .A3(a[5]), .ZN(n244) );
  VHSR_AOI22_2 U250 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n217) );
  VHSR_MAOI222_2 U251 ( .A(n218), .B(n271), .C(n221), .ZN(n223) );
  VHSR_AOI21_2 U252 ( .A1(a[4]), .A2(b[0]), .B(n219), .ZN(n265) );
  VHSR_AOI21_2 U253 ( .A1(n279), .A2(n220), .B(n316), .ZN(n264) );
  VHSR_MAOI222_2 U254 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_OR2_2 U255 ( .A1(n271), .A2(n221), .Z(n222) );
  VHSR_AOI32_2 U256 ( .A1(b[1]), .A2(n223), .A3(a[6]), .B1(n222), .B2(n223), 
        .ZN(n256) );
  VHSR_NOR2_1 U257 ( .A1(n263), .A2(n256), .ZN(n255) );
  VHSR_AOI32_2 U258 ( .A1(a[6]), .A2(n225), .A3(b[2]), .B1(n224), .B2(n225), 
        .ZN(n253) );
  VHSR_CLKNAND2_2 U259 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_CLKNAND2_2 U260 ( .A1(n247), .A2(n243), .ZN(n234) );
  VHSR_IN_2 U261 ( .I(b[7]), .ZN(n237) );
  VHSR_IN_2 U262 ( .I(a[3]), .ZN(n322) );
  VHSR_IN_2 U263 ( .I(a[2]), .ZN(n317) );
  VHSR_AOI22_2 U264 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[2]), .ZN(n262) );
  VHSR_NAND3_2 U266 ( .A1(a[3]), .A2(b[5]), .A3(n262), .ZN(n231) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[7]), .A2(a[2]), .ZN(n226) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[6]), .A2(a[1]), .ZN(n228) );
  VHSR_OAI22_2 U269 ( .A1(n232), .A2(n231), .B1(n226), .B2(n228), .ZN(n233) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[4]), .A2(a[0]), .ZN(n391) );
  VHSR_NAND3_2 U271 ( .A1(a[1]), .A2(b[5]), .A3(n391), .ZN(n261) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[0]), .ZN(n260) );
  VHSR_MAOI222_2 U273 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_NAND4_2 U274 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_IN_2 U275 ( .I(b[4]), .ZN(n276) );
  VHSR_IN_2 U276 ( .I(b[5]), .ZN(n278) );
  VHSR_OAI22_2 U277 ( .A1(n276), .A2(n322), .B1(n278), .B2(n317), .ZN(n227) );
  VHSR_AND2_2 U278 ( .A1(n239), .A2(n227), .Z(n230) );
  VHSR_IN_2 U279 ( .I(a[0]), .ZN(n313) );
  VHSR_OAI21_2 U280 ( .A1(n237), .A2(n313), .B(n228), .ZN(n229) );
  VHSR_NOR4_2 U281 ( .A1(n276), .A2(n278), .A3(n312), .A4(n313), .ZN(n269) );
  VHSR_AND2_2 U282 ( .A1(n259), .A2(n258), .Z(n257) );
  VHSR_AD1_1 U283 ( .A(n230), .B(n229), .CI(n269), .CO(n248), .S(n258) );
  VHSR_AOI21_2 U284 ( .A1(n232), .A2(n231), .B(n233), .ZN(n251) );
  VHSR_OAI32_2 U285 ( .A1(n233), .A2(n257), .A3(n248), .B1(n251), .B2(n233), 
        .ZN(n240) );
  VHSR_CLKNAND2_2 U286 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U287 ( .A1(n242), .A2(n238), .ZN(n236) );
  VHSR_NOR3_2 U288 ( .A1(n237), .A2(n322), .A3(n236), .ZN(n293) );
  VHSR_AOI21_2 U289 ( .A1(n235), .A2(n234), .B(n294), .ZN(n297) );
  VHSR_OAI32_2 U290 ( .A1(n293), .A2(n322), .A3(n237), .B1(n236), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U291 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U292 ( .A1(n242), .A2(n241), .ZN(n304) );
  VHSR_OAI21_2 U293 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U294 ( .A1(n247), .A2(n246), .ZN(n303) );
  VHSR_NOR2_1 U295 ( .A1(n257), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U296 ( .A1(n257), .A2(n248), .B1(n251), .B2(n250), .ZN(n249) );
  VHSR_OAI21_2 U297 ( .A1(n251), .A2(n250), .B(n249), .ZN(n309) );
  VHSR_AOI21_2 U298 ( .A1(n254), .A2(n253), .B(n252), .ZN(n308) );
  VHSR_AOI21_2 U299 ( .A1(n263), .A2(n256), .B(n255), .ZN(n325) );
  VHSR_IAO21_2 U300 ( .A1(n259), .A2(n258), .B(n257), .ZN(n324) );
  VHSR_AOI31_2 U301 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n333) );
  VHSR_OAI31_2 U302 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n267) );
  VHSR_IN_2 U303 ( .I(n267), .ZN(n332) );
  VHSR_CLKNAND2_2 U304 ( .A1(b[5]), .A2(a[0]), .ZN(n268) );
  VHSR_OAI32_2 U305 ( .A1(n269), .A2(n312), .A3(n276), .B1(n268), .B2(n269), 
        .ZN(n338) );
  VHSR_AOI22_2 U306 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n270) );
  VHSR_NOR2_1 U307 ( .A1(n271), .A2(n270), .ZN(n337) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[6]), .A2(b[6]), .ZN(n358) );
  VHSR_IN_2 U309 ( .I(n358), .ZN(n385) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[4]), .A2(b[6]), .ZN(n300) );
  VHSR_NAND3_2 U311 ( .A1(b[7]), .A2(a[5]), .A3(n300), .ZN(n273) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n301) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n301), .ZN(n272) );
  VHSR_CLKNAND2_2 U314 ( .A1(n273), .A2(n272), .ZN(n275) );
  VHSR_MAOI222_2 U315 ( .A(n358), .B(n273), .C(n272), .ZN(n342) );
  VHSR_IN_2 U316 ( .I(n342), .ZN(n274) );
  VHSR_OAI21_2 U317 ( .A1(n385), .A2(n275), .B(n274), .ZN(n288) );
  VHSR_AND3_2 U318 ( .A1(n370), .A2(a[5]), .A3(b[5]), .Z(n305) );
  VHSR_NOR3_2 U319 ( .A1(n279), .A2(n301), .A3(n278), .ZN(n350) );
  VHSR_AOI22_2 U320 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n280) );
  VHSR_NOR2_1 U321 ( .A1(n350), .A2(n280), .ZN(n284) );
  VHSR_AOI22_2 U322 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n281) );
  VHSR_NOR2_1 U323 ( .A1(n348), .A2(n281), .ZN(n283) );
  VHSR_IN_2 U324 ( .I(n282), .ZN(n291) );
  VHSR_MAOI222_2 U325 ( .A(n301), .B(n300), .C(n299), .ZN(n298) );
  VHSR_IN_2 U326 ( .I(n298), .ZN(n290) );
  VHSR_AD1_1 U327 ( .A(n305), .B(n284), .CI(n283), .CO(n285), .S(n282) );
  VHSR_NOR2_1 U328 ( .A1(n289), .A2(n285), .ZN(n287) );
  VHSR_CLKNAND2_2 U329 ( .A1(n289), .A2(n285), .ZN(n286) );
  VHSR_NOR2_1 U330 ( .A1(n287), .A2(n288), .ZN(n343) );
  VHSR_AOI22_2 U331 ( .A1(n288), .A2(n287), .B1(n286), .B2(n343), .ZN(n383) );
  VHSR_AOI21_2 U332 ( .A1(n291), .A2(n290), .B(n289), .ZN(n362) );
  VHSR_AD1_1 U333 ( .A(n294), .B(n293), .CI(n292), .CO(n384), .S(n361) );
  VHSR_AD1_1 U334 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n365) );
  VHSR_AOI31_2 U335 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n364) );
  VHSR_AD1_1 U336 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n368) );
  VHSR_AOI22_2 U337 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n306) );
  VHSR_NOR2_1 U338 ( .A1(n306), .A2(n305), .ZN(n367) );
  VHSR_AD1_1 U339 ( .A(n309), .B(n308), .CI(n307), .CO(n302), .S(n371) );
  VHSR_CLKNAND2_2 U340 ( .A1(b[2]), .A2(a[2]), .ZN(n326) );
  VHSR_NOR2_1 U341 ( .A1(n315), .A2(n322), .ZN(n311) );
  VHSR_OAI21_2 U342 ( .A1(n321), .A2(n317), .B(n311), .ZN(n310) );
  VHSR_OAI31_2 U343 ( .A1(n321), .A2(n311), .A3(n317), .B(n310), .ZN(n336) );
  VHSR_AOI22_2 U344 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n319) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[3]), .A2(a[3]), .ZN(n329) );
  VHSR_CLKNAND2_2 U346 ( .A1(b[1]), .A2(a[1]), .ZN(n399) );
  VHSR_OAI22_2 U347 ( .A1(n326), .A2(n319), .B1(n329), .B2(n399), .ZN(n320) );
  VHSR_OAI22_2 U348 ( .A1(n321), .A2(n313), .B1(n315), .B2(n312), .ZN(n381) );
  VHSR_AOI21_2 U349 ( .A1(n318), .A2(n316), .B(n313), .ZN(product[0]) );
  VHSR_AOI32_2 U350 ( .A1(b[0]), .A2(product[0]), .A3(a[2]), .B1(a[1]), .B2(
        product[0]), .ZN(n314) );
  VHSR_AOI211_2 U351 ( .A1(n318), .A2(n317), .B(n315), .C(n314), .ZN(n397) );
  VHSR_OAI22_2 U352 ( .A1(n318), .A2(n317), .B1(n316), .B2(n322), .ZN(n380) );
  VHSR_AOI21_2 U353 ( .A1(n319), .A2(n326), .B(n320), .ZN(n340) );
  VHSR_CLKNAND2_2 U354 ( .A1(n341), .A2(n340), .ZN(n339) );
  VHSR_CLKNAND2_2 U355 ( .A1(n336), .A2(n335), .ZN(n327) );
  VHSR_AOI211_2 U356 ( .A1(n326), .A2(n327), .B(n322), .C(n321), .ZN(n374) );
  VHSR_AD1_1 U357 ( .A(n325), .B(n324), .CI(n323), .CO(n307), .S(n373) );
  VHSR_IN_2 U358 ( .I(n326), .ZN(n330) );
  VHSR_IN_2 U359 ( .I(n327), .ZN(n334) );
  VHSR_CLKNAND2_2 U360 ( .A1(n334), .A2(n329), .ZN(n328) );
  VHSR_OAI31_2 U361 ( .A1(n330), .A2(n334), .A3(n329), .B(n328), .ZN(n377) );
  VHSR_AD1_1 U362 ( .A(n333), .B(n332), .CI(n331), .CO(n323), .S(n376) );
  VHSR_IAO21_2 U363 ( .A1(n336), .A2(n335), .B(n334), .ZN(n379) );
  VHSR_AD1_1 U364 ( .A(n338), .B(n390), .CI(n337), .CO(n331), .S(n378) );
  VHSR_OAI21_2 U365 ( .A1(n341), .A2(n340), .B(n339), .ZN(n395) );
  VHSR_AOI211_2 U366 ( .A1(n392), .A2(n391), .B(n390), .C(n395), .ZN(n394) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[6]), .A2(b[7]), .ZN(n345) );
  VHSR_AOI21_2 U368 ( .A1(a[7]), .A2(b[6]), .B(n345), .ZN(n344) );
  VHSR_AOI31_2 U369 ( .A1(a[7]), .A2(n345), .A3(b[6]), .B(n344), .ZN(n346) );
  VHSR_IN_2 U370 ( .I(n346), .ZN(n347) );
  VHSR_OR2_2 U371 ( .A1(n348), .A2(n347), .Z(n349) );
  VHSR_MAOI222_2 U372 ( .A(n350), .B(n348), .C(n347), .ZN(n357) );
  VHSR_OAI21_2 U373 ( .A1(n350), .A2(n349), .B(n357), .ZN(n354) );
  VHSR_CLKXOR2_2 U374 ( .A1(n355), .A2(n354), .Z(n351) );
  VHSR_CLKNAND2_2 U375 ( .A1(n352), .A2(n351), .ZN(n387) );
  VHSR_OAI21_2 U376 ( .A1(n352), .A2(n351), .B(n387), .ZN(n353) );
  VHSR_CLKNAND2_2 U377 ( .A1(a[7]), .A2(b[7]), .ZN(n386) );
  VHSR_NOR2_1 U378 ( .A1(n355), .A2(n354), .ZN(n356) );
  VHSR_AND3_2 U379 ( .A1(n388), .A2(n358), .A3(n387), .Z(n359) );
  VHSR_NOR2_1 U380 ( .A1(n386), .A2(n359), .ZN(product[15]) );
  VHSR_AD1_1 U381 ( .A(n384), .B(n383), .CI(n382), .CO(n352), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U382 ( .A1(n386), .A2(n385), .ZN(n389) );
  VHSR_XOR3_2 U383 ( .A1(n389), .A2(n388), .A3(n387), .Z(product[14]) );
  VHSR_AOI21_2 U384 ( .A1(n392), .A2(n391), .B(n390), .ZN(n393) );
  VHSR_IN_2 U385 ( .I(n393), .ZN(n396) );
  VHSR_AOI21_2 U386 ( .A1(n396), .A2(n395), .B(n394), .ZN(product[4]) );
  VHSR_AOI22_2 U387 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n398) );
  VHSR_AOI21_2 U388 ( .A1(n399), .A2(n398), .B(n397), .ZN(product[2]) );
endmodule


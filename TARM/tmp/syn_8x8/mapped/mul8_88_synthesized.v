
module mul8_88 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \mul_ll_ll/out[0] , \intadd_0/SUM[7] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n215, n216, n217, n218, n219, n220, n221, n222,
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
         n399, n400, n401, n402, n403, n404, n405, n406, n407, n408, n409;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U206 ( .A1(n245), .B1(n227), .ZN(n228) );
  VHSR_INOR2_2 U207 ( .A1(n235), .B1(n253), .ZN(n246) );
  VHSR_INAND2_2 U208 ( .A1(n362), .B1(n360), .ZN(n363) );
  VHSR_INOR2_2 U209 ( .A1(n231), .B1(n258), .ZN(n255) );
  VHSR_NOR2_1 U210 ( .A1(n273), .A2(n310), .ZN(n287) );
  VHSR_INAND2_2 U211 ( .A1(n332), .B1(n346), .ZN(n344) );
  VHSR_NOR2_1 U212 ( .A1(n296), .A2(n300), .ZN(n295) );
  VHSR_NOR2_1 U213 ( .A1(n238), .A2(n237), .ZN(n298) );
  VHSR_IOA21_2 U214 ( .A1(n328), .A2(n327), .B(n326), .ZN(n407) );
  VHSR_INOR2_2 U215 ( .A1(n371), .B1(n370), .ZN(n402) );
  VHSR_IN_2 U216 ( .I(n367), .ZN(product[13]) );
  VHSR_CLKN_1 U217 ( .I(n372), .ZN(n373) );
  VHSR_INAND3_1 U218 ( .A1(n399), .B1(n402), .B2(n401), .ZN(n372) );
  VHSR_INOR2_1 U219 ( .A1(n357), .B1(n356), .ZN(n369) );
  VHSR_NOR2_2 U220 ( .A1(n406), .A2(n405), .ZN(n404) );
  VHSR_INOR3_1 U221 ( .A1(n287), .B1(n278), .B2(n313), .ZN(n364) );
  VHSR_AD1_1 U222 ( .A(n391), .B(n390), .CI(n389), .CO(n386), .S(product[6])
         );
  VHSR_AD1_1 U223 ( .A(n385), .B(n384), .CI(n383), .CO(n380), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n379), .B(n378), .CI(n377), .CO(n374), .S(product[10])
         );
  VHSR_AD1_1 U225 ( .A(n395), .B(n407), .CI(n394), .CO(n348), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n393), .B(n404), .CI(n392), .CO(n389), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n388), .B(n387), .CI(n386), .CO(n383), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U228 ( .A(n382), .B(n381), .CI(n380), .CO(n377), .S(product[9])
         );
  VHSR_AD1_1 U229 ( .A(n376), .B(n375), .CI(n374), .CO(n396), .S(product[11])
         );
  VHSR_PULL0_0 U230 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U231 ( .I(b[1]), .ZN(n330) );
  VHSR_IN_2 U232 ( .I(a[0]), .ZN(n325) );
  VHSR_NOR2_1 U233 ( .A1(n330), .A2(n325), .ZN(product[1]) );
  VHSR_IN_2 U234 ( .I(b[7]), .ZN(n281) );
  VHSR_IN_2 U235 ( .I(a[3]), .ZN(n334) );
  VHSR_IN_2 U236 ( .I(b[6]), .ZN(n282) );
  VHSR_IN_2 U237 ( .I(a[2]), .ZN(n329) );
  VHSR_OAI22_2 U238 ( .A1(n282), .A2(n334), .B1(n281), .B2(n329), .ZN(n243) );
  VHSR_AOI22_2 U239 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n221) );
  VHSR_CLKNAND2_2 U240 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_NAND3_2 U241 ( .A1(a[3]), .A2(b[5]), .A3(n263), .ZN(n220) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[7]), .A2(a[2]), .ZN(n215) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[6]), .A2(a[1]), .ZN(n217) );
  VHSR_OAI22_2 U244 ( .A1(n221), .A2(n220), .B1(n215), .B2(n217), .ZN(n222) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[6]), .A2(a[0]), .ZN(n262) );
  VHSR_IN_2 U246 ( .I(b[4]), .ZN(n310) );
  VHSR_OAI211_2 U247 ( .A1(n310), .A2(n325), .B(b[5]), .C(a[1]), .ZN(n261) );
  VHSR_MAOI222_2 U248 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_IN_2 U249 ( .I(b[5]), .ZN(n313) );
  VHSR_IN_2 U250 ( .I(a[1]), .ZN(n323) );
  VHSR_NOR4_2 U251 ( .A1(n310), .A2(n313), .A3(n323), .A4(n325), .ZN(n270) );
  VHSR_NAND4_2 U252 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n240) );
  VHSR_OAI22_2 U253 ( .A1(n310), .A2(n334), .B1(n313), .B2(n329), .ZN(n216) );
  VHSR_AND2_2 U254 ( .A1(n240), .A2(n216), .Z(n219) );
  VHSR_OAI21_2 U255 ( .A1(n281), .A2(n325), .B(n217), .ZN(n218) );
  VHSR_AND2_2 U256 ( .A1(n260), .A2(n257), .Z(n256) );
  VHSR_AD1_1 U257 ( .A(n270), .B(n219), .CI(n218), .CO(n249), .S(n257) );
  VHSR_AOI21_2 U258 ( .A1(n221), .A2(n220), .B(n222), .ZN(n252) );
  VHSR_OAI32_2 U259 ( .A1(n222), .A2(n256), .A3(n249), .B1(n252), .B2(n222), 
        .ZN(n241) );
  VHSR_CLKNAND2_2 U260 ( .A1(n241), .A2(n240), .ZN(n239) );
  VHSR_CLKNAND2_2 U261 ( .A1(n243), .A2(n239), .ZN(n236) );
  VHSR_NOR3_2 U262 ( .A1(n281), .A2(n334), .A3(n236), .ZN(n299) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[3]), .A2(a[7]), .ZN(n238) );
  VHSR_IN_2 U264 ( .I(b[3]), .ZN(n333) );
  VHSR_IN_2 U265 ( .I(a[6]), .ZN(n273) );
  VHSR_IN_2 U266 ( .I(a[7]), .ZN(n278) );
  VHSR_IN_2 U267 ( .I(b[2]), .ZN(n324) );
  VHSR_OAI22_2 U268 ( .A1(n333), .A2(n273), .B1(n278), .B2(n324), .ZN(n248) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[2]), .A2(a[4]), .ZN(n226) );
  VHSR_CLKNAND2_2 U270 ( .A1(a[6]), .A2(b[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[3]), .A2(a[6]), .ZN(n223) );
  VHSR_OAI22_2 U272 ( .A1(n238), .A2(n232), .B1(n223), .B2(n324), .ZN(n225) );
  VHSR_NOR3_2 U273 ( .A1(n278), .A2(n324), .A3(n232), .ZN(n224) );
  VHSR_AOI31_2 U274 ( .A1(a[5]), .A2(n226), .A3(n225), .B(n224), .ZN(n235) );
  VHSR_IN_2 U275 ( .I(n232), .ZN(n230) );
  VHSR_IN_2 U276 ( .I(a[4]), .ZN(n350) );
  VHSR_IN_2 U277 ( .I(a[5]), .ZN(n311) );
  VHSR_IN_2 U278 ( .I(b[0]), .ZN(n349) );
  VHSR_NOR4_2 U279 ( .A1(n350), .A2(n311), .A3(n330), .A4(n349), .ZN(n272) );
  VHSR_IN_2 U280 ( .I(n226), .ZN(n267) );
  VHSR_NAND3_2 U281 ( .A1(b[3]), .A2(n267), .A3(a[5]), .ZN(n245) );
  VHSR_AOI22_2 U282 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n227) );
  VHSR_MAOI222_2 U283 ( .A(n230), .B(n272), .C(n228), .ZN(n231) );
  VHSR_AOI211_2 U284 ( .A1(a[4]), .A2(b[0]), .B(n311), .C(n330), .ZN(n266) );
  VHSR_NOR2_1 U285 ( .A1(n273), .A2(n349), .ZN(n265) );
  VHSR_MAOI222_2 U286 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_OR2_2 U287 ( .A1(n272), .A2(n228), .Z(n229) );
  VHSR_OAI21_2 U288 ( .A1(n230), .A2(n229), .B(n231), .ZN(n259) );
  VHSR_NOR2_1 U289 ( .A1(n264), .A2(n259), .ZN(n258) );
  VHSR_CLKNAND2_2 U290 ( .A1(b[3]), .A2(a[5]), .ZN(n233) );
  VHSR_OAI22_2 U291 ( .A1(n267), .A2(n233), .B1(n278), .B2(n232), .ZN(n234) );
  VHSR_AOI32_2 U292 ( .A1(b[2]), .A2(n235), .A3(a[6]), .B1(n234), .B2(n235), 
        .ZN(n254) );
  VHSR_NOR2_1 U293 ( .A1(n255), .A2(n254), .ZN(n253) );
  VHSR_CLKNAND2_2 U294 ( .A1(n246), .A2(n245), .ZN(n244) );
  VHSR_CLKNAND2_2 U295 ( .A1(n248), .A2(n244), .ZN(n237) );
  VHSR_OAI32_2 U296 ( .A1(n299), .A2(n334), .A3(n281), .B1(n236), .B2(n299), 
        .ZN(n306) );
  VHSR_AOI21_2 U297 ( .A1(n238), .A2(n237), .B(n298), .ZN(n305) );
  VHSR_OAI21_2 U298 ( .A1(n241), .A2(n240), .B(n239), .ZN(n242) );
  VHSR_XNOR2_2 U299 ( .A1(n243), .A2(n242), .ZN(n309) );
  VHSR_OAI21_2 U300 ( .A1(n246), .A2(n245), .B(n244), .ZN(n247) );
  VHSR_XNOR2_2 U301 ( .A1(n248), .A2(n247), .ZN(n308) );
  VHSR_NOR2_1 U302 ( .A1(n256), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U303 ( .A1(n256), .A2(n249), .B1(n252), .B2(n251), .ZN(n250) );
  VHSR_OAI21_2 U304 ( .A1(n252), .A2(n251), .B(n250), .ZN(n317) );
  VHSR_AOI21_2 U305 ( .A1(n255), .A2(n254), .B(n253), .ZN(n316) );
  VHSR_IAO21_2 U306 ( .A1(n260), .A2(n257), .B(n256), .ZN(n320) );
  VHSR_AOI21_2 U307 ( .A1(n264), .A2(n259), .B(n258), .ZN(n319) );
  VHSR_AOI31_2 U308 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n338) );
  VHSR_OAI31_2 U309 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n268) );
  VHSR_IN_2 U310 ( .I(n268), .ZN(n337) );
  VHSR_CLKNAND2_2 U311 ( .A1(b[5]), .A2(a[0]), .ZN(n269) );
  VHSR_OAI32_2 U312 ( .A1(n270), .A2(n323), .A3(n310), .B1(n269), .B2(n270), 
        .ZN(n355) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[4]), .A2(b[4]), .ZN(n280) );
  VHSR_IN_2 U314 ( .I(n280), .ZN(n384) );
  VHSR_NAND3_2 U315 ( .A1(b[0]), .A2(n384), .A3(a[0]), .ZN(n352) );
  VHSR_IN_2 U316 ( .I(n352), .ZN(n354) );
  VHSR_AOI22_2 U317 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n271) );
  VHSR_NOR2_1 U318 ( .A1(n272), .A2(n271), .ZN(n353) );
  VHSR_NOR2_1 U319 ( .A1(n273), .A2(n282), .ZN(n399) );
  VHSR_NOR2_1 U320 ( .A1(n350), .A2(n282), .ZN(n286) );
  VHSR_CLKNAND2_2 U321 ( .A1(a[5]), .A2(b[7]), .ZN(n275) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[7]), .A2(b[5]), .ZN(n274) );
  VHSR_OAI22_2 U323 ( .A1(n286), .A2(n275), .B1(n287), .B2(n274), .ZN(n277) );
  VHSR_OR2_2 U324 ( .A1(n286), .A2(n287), .Z(n301) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[5]), .A2(b[5]), .ZN(n285) );
  VHSR_CLKNAND2_2 U326 ( .A1(a[7]), .A2(b[7]), .ZN(n400) );
  VHSR_NOR3_2 U327 ( .A1(n301), .A2(n285), .A3(n400), .ZN(n276) );
  VHSR_AOI31_2 U328 ( .A1(b[6]), .A2(a[6]), .A3(n277), .B(n276), .ZN(n357) );
  VHSR_OAI21_2 U329 ( .A1(n399), .A2(n277), .B(n357), .ZN(n294) );
  VHSR_AOI22_2 U330 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n279) );
  VHSR_NOR2_1 U331 ( .A1(n364), .A2(n279), .ZN(n290) );
  VHSR_NOR2_1 U332 ( .A1(n285), .A2(n280), .ZN(n289) );
  VHSR_NOR4_2 U333 ( .A1(n350), .A2(n311), .A3(n282), .A4(n281), .ZN(n362) );
  VHSR_AOI22_2 U334 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n283) );
  VHSR_NOR2_1 U335 ( .A1(n362), .A2(n283), .ZN(n288) );
  VHSR_IN_2 U336 ( .I(n284), .ZN(n296) );
  VHSR_NOR2_1 U337 ( .A1(n384), .A2(n285), .ZN(n302) );
  VHSR_AOI22_2 U338 ( .A1(n287), .A2(n286), .B1(n302), .B2(n301), .ZN(n300) );
  VHSR_AD1_1 U339 ( .A(n290), .B(n289), .CI(n288), .CO(n291), .S(n284) );
  VHSR_NOR2_1 U340 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U341 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_NOR2_1 U342 ( .A1(n293), .A2(n294), .ZN(n356) );
  VHSR_AOI22_2 U343 ( .A1(n294), .A2(n293), .B1(n292), .B2(n356), .ZN(n397) );
  VHSR_AOI21_2 U344 ( .A1(n300), .A2(n296), .B(n295), .ZN(n376) );
  VHSR_AD1_1 U345 ( .A(n299), .B(n298), .CI(n297), .CO(n398), .S(n375) );
  VHSR_OAI21_2 U346 ( .A1(n302), .A2(n301), .B(n300), .ZN(n303) );
  VHSR_IN_2 U347 ( .I(n303), .ZN(n379) );
  VHSR_AD1_1 U348 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n378) );
  VHSR_AD1_1 U349 ( .A(n309), .B(n308), .CI(n307), .CO(n304), .S(n382) );
  VHSR_NOR2_1 U350 ( .A1(n311), .A2(n310), .ZN(n314) );
  VHSR_OAI21_2 U351 ( .A1(n350), .A2(n313), .B(n314), .ZN(n312) );
  VHSR_OAI31_2 U352 ( .A1(n350), .A2(n314), .A3(n313), .B(n312), .ZN(n381) );
  VHSR_AD1_1 U353 ( .A(n317), .B(n316), .CI(n315), .CO(n307), .S(n385) );
  VHSR_AD1_1 U354 ( .A(n320), .B(n319), .CI(n318), .CO(n315), .S(n388) );
  VHSR_NOR2_1 U355 ( .A1(n324), .A2(n329), .ZN(n342) );
  VHSR_IN_2 U356 ( .I(n342), .ZN(n335) );
  VHSR_NOR2_1 U357 ( .A1(n324), .A2(n334), .ZN(n322) );
  VHSR_OAI21_2 U358 ( .A1(n333), .A2(n329), .B(n322), .ZN(n321) );
  VHSR_OAI31_2 U359 ( .A1(n333), .A2(n322), .A3(n329), .B(n321), .ZN(n345) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[3]), .A2(a[3]), .ZN(n341) );
  VHSR_CLKNAND2_2 U361 ( .A1(b[1]), .A2(a[1]), .ZN(n408) );
  VHSR_AOI22_2 U362 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n331) );
  VHSR_OAI22_2 U363 ( .A1(n341), .A2(n408), .B1(n335), .B2(n331), .ZN(n332) );
  VHSR_OAI22_2 U364 ( .A1(n333), .A2(n325), .B1(n324), .B2(n323), .ZN(n395) );
  VHSR_IN_2 U365 ( .I(n408), .ZN(n328) );
  VHSR_NOR2_1 U366 ( .A1(n349), .A2(n329), .ZN(n327) );
  VHSR_OAI211_2 U367 ( .A1(n327), .A2(n328), .B(b[2]), .C(a[0]), .ZN(n326) );
  VHSR_OAI22_2 U368 ( .A1(n330), .A2(n329), .B1(n349), .B2(n334), .ZN(n394) );
  VHSR_AOI21_2 U369 ( .A1(n331), .A2(n335), .B(n332), .ZN(n347) );
  VHSR_CLKNAND2_2 U370 ( .A1(n348), .A2(n347), .ZN(n346) );
  VHSR_CLKNAND2_2 U371 ( .A1(n345), .A2(n344), .ZN(n339) );
  VHSR_AOI211_2 U372 ( .A1(n335), .A2(n339), .B(n334), .C(n333), .ZN(n387) );
  VHSR_AD1_1 U373 ( .A(n338), .B(n337), .CI(n336), .CO(n318), .S(n391) );
  VHSR_IN_2 U374 ( .I(n339), .ZN(n343) );
  VHSR_CLKNAND2_2 U375 ( .A1(n343), .A2(n341), .ZN(n340) );
  VHSR_OAI31_2 U376 ( .A1(n342), .A2(n343), .A3(n341), .B(n340), .ZN(n390) );
  VHSR_IAO21_2 U377 ( .A1(n345), .A2(n344), .B(n343), .ZN(n393) );
  VHSR_OAI21_2 U378 ( .A1(n348), .A2(n347), .B(n346), .ZN(n406) );
  VHSR_NOR2_1 U379 ( .A1(n350), .A2(n349), .ZN(n351) );
  VHSR_AOI32_2 U380 ( .A1(b[4]), .A2(n352), .A3(a[0]), .B1(n351), .B2(n352), 
        .ZN(n405) );
  VHSR_AD1_1 U381 ( .A(n355), .B(n354), .CI(n353), .CO(n336), .S(n392) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[6]), .A2(b[7]), .ZN(n359) );
  VHSR_AOI21_2 U383 ( .A1(a[7]), .A2(b[6]), .B(n359), .ZN(n358) );
  VHSR_AOI31_2 U384 ( .A1(a[7]), .A2(n359), .A3(b[6]), .B(n358), .ZN(n360) );
  VHSR_IN_2 U385 ( .I(n360), .ZN(n361) );
  VHSR_MAOI222_2 U386 ( .A(n364), .B(n362), .C(n361), .ZN(n371) );
  VHSR_OAI21_2 U387 ( .A1(n364), .A2(n363), .B(n371), .ZN(n368) );
  VHSR_CLKXOR2_2 U388 ( .A1(n369), .A2(n368), .Z(n365) );
  VHSR_CLKNAND2_2 U389 ( .A1(n366), .A2(n365), .ZN(n401) );
  VHSR_OAI21_2 U390 ( .A1(n366), .A2(n365), .B(n401), .ZN(n367) );
  VHSR_NOR2_1 U391 ( .A1(n369), .A2(n368), .ZN(n370) );
  VHSR_NOR2_1 U392 ( .A1(n400), .A2(n373), .ZN(product[15]) );
  VHSR_AD1_1 U393 ( .A(n398), .B(n397), .CI(n396), .CO(n366), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U394 ( .A1(n400), .A2(n399), .ZN(n403) );
  VHSR_XOR3_2 U395 ( .A1(n403), .A2(n402), .A3(n401), .Z(product[14]) );
  VHSR_AOI21_2 U396 ( .A1(n406), .A2(n405), .B(n404), .ZN(product[4]) );
  VHSR_AOI22_2 U397 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n409) );
  VHSR_AOI21_2 U398 ( .A1(n409), .A2(n408), .B(n407), .ZN(product[2]) );
endmodule


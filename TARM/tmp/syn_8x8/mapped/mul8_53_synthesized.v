
module mul8_53 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n213, n214,
         n215, n216, n217, n218, n219, n220, n221, n222, n223, n224, n225,
         n226, n227, n228, n229, n230, n231, n232, n233, n234, n235, n236,
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
         n402, n403, n404, n405, n406, n407;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U204 ( .A1(n378), .B1(n283), .ZN(n287) );
  VHSR_NOR2_1 U205 ( .A1(n312), .A2(n280), .ZN(n284) );
  VHSR_NOR2_1 U206 ( .A1(n246), .A2(n242), .ZN(n236) );
  VHSR_INAND2_2 U207 ( .A1(n324), .B1(n341), .ZN(n339) );
  VHSR_NOR2_1 U208 ( .A1(n294), .A2(n298), .ZN(n293) );
  VHSR_NOR2_1 U209 ( .A1(n291), .A2(n292), .ZN(n350) );
  VHSR_IOA21_2 U210 ( .A1(n321), .A2(n320), .B(n319), .ZN(n405) );
  VHSR_NOR2_1 U211 ( .A1(n312), .A2(n344), .ZN(n378) );
  VHSR_IN_2 U212 ( .I(n361), .ZN(product[13]) );
  VHSR_INOR3_1 U213 ( .A1(n236), .B1(n325), .B2(n277), .ZN(n296) );
  VHSR_NOR2_2 U214 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_INOR2_1 U215 ( .A1(n365), .B1(n364), .ZN(n396) );
  VHSR_INOR2_1 U216 ( .A1(n351), .B1(n350), .ZN(n363) );
  VHSR_NOR2_2 U217 ( .A1(n400), .A2(n399), .ZN(n398) );
  VHSR_NOR2_2 U218 ( .A1(n293), .A2(n289), .ZN(n291) );
  VHSR_INAND2_1 U219 ( .A1(n356), .B1(n354), .ZN(n357) );
  VHSR_MOAI22_1 U220 ( .A1(n277), .A2(n404), .B1(a[6]), .B2(b[2]), .ZN(n224)
         );
  VHSR_AD1_1 U221 ( .A(n385), .B(n384), .CI(n383), .CO(n380), .S(product[6])
         );
  VHSR_AD1_1 U222 ( .A(n379), .B(n378), .CI(n377), .CO(n374), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U223 ( .A(n373), .B(n372), .CI(n371), .CO(n368), .S(product[10])
         );
  VHSR_AD1_1 U224 ( .A(n389), .B(n405), .CI(n388), .CO(n343), .S(product[3])
         );
  VHSR_AD1_1 U225 ( .A(n387), .B(n398), .CI(n386), .CO(n383), .S(product[5])
         );
  VHSR_AD1_1 U226 ( .A(n382), .B(n381), .CI(n380), .CO(n377), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U227 ( .A(n376), .B(n375), .CI(n374), .CO(n371), .S(product[9])
         );
  VHSR_AD1_1 U228 ( .A(n370), .B(n369), .CI(n368), .CO(n390), .S(product[11])
         );
  VHSR_IN_2 U229 ( .I(b[7]), .ZN(n279) );
  VHSR_IN_2 U230 ( .I(a[3]), .ZN(n326) );
  VHSR_IN_2 U231 ( .I(b[6]), .ZN(n280) );
  VHSR_IN_2 U232 ( .I(a[2]), .ZN(n322) );
  VHSR_OAI22_2 U233 ( .A1(n280), .A2(n326), .B1(n279), .B2(n322), .ZN(n241) );
  VHSR_AOI22_2 U234 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n219) );
  VHSR_CLKNAND2_2 U235 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_NAND3_2 U236 ( .A1(a[3]), .A2(b[5]), .A3(n263), .ZN(n218) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[7]), .A2(a[2]), .ZN(n213) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[6]), .A2(a[1]), .ZN(n215) );
  VHSR_OAI22_2 U239 ( .A1(n219), .A2(n218), .B1(n213), .B2(n215), .ZN(n220) );
  VHSR_IN_2 U240 ( .I(b[4]), .ZN(n344) );
  VHSR_IN_2 U241 ( .I(a[0]), .ZN(n403) );
  VHSR_OAI211_2 U242 ( .A1(n344), .A2(n403), .B(b[5]), .C(a[1]), .ZN(n262) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[6]), .A2(a[0]), .ZN(n261) );
  VHSR_MAOI222_2 U244 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NAND4_2 U245 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_IN_2 U246 ( .I(b[5]), .ZN(n310) );
  VHSR_OAI22_2 U247 ( .A1(n344), .A2(n326), .B1(n310), .B2(n322), .ZN(n214) );
  VHSR_AND2_2 U248 ( .A1(n238), .A2(n214), .Z(n217) );
  VHSR_OAI21_2 U249 ( .A1(n279), .A2(n403), .B(n215), .ZN(n216) );
  VHSR_IN_2 U250 ( .I(a[1]), .ZN(n401) );
  VHSR_NOR4_2 U251 ( .A1(n344), .A2(n310), .A3(n401), .A4(n403), .ZN(n269) );
  VHSR_AND2_2 U252 ( .A1(n260), .A2(n259), .Z(n258) );
  VHSR_AD1_1 U253 ( .A(n217), .B(n216), .CI(n269), .CO(n247), .S(n259) );
  VHSR_AOI21_2 U254 ( .A1(n219), .A2(n218), .B(n220), .ZN(n250) );
  VHSR_OAI32_2 U255 ( .A1(n220), .A2(n258), .A3(n247), .B1(n250), .B2(n220), 
        .ZN(n239) );
  VHSR_CLKNAND2_2 U256 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U257 ( .A1(n241), .A2(n237), .ZN(n234) );
  VHSR_NOR3_2 U258 ( .A1(n279), .A2(n326), .A3(n234), .ZN(n297) );
  VHSR_AOI22_2 U259 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n246) );
  VHSR_IN_2 U260 ( .I(b[3]), .ZN(n325) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[2]), .A2(a[4]), .ZN(n267) );
  VHSR_IN_2 U262 ( .I(a[5]), .ZN(n308) );
  VHSR_NOR3_2 U263 ( .A1(n325), .A2(n267), .A3(n308), .ZN(n244) );
  VHSR_IN_2 U264 ( .I(a[7]), .ZN(n277) );
  VHSR_IN_2 U265 ( .I(b[1]), .ZN(n404) );
  VHSR_NOR2_1 U266 ( .A1(n277), .A2(n404), .ZN(n222) );
  VHSR_AND2_2 U267 ( .A1(a[6]), .A2(b[2]), .Z(n221) );
  VHSR_AOI211_2 U268 ( .A1(a[4]), .A2(b[2]), .B(n325), .C(n308), .ZN(n223) );
  VHSR_MAOI222_2 U269 ( .A(n222), .B(n221), .C(n223), .ZN(n233) );
  VHSR_OAI21_2 U270 ( .A1(n224), .A2(n223), .B(n233), .ZN(n225) );
  VHSR_IN_2 U271 ( .I(n225), .ZN(n253) );
  VHSR_CLKNAND2_2 U272 ( .A1(a[6]), .A2(b[1]), .ZN(n230) );
  VHSR_IN_2 U273 ( .I(n230), .ZN(n227) );
  VHSR_IN_2 U274 ( .I(a[4]), .ZN(n312) );
  VHSR_IN_2 U275 ( .I(b[0]), .ZN(n402) );
  VHSR_NOR4_2 U276 ( .A1(n312), .A2(n308), .A3(n404), .A4(n402), .ZN(n271) );
  VHSR_AOI22_2 U277 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n226) );
  VHSR_NOR2_1 U278 ( .A1(n244), .A2(n226), .ZN(n228) );
  VHSR_MAOI222_2 U279 ( .A(n227), .B(n271), .C(n228), .ZN(n232) );
  VHSR_OAI21_2 U280 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n266) );
  VHSR_OAI211_2 U281 ( .A1(n312), .A2(n402), .B(a[5]), .C(b[1]), .ZN(n265) );
  VHSR_MAOI222_2 U282 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NOR2_1 U283 ( .A1(n271), .A2(n228), .ZN(n231) );
  VHSR_IN_2 U284 ( .I(n232), .ZN(n229) );
  VHSR_AOI21_2 U285 ( .A1(n231), .A2(n230), .B(n229), .ZN(n256) );
  VHSR_CLKNAND2_2 U286 ( .A1(n264), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U287 ( .A1(n232), .A2(n255), .ZN(n252) );
  VHSR_CLKNAND2_2 U288 ( .A1(n253), .A2(n252), .ZN(n251) );
  VHSR_CLKNAND2_2 U289 ( .A1(n233), .A2(n251), .ZN(n243) );
  VHSR_OAI32_2 U290 ( .A1(n297), .A2(n326), .A3(n279), .B1(n234), .B2(n297), 
        .ZN(n304) );
  VHSR_NOR2_1 U291 ( .A1(n325), .A2(n277), .ZN(n235) );
  VHSR_IAO21_2 U292 ( .A1(n236), .A2(n235), .B(n296), .ZN(n303) );
  VHSR_OAI21_2 U293 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U294 ( .A1(n241), .A2(n240), .ZN(n307) );
  VHSR_AOI21_2 U295 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U296 ( .A1(n246), .A2(n245), .ZN(n306) );
  VHSR_NOR2_1 U297 ( .A1(n258), .A2(n247), .ZN(n249) );
  VHSR_AOI22_2 U298 ( .A1(n258), .A2(n247), .B1(n250), .B2(n249), .ZN(n248) );
  VHSR_OAI21_2 U299 ( .A1(n250), .A2(n249), .B(n248), .ZN(n315) );
  VHSR_OAI21_2 U300 ( .A1(n253), .A2(n252), .B(n251), .ZN(n254) );
  VHSR_IN_2 U301 ( .I(n254), .ZN(n314) );
  VHSR_OAI21_2 U302 ( .A1(n264), .A2(n256), .B(n255), .ZN(n257) );
  VHSR_IN_2 U303 ( .I(n257), .ZN(n330) );
  VHSR_IAO21_2 U304 ( .A1(n260), .A2(n259), .B(n258), .ZN(n329) );
  VHSR_AOI31_2 U305 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n333) );
  VHSR_AOI31_2 U306 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n332) );
  VHSR_CLKNAND2_2 U307 ( .A1(b[5]), .A2(a[0]), .ZN(n268) );
  VHSR_OAI32_2 U308 ( .A1(n269), .A2(n401), .A3(n344), .B1(n268), .B2(n269), 
        .ZN(n349) );
  VHSR_NOR2_1 U309 ( .A1(n402), .A2(n403), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U310 ( .A1(n378), .A2(product[0]), .ZN(n346) );
  VHSR_IN_2 U311 ( .I(n346), .ZN(n348) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[5]), .A2(b[0]), .ZN(n270) );
  VHSR_OAI32_2 U313 ( .A1(n271), .A2(n404), .A3(n312), .B1(n270), .B2(n271), 
        .ZN(n347) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[6]), .A2(b[6]), .ZN(n366) );
  VHSR_IN_2 U315 ( .I(n366), .ZN(n393) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[5]), .A2(b[7]), .ZN(n273) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[6]), .A2(b[4]), .ZN(n276) );
  VHSR_IN_2 U318 ( .I(n276), .ZN(n285) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[7]), .A2(b[5]), .ZN(n272) );
  VHSR_OAI22_2 U320 ( .A1(n284), .A2(n273), .B1(n285), .B2(n272), .ZN(n275) );
  VHSR_OR2_2 U321 ( .A1(n284), .A2(n285), .Z(n299) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[5]), .A2(b[5]), .ZN(n283) );
  VHSR_CLKNAND2_2 U323 ( .A1(a[7]), .A2(b[7]), .ZN(n394) );
  VHSR_NOR3_2 U324 ( .A1(n299), .A2(n283), .A3(n394), .ZN(n274) );
  VHSR_AOI31_2 U325 ( .A1(b[6]), .A2(a[6]), .A3(n275), .B(n274), .ZN(n351) );
  VHSR_OAI21_2 U326 ( .A1(n393), .A2(n275), .B(n351), .ZN(n292) );
  VHSR_NOR3_2 U327 ( .A1(n277), .A2(n276), .A3(n310), .ZN(n358) );
  VHSR_AOI22_2 U328 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n278) );
  VHSR_NOR2_1 U329 ( .A1(n358), .A2(n278), .ZN(n288) );
  VHSR_NOR4_2 U330 ( .A1(n312), .A2(n308), .A3(n280), .A4(n279), .ZN(n356) );
  VHSR_AOI22_2 U331 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n281) );
  VHSR_NOR2_1 U332 ( .A1(n356), .A2(n281), .ZN(n286) );
  VHSR_IN_2 U333 ( .I(n282), .ZN(n294) );
  VHSR_NOR2_1 U334 ( .A1(n378), .A2(n283), .ZN(n300) );
  VHSR_AOI22_2 U335 ( .A1(n285), .A2(n284), .B1(n300), .B2(n299), .ZN(n298) );
  VHSR_AD1_1 U336 ( .A(n288), .B(n287), .CI(n286), .CO(n289), .S(n282) );
  VHSR_CLKNAND2_2 U337 ( .A1(n293), .A2(n289), .ZN(n290) );
  VHSR_AOI22_2 U338 ( .A1(n292), .A2(n291), .B1(n290), .B2(n350), .ZN(n391) );
  VHSR_AOI21_2 U339 ( .A1(n298), .A2(n294), .B(n293), .ZN(n370) );
  VHSR_AD1_1 U340 ( .A(n297), .B(n296), .CI(n295), .CO(n392), .S(n369) );
  VHSR_OAI21_2 U341 ( .A1(n300), .A2(n299), .B(n298), .ZN(n301) );
  VHSR_IN_2 U342 ( .I(n301), .ZN(n373) );
  VHSR_AD1_1 U343 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n372) );
  VHSR_AD1_1 U344 ( .A(n307), .B(n306), .CI(n305), .CO(n302), .S(n376) );
  VHSR_NOR2_1 U345 ( .A1(n308), .A2(n344), .ZN(n311) );
  VHSR_OAI21_2 U346 ( .A1(n312), .A2(n310), .B(n311), .ZN(n309) );
  VHSR_OAI31_2 U347 ( .A1(n312), .A2(n311), .A3(n310), .B(n309), .ZN(n375) );
  VHSR_AD1_1 U348 ( .A(n315), .B(n314), .CI(n313), .CO(n305), .S(n379) );
  VHSR_IN_2 U349 ( .I(b[2]), .ZN(n318) );
  VHSR_NOR2_1 U350 ( .A1(n318), .A2(n322), .ZN(n337) );
  VHSR_IN_2 U351 ( .I(n337), .ZN(n327) );
  VHSR_NOR2_1 U352 ( .A1(n318), .A2(n326), .ZN(n317) );
  VHSR_OAI21_2 U353 ( .A1(n325), .A2(n322), .B(n317), .ZN(n316) );
  VHSR_OAI31_2 U354 ( .A1(n325), .A2(n317), .A3(n322), .B(n316), .ZN(n340) );
  VHSR_CLKNAND2_2 U355 ( .A1(b[3]), .A2(a[3]), .ZN(n336) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[1]), .A2(a[1]), .ZN(n406) );
  VHSR_AOI22_2 U357 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n323) );
  VHSR_OAI22_2 U358 ( .A1(n336), .A2(n406), .B1(n327), .B2(n323), .ZN(n324) );
  VHSR_OAI22_2 U359 ( .A1(n325), .A2(n403), .B1(n318), .B2(n401), .ZN(n389) );
  VHSR_IN_2 U360 ( .I(n406), .ZN(n321) );
  VHSR_NOR2_1 U361 ( .A1(n402), .A2(n322), .ZN(n320) );
  VHSR_OAI211_2 U362 ( .A1(n320), .A2(n321), .B(b[2]), .C(a[0]), .ZN(n319) );
  VHSR_OAI22_2 U363 ( .A1(n404), .A2(n322), .B1(n402), .B2(n326), .ZN(n388) );
  VHSR_AOI21_2 U364 ( .A1(n323), .A2(n327), .B(n324), .ZN(n342) );
  VHSR_CLKNAND2_2 U365 ( .A1(n343), .A2(n342), .ZN(n341) );
  VHSR_CLKNAND2_2 U366 ( .A1(n340), .A2(n339), .ZN(n334) );
  VHSR_AOI211_2 U367 ( .A1(n327), .A2(n334), .B(n326), .C(n325), .ZN(n382) );
  VHSR_AD1_1 U368 ( .A(n330), .B(n329), .CI(n328), .CO(n313), .S(n381) );
  VHSR_AD1_1 U369 ( .A(n333), .B(n332), .CI(n331), .CO(n328), .S(n385) );
  VHSR_IN_2 U370 ( .I(n334), .ZN(n338) );
  VHSR_CLKNAND2_2 U371 ( .A1(n338), .A2(n336), .ZN(n335) );
  VHSR_OAI31_2 U372 ( .A1(n337), .A2(n338), .A3(n336), .B(n335), .ZN(n384) );
  VHSR_IAO21_2 U373 ( .A1(n340), .A2(n339), .B(n338), .ZN(n387) );
  VHSR_OAI21_2 U374 ( .A1(n343), .A2(n342), .B(n341), .ZN(n400) );
  VHSR_NOR2_1 U375 ( .A1(n344), .A2(n403), .ZN(n345) );
  VHSR_AOI32_2 U376 ( .A1(b[0]), .A2(n346), .A3(a[4]), .B1(n345), .B2(n346), 
        .ZN(n399) );
  VHSR_AD1_1 U377 ( .A(n349), .B(n348), .CI(n347), .CO(n331), .S(n386) );
  VHSR_CLKNAND2_2 U378 ( .A1(a[7]), .A2(b[6]), .ZN(n353) );
  VHSR_AOI21_2 U379 ( .A1(a[6]), .A2(b[7]), .B(n353), .ZN(n352) );
  VHSR_AOI31_2 U380 ( .A1(a[6]), .A2(n353), .A3(b[7]), .B(n352), .ZN(n354) );
  VHSR_IN_2 U381 ( .I(n354), .ZN(n355) );
  VHSR_MAOI222_2 U382 ( .A(n358), .B(n356), .C(n355), .ZN(n365) );
  VHSR_OAI21_2 U383 ( .A1(n358), .A2(n357), .B(n365), .ZN(n362) );
  VHSR_CLKXOR2_2 U384 ( .A1(n363), .A2(n362), .Z(n359) );
  VHSR_CLKNAND2_2 U385 ( .A1(n360), .A2(n359), .ZN(n395) );
  VHSR_OAI21_2 U386 ( .A1(n360), .A2(n359), .B(n395), .ZN(n361) );
  VHSR_NOR2_1 U387 ( .A1(n363), .A2(n362), .ZN(n364) );
  VHSR_AND3_2 U388 ( .A1(n396), .A2(n366), .A3(n395), .Z(n367) );
  VHSR_NOR2_1 U389 ( .A1(n394), .A2(n367), .ZN(product[15]) );
  VHSR_AD1_1 U390 ( .A(n392), .B(n391), .CI(n390), .CO(n360), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U391 ( .A1(n394), .A2(n393), .ZN(n397) );
  VHSR_XOR3_2 U392 ( .A1(n397), .A2(n396), .A3(n395), .Z(product[14]) );
  VHSR_AOI21_2 U393 ( .A1(n400), .A2(n399), .B(n398), .ZN(product[4]) );
  VHSR_OAI22_2 U394 ( .A1(n404), .A2(n403), .B1(n402), .B2(n401), .ZN(
        product[1]) );
  VHSR_AOI22_2 U395 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n407) );
  VHSR_AOI21_2 U396 ( .A1(n407), .A2(n406), .B(n405), .ZN(product[2]) );
endmodule


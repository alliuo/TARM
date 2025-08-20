
module mul8_45 ( a, b, product );
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
         n403, n404, n405, n406, n407, n408, n409;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U205 ( .A1(n221), .B1(n253), .ZN(n240) );
  VHSR_INOR2_2 U206 ( .A1(n352), .B1(n351), .ZN(n364) );
  VHSR_NOR2_1 U207 ( .A1(n322), .A2(n324), .ZN(n335) );
  VHSR_NOR2_1 U208 ( .A1(n296), .A2(n300), .ZN(n295) );
  VHSR_NOR2_1 U209 ( .A1(n293), .A2(n294), .ZN(n351) );
  VHSR_NOR2_1 U210 ( .A1(n401), .A2(n400), .ZN(n399) );
  VHSR_NOR2_1 U211 ( .A1(n314), .A2(n348), .ZN(n379) );
  VHSR_IN_2 U212 ( .I(n362), .ZN(product[13]) );
  VHSR_INOR3_1 U213 ( .A1(n237), .B1(n323), .B2(n279), .ZN(n298) );
  VHSR_INOR2_1 U214 ( .A1(n366), .B1(n365), .ZN(n397) );
  VHSR_INAND2_1 U215 ( .A1(n357), .B1(n355), .ZN(n358) );
  VHSR_INOR2_1 U216 ( .A1(n379), .B1(n285), .ZN(n289) );
  VHSR_MOAI22_1 U217 ( .A1(n279), .A2(n405), .B1(a[6]), .B2(b[2]), .ZN(n225)
         );
  VHSR_AD1_1 U218 ( .A(n386), .B(n385), .CI(n384), .CO(n381), .S(product[6])
         );
  VHSR_AD1_1 U219 ( .A(n380), .B(n379), .CI(n378), .CO(n375), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U220 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(product[10])
         );
  VHSR_AD1_1 U221 ( .A(n390), .B(n406), .CI(n389), .CO(n347), .S(product[3])
         );
  VHSR_AD1_1 U222 ( .A(n388), .B(n387), .CI(n399), .CO(n384), .S(product[5])
         );
  VHSR_AD1_1 U223 ( .A(n383), .B(n382), .CI(n381), .CO(n378), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U224 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(product[9])
         );
  VHSR_AD1_1 U225 ( .A(n371), .B(n370), .CI(n369), .CO(n391), .S(product[11])
         );
  VHSR_IN_2 U226 ( .I(b[7]), .ZN(n281) );
  VHSR_IN_2 U227 ( .I(a[3]), .ZN(n326) );
  VHSR_IN_2 U228 ( .I(b[6]), .ZN(n282) );
  VHSR_IN_2 U229 ( .I(a[2]), .ZN(n324) );
  VHSR_OAI22_2 U230 ( .A1(n282), .A2(n326), .B1(n281), .B2(n324), .ZN(n242) );
  VHSR_NOR2_1 U231 ( .A1(n281), .A2(n324), .ZN(n215) );
  VHSR_IN_2 U232 ( .I(a[1]), .ZN(n402) );
  VHSR_NOR2_1 U233 ( .A1(n282), .A2(n402), .ZN(n214) );
  VHSR_IN_2 U234 ( .I(b[5]), .ZN(n312) );
  VHSR_AOI211_2 U235 ( .A1(b[4]), .A2(a[2]), .B(n312), .C(n326), .ZN(n220) );
  VHSR_OAI22_2 U236 ( .A1(n282), .A2(n324), .B1(n281), .B2(n402), .ZN(n219) );
  VHSR_AOI22_2 U237 ( .A1(n215), .A2(n214), .B1(n220), .B2(n219), .ZN(n221) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[4]), .A2(a[2]), .ZN(n265) );
  VHSR_IN_2 U239 ( .I(b[4]), .ZN(n348) );
  VHSR_IN_2 U240 ( .I(a[0]), .ZN(n404) );
  VHSR_OAI211_2 U241 ( .A1(n348), .A2(n404), .B(b[5]), .C(a[1]), .ZN(n264) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[6]), .A2(a[0]), .ZN(n263) );
  VHSR_MAOI222_2 U243 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_NOR4_2 U244 ( .A1(n348), .A2(n312), .A3(n402), .A4(n404), .ZN(n273) );
  VHSR_NAND4_2 U245 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_OAI22_2 U246 ( .A1(n348), .A2(n326), .B1(n312), .B2(n324), .ZN(n216) );
  VHSR_AND2_2 U247 ( .A1(n239), .A2(n216), .Z(n218) );
  VHSR_OAI22_2 U248 ( .A1(n282), .A2(n402), .B1(n281), .B2(n404), .ZN(n217) );
  VHSR_AND2_2 U249 ( .A1(n262), .A2(n258), .Z(n257) );
  VHSR_AD1_1 U250 ( .A(n273), .B(n218), .CI(n217), .CO(n252), .S(n258) );
  VHSR_NOR2_1 U251 ( .A1(n257), .A2(n252), .ZN(n255) );
  VHSR_OAI21_2 U252 ( .A1(n220), .A2(n219), .B(n221), .ZN(n256) );
  VHSR_NOR2_1 U253 ( .A1(n255), .A2(n256), .ZN(n253) );
  VHSR_CLKNAND2_2 U254 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U255 ( .A1(n242), .A2(n238), .ZN(n235) );
  VHSR_NOR3_2 U256 ( .A1(n281), .A2(n326), .A3(n235), .ZN(n299) );
  VHSR_AOI22_2 U257 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n247) );
  VHSR_IN_2 U258 ( .I(b[3]), .ZN(n323) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[2]), .A2(a[4]), .ZN(n269) );
  VHSR_IN_2 U260 ( .I(a[5]), .ZN(n310) );
  VHSR_NOR3_2 U261 ( .A1(n323), .A2(n269), .A3(n310), .ZN(n245) );
  VHSR_IN_2 U262 ( .I(a[7]), .ZN(n279) );
  VHSR_IN_2 U263 ( .I(b[1]), .ZN(n405) );
  VHSR_NOR2_1 U264 ( .A1(n279), .A2(n405), .ZN(n223) );
  VHSR_AND2_2 U265 ( .A1(a[6]), .A2(b[2]), .Z(n222) );
  VHSR_AOI211_2 U266 ( .A1(a[4]), .A2(b[2]), .B(n323), .C(n310), .ZN(n224) );
  VHSR_MAOI222_2 U267 ( .A(n223), .B(n222), .C(n224), .ZN(n234) );
  VHSR_OAI21_2 U268 ( .A1(n225), .A2(n224), .B(n234), .ZN(n226) );
  VHSR_IN_2 U269 ( .I(n226), .ZN(n250) );
  VHSR_IN_2 U270 ( .I(a[4]), .ZN(n314) );
  VHSR_IN_2 U271 ( .I(b[0]), .ZN(n403) );
  VHSR_NOR4_2 U272 ( .A1(n314), .A2(n310), .A3(n405), .A4(n403), .ZN(n271) );
  VHSR_AOI22_2 U273 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n227) );
  VHSR_NOR2_1 U274 ( .A1(n245), .A2(n227), .ZN(n229) );
  VHSR_AOI22_2 U275 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n231) );
  VHSR_IN_2 U276 ( .I(n231), .ZN(n228) );
  VHSR_MAOI222_2 U277 ( .A(n271), .B(n229), .C(n228), .ZN(n233) );
  VHSR_OAI211_2 U278 ( .A1(n314), .A2(n403), .B(a[5]), .C(b[1]), .ZN(n268) );
  VHSR_CLKNAND2_2 U279 ( .A1(a[6]), .A2(b[0]), .ZN(n267) );
  VHSR_MAOI222_2 U280 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_NOR2_1 U281 ( .A1(n271), .A2(n229), .ZN(n232) );
  VHSR_IN_2 U282 ( .I(n233), .ZN(n230) );
  VHSR_AOI21_2 U283 ( .A1(n232), .A2(n231), .B(n230), .ZN(n260) );
  VHSR_CLKNAND2_2 U284 ( .A1(n266), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U285 ( .A1(n233), .A2(n259), .ZN(n249) );
  VHSR_CLKNAND2_2 U286 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_CLKNAND2_2 U287 ( .A1(n234), .A2(n248), .ZN(n244) );
  VHSR_NOR2_1 U288 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_NOR2_1 U289 ( .A1(n247), .A2(n243), .ZN(n237) );
  VHSR_OAI32_2 U290 ( .A1(n299), .A2(n326), .A3(n281), .B1(n235), .B2(n299), 
        .ZN(n306) );
  VHSR_NOR2_1 U291 ( .A1(n323), .A2(n279), .ZN(n236) );
  VHSR_IAO21_2 U292 ( .A1(n237), .A2(n236), .B(n298), .ZN(n305) );
  VHSR_OAI21_2 U293 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U294 ( .A1(n242), .A2(n241), .ZN(n309) );
  VHSR_AOI21_2 U295 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U296 ( .A1(n247), .A2(n246), .ZN(n308) );
  VHSR_OAI21_2 U297 ( .A1(n250), .A2(n249), .B(n248), .ZN(n251) );
  VHSR_IN_2 U298 ( .I(n251), .ZN(n317) );
  VHSR_CLKNAND2_2 U299 ( .A1(n257), .A2(n252), .ZN(n254) );
  VHSR_AOI22_2 U300 ( .A1(n256), .A2(n255), .B1(n254), .B2(n253), .ZN(n316) );
  VHSR_IAO21_2 U301 ( .A1(n262), .A2(n258), .B(n257), .ZN(n332) );
  VHSR_OAI21_2 U302 ( .A1(n266), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U303 ( .I(n261), .ZN(n331) );
  VHSR_AOI31_2 U304 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n338) );
  VHSR_AOI31_2 U305 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n337) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[5]), .A2(b[0]), .ZN(n270) );
  VHSR_OAI32_2 U307 ( .A1(n271), .A2(n405), .A3(n314), .B1(n270), .B2(n271), 
        .ZN(n341) );
  VHSR_NOR2_1 U308 ( .A1(n403), .A2(n404), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U309 ( .A1(n379), .A2(product[0]), .ZN(n350) );
  VHSR_IN_2 U310 ( .I(n350), .ZN(n340) );
  VHSR_CLKNAND2_2 U311 ( .A1(b[5]), .A2(a[0]), .ZN(n272) );
  VHSR_OAI32_2 U312 ( .A1(n273), .A2(n402), .A3(n348), .B1(n272), .B2(n273), 
        .ZN(n339) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[6]), .A2(b[6]), .ZN(n367) );
  VHSR_IN_2 U314 ( .I(n367), .ZN(n394) );
  VHSR_NOR2_1 U315 ( .A1(n314), .A2(n282), .ZN(n286) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[5]), .A2(b[7]), .ZN(n275) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[6]), .A2(b[4]), .ZN(n278) );
  VHSR_IN_2 U318 ( .I(n278), .ZN(n287) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[7]), .A2(b[5]), .ZN(n274) );
  VHSR_OAI22_2 U320 ( .A1(n286), .A2(n275), .B1(n287), .B2(n274), .ZN(n277) );
  VHSR_OR2_2 U321 ( .A1(n286), .A2(n287), .Z(n301) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[5]), .A2(b[5]), .ZN(n285) );
  VHSR_CLKNAND2_2 U323 ( .A1(a[7]), .A2(b[7]), .ZN(n395) );
  VHSR_NOR3_2 U324 ( .A1(n301), .A2(n285), .A3(n395), .ZN(n276) );
  VHSR_AOI31_2 U325 ( .A1(b[6]), .A2(a[6]), .A3(n277), .B(n276), .ZN(n352) );
  VHSR_OAI21_2 U326 ( .A1(n394), .A2(n277), .B(n352), .ZN(n294) );
  VHSR_NOR3_2 U327 ( .A1(n279), .A2(n278), .A3(n312), .ZN(n359) );
  VHSR_AOI22_2 U328 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n280) );
  VHSR_NOR2_1 U329 ( .A1(n359), .A2(n280), .ZN(n290) );
  VHSR_NOR4_2 U330 ( .A1(n314), .A2(n310), .A3(n282), .A4(n281), .ZN(n357) );
  VHSR_AOI22_2 U331 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n283) );
  VHSR_NOR2_1 U332 ( .A1(n357), .A2(n283), .ZN(n288) );
  VHSR_IN_2 U333 ( .I(n284), .ZN(n296) );
  VHSR_NOR2_1 U334 ( .A1(n379), .A2(n285), .ZN(n302) );
  VHSR_AOI22_2 U335 ( .A1(n287), .A2(n286), .B1(n302), .B2(n301), .ZN(n300) );
  VHSR_AD1_1 U336 ( .A(n290), .B(n289), .CI(n288), .CO(n291), .S(n284) );
  VHSR_NOR2_1 U337 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U338 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_AOI22_2 U339 ( .A1(n294), .A2(n293), .B1(n292), .B2(n351), .ZN(n392) );
  VHSR_AOI21_2 U340 ( .A1(n300), .A2(n296), .B(n295), .ZN(n371) );
  VHSR_AD1_1 U341 ( .A(n299), .B(n298), .CI(n297), .CO(n393), .S(n370) );
  VHSR_OAI21_2 U342 ( .A1(n302), .A2(n301), .B(n300), .ZN(n303) );
  VHSR_IN_2 U343 ( .I(n303), .ZN(n374) );
  VHSR_AD1_1 U344 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n373) );
  VHSR_AD1_1 U345 ( .A(n309), .B(n308), .CI(n307), .CO(n304), .S(n377) );
  VHSR_NOR2_1 U346 ( .A1(n310), .A2(n348), .ZN(n313) );
  VHSR_OAI21_2 U347 ( .A1(n314), .A2(n312), .B(n313), .ZN(n311) );
  VHSR_OAI31_2 U348 ( .A1(n314), .A2(n313), .A3(n312), .B(n311), .ZN(n376) );
  VHSR_AD1_1 U349 ( .A(n317), .B(n316), .CI(n315), .CO(n307), .S(n380) );
  VHSR_IN_2 U350 ( .I(b[2]), .ZN(n322) );
  VHSR_NOR2_1 U351 ( .A1(n322), .A2(n326), .ZN(n319) );
  VHSR_OAI21_2 U352 ( .A1(n323), .A2(n324), .B(n319), .ZN(n318) );
  VHSR_OAI31_2 U353 ( .A1(n323), .A2(n319), .A3(n324), .B(n318), .ZN(n344) );
  VHSR_NOR2_1 U354 ( .A1(n405), .A2(n326), .ZN(n321) );
  VHSR_NOR2_1 U355 ( .A1(n323), .A2(n402), .ZN(n320) );
  VHSR_MAOI222_2 U356 ( .A(n335), .B(n321), .C(n320), .ZN(n328) );
  VHSR_OAI22_2 U357 ( .A1(n323), .A2(n404), .B1(n322), .B2(n402), .ZN(n390) );
  VHSR_CLKNAND2_2 U358 ( .A1(b[0]), .A2(a[2]), .ZN(n409) );
  VHSR_CLKNAND2_2 U359 ( .A1(b[2]), .A2(a[0]), .ZN(n408) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[1]), .A2(a[1]), .ZN(n407) );
  VHSR_MAOI222_2 U361 ( .A(n409), .B(n408), .C(n407), .ZN(n406) );
  VHSR_OAI22_2 U362 ( .A1(n405), .A2(n324), .B1(n403), .B2(n326), .ZN(n389) );
  VHSR_IN_2 U363 ( .I(n328), .ZN(n327) );
  VHSR_AOI21_2 U364 ( .A1(a[1]), .A2(b[3]), .B(n335), .ZN(n325) );
  VHSR_OAI32_2 U365 ( .A1(n327), .A2(n326), .A3(n405), .B1(n325), .B2(n327), 
        .ZN(n346) );
  VHSR_CLKNAND2_2 U366 ( .A1(n347), .A2(n346), .ZN(n345) );
  VHSR_CLKNAND2_2 U367 ( .A1(n328), .A2(n345), .ZN(n343) );
  VHSR_AND2_2 U368 ( .A1(n344), .A2(n343), .Z(n342) );
  VHSR_OAI211_2 U369 ( .A1(n335), .A2(n342), .B(a[3]), .C(b[3]), .ZN(n329) );
  VHSR_IN_2 U370 ( .I(n329), .ZN(n383) );
  VHSR_AD1_1 U371 ( .A(n332), .B(n331), .CI(n330), .CO(n315), .S(n382) );
  VHSR_CLKNAND2_2 U372 ( .A1(b[3]), .A2(a[3]), .ZN(n334) );
  VHSR_CLKNAND2_2 U373 ( .A1(n342), .A2(n334), .ZN(n333) );
  VHSR_OAI31_2 U374 ( .A1(n335), .A2(n342), .A3(n334), .B(n333), .ZN(n386) );
  VHSR_AD1_1 U375 ( .A(n338), .B(n337), .CI(n336), .CO(n330), .S(n385) );
  VHSR_AD1_1 U376 ( .A(n341), .B(n340), .CI(n339), .CO(n336), .S(n388) );
  VHSR_IAO21_2 U377 ( .A1(n344), .A2(n343), .B(n342), .ZN(n387) );
  VHSR_OAI21_2 U378 ( .A1(n347), .A2(n346), .B(n345), .ZN(n401) );
  VHSR_NOR2_1 U379 ( .A1(n348), .A2(n404), .ZN(n349) );
  VHSR_AOI32_2 U380 ( .A1(b[0]), .A2(n350), .A3(a[4]), .B1(n349), .B2(n350), 
        .ZN(n400) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[7]), .A2(b[6]), .ZN(n354) );
  VHSR_AOI21_2 U382 ( .A1(a[6]), .A2(b[7]), .B(n354), .ZN(n353) );
  VHSR_AOI31_2 U383 ( .A1(a[6]), .A2(n354), .A3(b[7]), .B(n353), .ZN(n355) );
  VHSR_IN_2 U384 ( .I(n355), .ZN(n356) );
  VHSR_MAOI222_2 U385 ( .A(n359), .B(n357), .C(n356), .ZN(n366) );
  VHSR_OAI21_2 U386 ( .A1(n359), .A2(n358), .B(n366), .ZN(n363) );
  VHSR_CLKXOR2_2 U387 ( .A1(n364), .A2(n363), .Z(n360) );
  VHSR_CLKNAND2_2 U388 ( .A1(n361), .A2(n360), .ZN(n396) );
  VHSR_OAI21_2 U389 ( .A1(n361), .A2(n360), .B(n396), .ZN(n362) );
  VHSR_NOR2_1 U390 ( .A1(n364), .A2(n363), .ZN(n365) );
  VHSR_AND3_2 U391 ( .A1(n397), .A2(n367), .A3(n396), .Z(n368) );
  VHSR_NOR2_1 U392 ( .A1(n395), .A2(n368), .ZN(product[15]) );
  VHSR_AD1_1 U393 ( .A(n393), .B(n392), .CI(n391), .CO(n361), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U394 ( .A1(n395), .A2(n394), .ZN(n398) );
  VHSR_XOR3_2 U395 ( .A1(n398), .A2(n397), .A3(n396), .Z(product[14]) );
  VHSR_AOI21_2 U396 ( .A1(n401), .A2(n400), .B(n399), .ZN(product[4]) );
  VHSR_OAI22_2 U397 ( .A1(n405), .A2(n404), .B1(n403), .B2(n402), .ZN(
        product[1]) );
  VHSR_AOI31_2 U398 ( .A1(n409), .A2(n408), .A3(n407), .B(n406), .ZN(
        product[2]) );
endmodule



module mul8_111 ( a, b, product );
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
         n392, n393, n394, n395, n396, n397, n398, n399, n400, n401;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_IN_2 U205 ( .I(n242), .ZN(n216) );
  VHSR_INAND3_2 U206 ( .A1(n264), .B1(a[5]), .B2(b[3]), .ZN(n214) );
  VHSR_NOR2_1 U207 ( .A1(n312), .A2(n280), .ZN(n264) );
  VHSR_INOR2_2 U208 ( .A1(n224), .B1(n250), .ZN(n243) );
  VHSR_INOR2_2 U209 ( .A1(n222), .B1(n253), .ZN(n252) );
  VHSR_NOR2_1 U210 ( .A1(n342), .A2(n341), .ZN(n354) );
  VHSR_INAND2_2 U211 ( .A1(n318), .B1(n337), .ZN(n333) );
  VHSR_NOR2_1 U212 ( .A1(n234), .A2(n233), .ZN(n294) );
  VHSR_IOA21_2 U213 ( .A1(n315), .A2(n314), .B(n313), .ZN(n399) );
  VHSR_INOR2_2 U214 ( .A1(n356), .B1(n355), .ZN(n387) );
  VHSR_IN_2 U215 ( .I(n352), .ZN(product[13]) );
  VHSR_IOA21_1 U216 ( .A1(n391), .A2(n390), .B(n389), .ZN(n394) );
  VHSR_MOAI22_1 U217 ( .A1(n275), .A2(n316), .B1(b[4]), .B2(a[3]), .ZN(n226)
         );
  VHSR_NOR2_2 U218 ( .A1(n270), .A2(n279), .ZN(n384) );
  VHSR_AD1_1 U219 ( .A(n373), .B(n372), .CI(n371), .CO(n368), .S(product[6])
         );
  VHSR_AD1_1 U220 ( .A(n364), .B(n363), .CI(n362), .CO(n359), .S(product[9])
         );
  VHSR_AD1_1 U221 ( .A(n377), .B(n399), .CI(n376), .CO(n339), .S(product[3])
         );
  VHSR_AD1_1 U222 ( .A(n375), .B(n374), .CI(n392), .CO(n371), .S(product[5])
         );
  VHSR_AD1_1 U223 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U224 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U225 ( .A(n361), .B(n360), .CI(n359), .CO(n378), .S(product[10])
         );
  VHSR_CLKNAND2_2 U226 ( .A1(b[3]), .A2(a[7]), .ZN(n234) );
  VHSR_IN_2 U227 ( .I(b[3]), .ZN(n319) );
  VHSR_IN_2 U228 ( .I(a[6]), .ZN(n270) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n276) );
  VHSR_IN_2 U230 ( .I(b[2]), .ZN(n312) );
  VHSR_OAI22_2 U231 ( .A1(n319), .A2(n270), .B1(n276), .B2(n312), .ZN(n245) );
  VHSR_IN_2 U232 ( .I(b[1]), .ZN(n398) );
  VHSR_IN_2 U233 ( .I(a[4]), .ZN(n280) );
  VHSR_OAI21_2 U234 ( .A1(n398), .A2(n276), .B(n214), .ZN(n223) );
  VHSR_IN_2 U235 ( .I(a[5]), .ZN(n281) );
  VHSR_NOR4_2 U236 ( .A1(n264), .A2(n281), .A3(n234), .A4(n398), .ZN(n215) );
  VHSR_AOI31_2 U237 ( .A1(b[2]), .A2(a[6]), .A3(n223), .B(n215), .ZN(n224) );
  VHSR_NOR2_1 U238 ( .A1(n270), .A2(n398), .ZN(n219) );
  VHSR_IN_2 U239 ( .I(b[0]), .ZN(n396) );
  VHSR_NOR4_2 U240 ( .A1(n281), .A2(n280), .A3(n398), .A4(n396), .ZN(n269) );
  VHSR_CLKNAND2_2 U241 ( .A1(b[2]), .A2(a[5]), .ZN(n218) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[3]), .A2(a[4]), .ZN(n217) );
  VHSR_NAND4_2 U243 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n242) );
  VHSR_AOI21_2 U244 ( .A1(n218), .A2(n217), .B(n216), .ZN(n220) );
  VHSR_MAOI222_2 U245 ( .A(n219), .B(n269), .C(n220), .ZN(n222) );
  VHSR_AOI211_2 U246 ( .A1(a[4]), .A2(b[0]), .B(n281), .C(n398), .ZN(n263) );
  VHSR_AOI21_2 U247 ( .A1(n276), .A2(n270), .B(n396), .ZN(n262) );
  VHSR_MAOI222_2 U248 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_OR2_2 U249 ( .A1(n269), .A2(n220), .Z(n221) );
  VHSR_AOI32_2 U250 ( .A1(b[1]), .A2(n222), .A3(a[6]), .B1(n221), .B2(n222), 
        .ZN(n254) );
  VHSR_NOR2_1 U251 ( .A1(n261), .A2(n254), .ZN(n253) );
  VHSR_AOI32_2 U252 ( .A1(b[2]), .A2(n224), .A3(a[6]), .B1(n223), .B2(n224), 
        .ZN(n251) );
  VHSR_NOR2_1 U253 ( .A1(n252), .A2(n251), .ZN(n250) );
  VHSR_CLKNAND2_2 U254 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U255 ( .A1(n245), .A2(n241), .ZN(n233) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n320) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n279) );
  VHSR_IN_2 U259 ( .I(a[2]), .ZN(n316) );
  VHSR_OAI22_2 U260 ( .A1(n279), .A2(n320), .B1(n278), .B2(n316), .ZN(n240) );
  VHSR_AOI22_2 U261 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n231) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[4]), .A2(a[2]), .ZN(n260) );
  VHSR_NAND3_2 U263 ( .A1(a[3]), .A2(b[5]), .A3(n260), .ZN(n230) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[7]), .A2(a[2]), .ZN(n225) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[1]), .ZN(n227) );
  VHSR_OAI22_2 U266 ( .A1(n231), .A2(n230), .B1(n225), .B2(n227), .ZN(n232) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[4]), .A2(a[0]), .ZN(n390) );
  VHSR_NAND3_2 U268 ( .A1(a[1]), .A2(b[5]), .A3(n390), .ZN(n259) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[0]), .ZN(n258) );
  VHSR_MAOI222_2 U270 ( .A(n260), .B(n259), .C(n258), .ZN(n257) );
  VHSR_NAND4_2 U271 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_IN_2 U272 ( .I(b[5]), .ZN(n275) );
  VHSR_AND2_2 U273 ( .A1(n237), .A2(n226), .Z(n229) );
  VHSR_IN_2 U274 ( .I(a[0]), .ZN(n397) );
  VHSR_OAI21_2 U275 ( .A1(n278), .A2(n397), .B(n227), .ZN(n228) );
  VHSR_IN_2 U276 ( .I(a[1]), .ZN(n395) );
  VHSR_NOR3_2 U277 ( .A1(n275), .A2(n395), .A3(n390), .ZN(n267) );
  VHSR_AND2_2 U278 ( .A1(n257), .A2(n256), .Z(n255) );
  VHSR_AD1_1 U279 ( .A(n229), .B(n228), .CI(n267), .CO(n246), .S(n256) );
  VHSR_AOI21_2 U280 ( .A1(n231), .A2(n230), .B(n232), .ZN(n249) );
  VHSR_OAI32_2 U281 ( .A1(n232), .A2(n255), .A3(n246), .B1(n249), .B2(n232), 
        .ZN(n238) );
  VHSR_CLKNAND2_2 U282 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U283 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U284 ( .A1(n278), .A2(n320), .A3(n235), .ZN(n293) );
  VHSR_AOI21_2 U285 ( .A1(n234), .A2(n233), .B(n294), .ZN(n297) );
  VHSR_OAI32_2 U286 ( .A1(n293), .A2(n320), .A3(n278), .B1(n235), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U287 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U288 ( .A1(n240), .A2(n239), .ZN(n304) );
  VHSR_OAI21_2 U289 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U290 ( .A1(n245), .A2(n244), .ZN(n303) );
  VHSR_NOR2_1 U291 ( .A1(n255), .A2(n246), .ZN(n248) );
  VHSR_AOI22_2 U292 ( .A1(n255), .A2(n246), .B1(n249), .B2(n248), .ZN(n247) );
  VHSR_OAI21_2 U293 ( .A1(n249), .A2(n248), .B(n247), .ZN(n309) );
  VHSR_AOI21_2 U294 ( .A1(n252), .A2(n251), .B(n250), .ZN(n308) );
  VHSR_AOI21_2 U295 ( .A1(n261), .A2(n254), .B(n253), .ZN(n323) );
  VHSR_IAO21_2 U296 ( .A1(n257), .A2(n256), .B(n255), .ZN(n322) );
  VHSR_AOI31_2 U297 ( .A1(n260), .A2(n259), .A3(n258), .B(n257), .ZN(n331) );
  VHSR_OAI31_2 U298 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n265) );
  VHSR_IN_2 U299 ( .I(n265), .ZN(n330) );
  VHSR_AOI22_2 U300 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n266) );
  VHSR_NOR2_1 U301 ( .A1(n267), .A2(n266), .ZN(n336) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[4]), .A2(b[4]), .ZN(n283) );
  VHSR_IN_2 U303 ( .I(n283), .ZN(n366) );
  VHSR_NOR2_1 U304 ( .A1(n396), .A2(n397), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U305 ( .A1(n366), .A2(product[0]), .ZN(n389) );
  VHSR_IN_2 U306 ( .I(n389), .ZN(n340) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[4]), .A2(b[1]), .ZN(n268) );
  VHSR_OAI32_2 U308 ( .A1(n269), .A2(n396), .A3(n281), .B1(n268), .B2(n269), 
        .ZN(n335) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[4]), .A2(b[6]), .ZN(n300) );
  VHSR_NAND3_2 U310 ( .A1(b[7]), .A2(a[5]), .A3(n300), .ZN(n272) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[4]), .ZN(n301) );
  VHSR_NAND3_2 U312 ( .A1(a[7]), .A2(b[5]), .A3(n301), .ZN(n271) );
  VHSR_CLKNAND2_2 U313 ( .A1(n272), .A2(n271), .ZN(n274) );
  VHSR_IN_2 U314 ( .I(n384), .ZN(n357) );
  VHSR_MAOI222_2 U315 ( .A(n357), .B(n272), .C(n271), .ZN(n341) );
  VHSR_IN_2 U316 ( .I(n341), .ZN(n273) );
  VHSR_OAI21_2 U317 ( .A1(n384), .A2(n274), .B(n273), .ZN(n289) );
  VHSR_NOR3_2 U318 ( .A1(n281), .A2(n275), .A3(n283), .ZN(n305) );
  VHSR_NOR3_2 U319 ( .A1(n276), .A2(n301), .A3(n275), .ZN(n349) );
  VHSR_AOI22_2 U320 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n277) );
  VHSR_NOR2_1 U321 ( .A1(n349), .A2(n277), .ZN(n285) );
  VHSR_NOR4_2 U322 ( .A1(n281), .A2(n280), .A3(n279), .A4(n278), .ZN(n347) );
  VHSR_AOI22_2 U323 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n282) );
  VHSR_NOR2_1 U324 ( .A1(n347), .A2(n282), .ZN(n284) );
  VHSR_NAND3_2 U325 ( .A1(b[5]), .A2(a[5]), .A3(n283), .ZN(n299) );
  VHSR_MAOI222_2 U326 ( .A(n301), .B(n300), .C(n299), .ZN(n298) );
  VHSR_AND2_2 U327 ( .A1(n291), .A2(n298), .Z(n290) );
  VHSR_AD1_1 U328 ( .A(n305), .B(n285), .CI(n284), .CO(n286), .S(n291) );
  VHSR_NOR2_1 U329 ( .A1(n290), .A2(n286), .ZN(n288) );
  VHSR_CLKNAND2_2 U330 ( .A1(n290), .A2(n286), .ZN(n287) );
  VHSR_NOR2_1 U331 ( .A1(n288), .A2(n289), .ZN(n342) );
  VHSR_AOI22_2 U332 ( .A1(n289), .A2(n288), .B1(n287), .B2(n342), .ZN(n382) );
  VHSR_IAO21_2 U333 ( .A1(n291), .A2(n298), .B(n290), .ZN(n380) );
  VHSR_AD1_1 U334 ( .A(n294), .B(n293), .CI(n292), .CO(n383), .S(n379) );
  VHSR_AD1_1 U335 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n361) );
  VHSR_AOI31_2 U336 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n360) );
  VHSR_AD1_1 U337 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n364) );
  VHSR_AOI22_2 U338 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n306) );
  VHSR_NOR2_1 U339 ( .A1(n306), .A2(n305), .ZN(n363) );
  VHSR_AD1_1 U340 ( .A(n309), .B(n308), .CI(n307), .CO(n302), .S(n367) );
  VHSR_CLKNAND2_2 U341 ( .A1(b[2]), .A2(a[2]), .ZN(n324) );
  VHSR_NOR2_1 U342 ( .A1(n312), .A2(n320), .ZN(n311) );
  VHSR_OAI21_2 U343 ( .A1(n319), .A2(n316), .B(n311), .ZN(n310) );
  VHSR_OAI31_2 U344 ( .A1(n319), .A2(n311), .A3(n316), .B(n310), .ZN(n334) );
  VHSR_AOI22_2 U345 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n317) );
  VHSR_CLKNAND2_2 U346 ( .A1(b[3]), .A2(a[3]), .ZN(n327) );
  VHSR_NOR2_1 U347 ( .A1(n398), .A2(n395), .ZN(n315) );
  VHSR_IN_2 U348 ( .I(n315), .ZN(n400) );
  VHSR_OAI22_2 U349 ( .A1(n324), .A2(n317), .B1(n327), .B2(n400), .ZN(n318) );
  VHSR_OAI22_2 U350 ( .A1(n319), .A2(n397), .B1(n312), .B2(n395), .ZN(n377) );
  VHSR_NOR2_1 U351 ( .A1(n396), .A2(n316), .ZN(n314) );
  VHSR_OAI211_2 U352 ( .A1(n314), .A2(n315), .B(b[2]), .C(a[0]), .ZN(n313) );
  VHSR_OAI22_2 U353 ( .A1(n398), .A2(n316), .B1(n396), .B2(n320), .ZN(n376) );
  VHSR_AOI21_2 U354 ( .A1(n317), .A2(n324), .B(n318), .ZN(n338) );
  VHSR_CLKNAND2_2 U355 ( .A1(n339), .A2(n338), .ZN(n337) );
  VHSR_CLKNAND2_2 U356 ( .A1(n334), .A2(n333), .ZN(n325) );
  VHSR_AOI211_2 U357 ( .A1(n324), .A2(n325), .B(n320), .C(n319), .ZN(n370) );
  VHSR_AD1_1 U358 ( .A(n323), .B(n322), .CI(n321), .CO(n307), .S(n369) );
  VHSR_IN_2 U359 ( .I(n324), .ZN(n328) );
  VHSR_IN_2 U360 ( .I(n325), .ZN(n332) );
  VHSR_CLKNAND2_2 U361 ( .A1(n332), .A2(n327), .ZN(n326) );
  VHSR_OAI31_2 U362 ( .A1(n328), .A2(n332), .A3(n327), .B(n326), .ZN(n373) );
  VHSR_AD1_1 U363 ( .A(n331), .B(n330), .CI(n329), .CO(n321), .S(n372) );
  VHSR_IAO21_2 U364 ( .A1(n334), .A2(n333), .B(n332), .ZN(n375) );
  VHSR_AD1_1 U365 ( .A(n336), .B(n340), .CI(n335), .CO(n329), .S(n374) );
  VHSR_CLKNAND2_2 U366 ( .A1(a[4]), .A2(b[0]), .ZN(n391) );
  VHSR_OAI21_2 U367 ( .A1(n339), .A2(n338), .B(n337), .ZN(n393) );
  VHSR_AOI211_2 U368 ( .A1(n391), .A2(n390), .B(n340), .C(n393), .ZN(n392) );
  VHSR_CLKNAND2_2 U369 ( .A1(a[6]), .A2(b[7]), .ZN(n344) );
  VHSR_AOI21_2 U370 ( .A1(a[7]), .A2(b[6]), .B(n344), .ZN(n343) );
  VHSR_AOI31_2 U371 ( .A1(a[7]), .A2(n344), .A3(b[6]), .B(n343), .ZN(n345) );
  VHSR_IN_2 U372 ( .I(n345), .ZN(n346) );
  VHSR_OR2_2 U373 ( .A1(n347), .A2(n346), .Z(n348) );
  VHSR_MAOI222_2 U374 ( .A(n349), .B(n347), .C(n346), .ZN(n356) );
  VHSR_OAI21_2 U375 ( .A1(n349), .A2(n348), .B(n356), .ZN(n353) );
  VHSR_CLKXOR2_2 U376 ( .A1(n354), .A2(n353), .Z(n350) );
  VHSR_CLKNAND2_2 U377 ( .A1(n351), .A2(n350), .ZN(n386) );
  VHSR_OAI21_2 U378 ( .A1(n351), .A2(n350), .B(n386), .ZN(n352) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[7]), .A2(b[7]), .ZN(n385) );
  VHSR_NOR2_1 U380 ( .A1(n354), .A2(n353), .ZN(n355) );
  VHSR_AND3_2 U381 ( .A1(n387), .A2(n357), .A3(n386), .Z(n358) );
  VHSR_NOR2_1 U382 ( .A1(n385), .A2(n358), .ZN(product[15]) );
  VHSR_AD1_1 U383 ( .A(n380), .B(n379), .CI(n378), .CO(n381), .S(product[11])
         );
  VHSR_AD1_1 U384 ( .A(n383), .B(n382), .CI(n381), .CO(n351), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U385 ( .A1(n385), .A2(n384), .ZN(n388) );
  VHSR_XOR3_2 U386 ( .A1(n388), .A2(n387), .A3(n386), .Z(product[14]) );
  VHSR_AOI21_2 U387 ( .A1(n394), .A2(n393), .B(n392), .ZN(product[4]) );
  VHSR_OAI22_2 U388 ( .A1(n398), .A2(n397), .B1(n396), .B2(n395), .ZN(
        product[1]) );
  VHSR_AOI22_2 U389 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n401) );
  VHSR_AOI21_2 U390 ( .A1(n401), .A2(n400), .B(n399), .ZN(product[2]) );
endmodule


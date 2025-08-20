
module mul8_130 ( a, b, product );
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
         n403, n404, n405, n406, n407, n408;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U206 ( .A1(n244), .B1(n226), .ZN(n227) );
  VHSR_INOR2_2 U207 ( .A1(n234), .B1(n252), .ZN(n245) );
  VHSR_INAND2_2 U208 ( .A1(n361), .B1(n359), .ZN(n362) );
  VHSR_INOR2_2 U209 ( .A1(n230), .B1(n257), .ZN(n254) );
  VHSR_NOR2_1 U210 ( .A1(n272), .A2(n309), .ZN(n286) );
  VHSR_INAND2_2 U211 ( .A1(n331), .B1(n345), .ZN(n343) );
  VHSR_NOR2_1 U212 ( .A1(n295), .A2(n299), .ZN(n294) );
  VHSR_NOR2_1 U213 ( .A1(n237), .A2(n236), .ZN(n297) );
  VHSR_IOA21_2 U214 ( .A1(n327), .A2(n326), .B(n325), .ZN(n406) );
  VHSR_INOR2_2 U215 ( .A1(n370), .B1(n369), .ZN(n401) );
  VHSR_IN_2 U216 ( .I(n366), .ZN(product[13]) );
  VHSR_CLKN_1 U217 ( .I(n371), .ZN(n372) );
  VHSR_INAND3_1 U218 ( .A1(n398), .B1(n401), .B2(n400), .ZN(n371) );
  VHSR_INOR2_1 U219 ( .A1(n356), .B1(n355), .ZN(n368) );
  VHSR_NOR2_2 U220 ( .A1(n405), .A2(n404), .ZN(n403) );
  VHSR_INOR3_1 U221 ( .A1(n286), .B1(n277), .B2(n312), .ZN(n363) );
  VHSR_AD1_1 U222 ( .A(n390), .B(n389), .CI(n388), .CO(n385), .S(product[6])
         );
  VHSR_AD1_1 U223 ( .A(n384), .B(n383), .CI(n382), .CO(n379), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U224 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(product[10])
         );
  VHSR_AD1_1 U225 ( .A(n394), .B(n406), .CI(n393), .CO(n347), .S(product[3])
         );
  VHSR_AD1_1 U226 ( .A(n392), .B(n403), .CI(n391), .CO(n388), .S(product[5])
         );
  VHSR_AD1_1 U227 ( .A(n387), .B(n386), .CI(n385), .CO(n382), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U228 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(product[9])
         );
  VHSR_AD1_1 U229 ( .A(n375), .B(n374), .CI(n373), .CO(n395), .S(product[11])
         );
  VHSR_IN_2 U230 ( .I(b[1]), .ZN(n329) );
  VHSR_IN_2 U231 ( .I(a[0]), .ZN(n324) );
  VHSR_NOR2_1 U232 ( .A1(n329), .A2(n324), .ZN(product[0]) );
  VHSR_IN_2 U233 ( .I(b[0]), .ZN(n348) );
  VHSR_IN_2 U234 ( .I(a[1]), .ZN(n322) );
  VHSR_NOR2_1 U235 ( .A1(n348), .A2(n322), .ZN(product[1]) );
  VHSR_IN_2 U236 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U237 ( .I(a[3]), .ZN(n333) );
  VHSR_IN_2 U238 ( .I(b[6]), .ZN(n281) );
  VHSR_IN_2 U239 ( .I(a[2]), .ZN(n328) );
  VHSR_OAI22_2 U240 ( .A1(n281), .A2(n333), .B1(n280), .B2(n328), .ZN(n242) );
  VHSR_AOI22_2 U241 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n220) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[4]), .A2(a[2]), .ZN(n262) );
  VHSR_NAND3_2 U243 ( .A1(a[3]), .A2(b[5]), .A3(n262), .ZN(n219) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[7]), .A2(a[2]), .ZN(n214) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[6]), .A2(a[1]), .ZN(n216) );
  VHSR_OAI22_2 U246 ( .A1(n220), .A2(n219), .B1(n214), .B2(n216), .ZN(n221) );
  VHSR_CLKNAND2_2 U247 ( .A1(b[6]), .A2(a[0]), .ZN(n261) );
  VHSR_IN_2 U248 ( .I(b[4]), .ZN(n309) );
  VHSR_OAI211_2 U249 ( .A1(n309), .A2(n324), .B(b[5]), .C(a[1]), .ZN(n260) );
  VHSR_MAOI222_2 U250 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_IN_2 U251 ( .I(b[5]), .ZN(n312) );
  VHSR_NOR4_2 U252 ( .A1(n309), .A2(n312), .A3(n322), .A4(n324), .ZN(n269) );
  VHSR_NAND4_2 U253 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_OAI22_2 U254 ( .A1(n309), .A2(n333), .B1(n312), .B2(n328), .ZN(n215) );
  VHSR_AND2_2 U255 ( .A1(n239), .A2(n215), .Z(n218) );
  VHSR_OAI21_2 U256 ( .A1(n280), .A2(n324), .B(n216), .ZN(n217) );
  VHSR_AND2_2 U257 ( .A1(n259), .A2(n256), .Z(n255) );
  VHSR_AD1_1 U258 ( .A(n269), .B(n218), .CI(n217), .CO(n248), .S(n256) );
  VHSR_AOI21_2 U259 ( .A1(n220), .A2(n219), .B(n221), .ZN(n251) );
  VHSR_OAI32_2 U260 ( .A1(n221), .A2(n255), .A3(n248), .B1(n251), .B2(n221), 
        .ZN(n240) );
  VHSR_CLKNAND2_2 U261 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U262 ( .A1(n242), .A2(n238), .ZN(n235) );
  VHSR_NOR3_2 U263 ( .A1(n280), .A2(n333), .A3(n235), .ZN(n298) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[3]), .A2(a[7]), .ZN(n237) );
  VHSR_IN_2 U265 ( .I(b[3]), .ZN(n332) );
  VHSR_IN_2 U266 ( .I(a[6]), .ZN(n272) );
  VHSR_IN_2 U267 ( .I(a[7]), .ZN(n277) );
  VHSR_IN_2 U268 ( .I(b[2]), .ZN(n323) );
  VHSR_OAI22_2 U269 ( .A1(n332), .A2(n272), .B1(n277), .B2(n323), .ZN(n247) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[2]), .A2(a[4]), .ZN(n225) );
  VHSR_CLKNAND2_2 U271 ( .A1(a[6]), .A2(b[1]), .ZN(n231) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[3]), .A2(a[6]), .ZN(n222) );
  VHSR_OAI22_2 U273 ( .A1(n237), .A2(n231), .B1(n222), .B2(n323), .ZN(n224) );
  VHSR_NOR3_2 U274 ( .A1(n277), .A2(n323), .A3(n231), .ZN(n223) );
  VHSR_AOI31_2 U275 ( .A1(a[5]), .A2(n225), .A3(n224), .B(n223), .ZN(n234) );
  VHSR_IN_2 U276 ( .I(n231), .ZN(n229) );
  VHSR_IN_2 U277 ( .I(a[4]), .ZN(n349) );
  VHSR_IN_2 U278 ( .I(a[5]), .ZN(n310) );
  VHSR_NOR4_2 U279 ( .A1(n349), .A2(n310), .A3(n329), .A4(n348), .ZN(n271) );
  VHSR_IN_2 U280 ( .I(n225), .ZN(n266) );
  VHSR_NAND3_2 U281 ( .A1(b[3]), .A2(n266), .A3(a[5]), .ZN(n244) );
  VHSR_AOI22_2 U282 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n226) );
  VHSR_MAOI222_2 U283 ( .A(n229), .B(n271), .C(n227), .ZN(n230) );
  VHSR_AOI211_2 U284 ( .A1(a[4]), .A2(b[0]), .B(n310), .C(n329), .ZN(n265) );
  VHSR_NOR2_1 U285 ( .A1(n272), .A2(n348), .ZN(n264) );
  VHSR_MAOI222_2 U286 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_OR2_2 U287 ( .A1(n271), .A2(n227), .Z(n228) );
  VHSR_OAI21_2 U288 ( .A1(n229), .A2(n228), .B(n230), .ZN(n258) );
  VHSR_NOR2_1 U289 ( .A1(n263), .A2(n258), .ZN(n257) );
  VHSR_CLKNAND2_2 U290 ( .A1(b[3]), .A2(a[5]), .ZN(n232) );
  VHSR_OAI22_2 U291 ( .A1(n266), .A2(n232), .B1(n277), .B2(n231), .ZN(n233) );
  VHSR_AOI32_2 U292 ( .A1(b[2]), .A2(n234), .A3(a[6]), .B1(n233), .B2(n234), 
        .ZN(n253) );
  VHSR_NOR2_1 U293 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U294 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_CLKNAND2_2 U295 ( .A1(n247), .A2(n243), .ZN(n236) );
  VHSR_OAI32_2 U296 ( .A1(n298), .A2(n333), .A3(n280), .B1(n235), .B2(n298), 
        .ZN(n305) );
  VHSR_AOI21_2 U297 ( .A1(n237), .A2(n236), .B(n297), .ZN(n304) );
  VHSR_OAI21_2 U298 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U299 ( .A1(n242), .A2(n241), .ZN(n308) );
  VHSR_OAI21_2 U300 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U301 ( .A1(n247), .A2(n246), .ZN(n307) );
  VHSR_NOR2_1 U302 ( .A1(n255), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U303 ( .A1(n255), .A2(n248), .B1(n251), .B2(n250), .ZN(n249) );
  VHSR_OAI21_2 U304 ( .A1(n251), .A2(n250), .B(n249), .ZN(n316) );
  VHSR_AOI21_2 U305 ( .A1(n254), .A2(n253), .B(n252), .ZN(n315) );
  VHSR_IAO21_2 U306 ( .A1(n259), .A2(n256), .B(n255), .ZN(n319) );
  VHSR_AOI21_2 U307 ( .A1(n263), .A2(n258), .B(n257), .ZN(n318) );
  VHSR_AOI31_2 U308 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n337) );
  VHSR_OAI31_2 U309 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n267) );
  VHSR_IN_2 U310 ( .I(n267), .ZN(n336) );
  VHSR_CLKNAND2_2 U311 ( .A1(b[5]), .A2(a[0]), .ZN(n268) );
  VHSR_OAI32_2 U312 ( .A1(n269), .A2(n322), .A3(n309), .B1(n268), .B2(n269), 
        .ZN(n354) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[4]), .A2(b[4]), .ZN(n279) );
  VHSR_IN_2 U314 ( .I(n279), .ZN(n383) );
  VHSR_NAND3_2 U315 ( .A1(b[0]), .A2(n383), .A3(a[0]), .ZN(n351) );
  VHSR_IN_2 U316 ( .I(n351), .ZN(n353) );
  VHSR_AOI22_2 U317 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n270) );
  VHSR_NOR2_1 U318 ( .A1(n271), .A2(n270), .ZN(n352) );
  VHSR_NOR2_1 U319 ( .A1(n272), .A2(n281), .ZN(n398) );
  VHSR_NOR2_1 U320 ( .A1(n349), .A2(n281), .ZN(n285) );
  VHSR_CLKNAND2_2 U321 ( .A1(a[5]), .A2(b[7]), .ZN(n274) );
  VHSR_CLKNAND2_2 U322 ( .A1(a[7]), .A2(b[5]), .ZN(n273) );
  VHSR_OAI22_2 U323 ( .A1(n285), .A2(n274), .B1(n286), .B2(n273), .ZN(n276) );
  VHSR_OR2_2 U324 ( .A1(n285), .A2(n286), .Z(n300) );
  VHSR_CLKNAND2_2 U325 ( .A1(a[5]), .A2(b[5]), .ZN(n284) );
  VHSR_CLKNAND2_2 U326 ( .A1(a[7]), .A2(b[7]), .ZN(n399) );
  VHSR_NOR3_2 U327 ( .A1(n300), .A2(n284), .A3(n399), .ZN(n275) );
  VHSR_AOI31_2 U328 ( .A1(b[6]), .A2(a[6]), .A3(n276), .B(n275), .ZN(n356) );
  VHSR_OAI21_2 U329 ( .A1(n398), .A2(n276), .B(n356), .ZN(n293) );
  VHSR_AOI22_2 U330 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n278) );
  VHSR_NOR2_1 U331 ( .A1(n363), .A2(n278), .ZN(n289) );
  VHSR_NOR2_1 U332 ( .A1(n284), .A2(n279), .ZN(n288) );
  VHSR_NOR4_2 U333 ( .A1(n349), .A2(n310), .A3(n281), .A4(n280), .ZN(n361) );
  VHSR_AOI22_2 U334 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n282) );
  VHSR_NOR2_1 U335 ( .A1(n361), .A2(n282), .ZN(n287) );
  VHSR_IN_2 U336 ( .I(n283), .ZN(n295) );
  VHSR_NOR2_1 U337 ( .A1(n383), .A2(n284), .ZN(n301) );
  VHSR_AOI22_2 U338 ( .A1(n286), .A2(n285), .B1(n301), .B2(n300), .ZN(n299) );
  VHSR_AD1_1 U339 ( .A(n289), .B(n288), .CI(n287), .CO(n290), .S(n283) );
  VHSR_NOR2_1 U340 ( .A1(n294), .A2(n290), .ZN(n292) );
  VHSR_CLKNAND2_2 U341 ( .A1(n294), .A2(n290), .ZN(n291) );
  VHSR_NOR2_1 U342 ( .A1(n292), .A2(n293), .ZN(n355) );
  VHSR_AOI22_2 U343 ( .A1(n293), .A2(n292), .B1(n291), .B2(n355), .ZN(n396) );
  VHSR_AOI21_2 U344 ( .A1(n299), .A2(n295), .B(n294), .ZN(n375) );
  VHSR_AD1_1 U345 ( .A(n298), .B(n297), .CI(n296), .CO(n397), .S(n374) );
  VHSR_OAI21_2 U346 ( .A1(n301), .A2(n300), .B(n299), .ZN(n302) );
  VHSR_IN_2 U347 ( .I(n302), .ZN(n378) );
  VHSR_AD1_1 U348 ( .A(n305), .B(n304), .CI(n303), .CO(n296), .S(n377) );
  VHSR_AD1_1 U349 ( .A(n308), .B(n307), .CI(n306), .CO(n303), .S(n381) );
  VHSR_NOR2_1 U350 ( .A1(n310), .A2(n309), .ZN(n313) );
  VHSR_OAI21_2 U351 ( .A1(n349), .A2(n312), .B(n313), .ZN(n311) );
  VHSR_OAI31_2 U352 ( .A1(n349), .A2(n313), .A3(n312), .B(n311), .ZN(n380) );
  VHSR_AD1_1 U353 ( .A(n316), .B(n315), .CI(n314), .CO(n306), .S(n384) );
  VHSR_AD1_1 U354 ( .A(n319), .B(n318), .CI(n317), .CO(n314), .S(n387) );
  VHSR_NOR2_1 U355 ( .A1(n323), .A2(n328), .ZN(n341) );
  VHSR_IN_2 U356 ( .I(n341), .ZN(n334) );
  VHSR_NOR2_1 U357 ( .A1(n323), .A2(n333), .ZN(n321) );
  VHSR_OAI21_2 U358 ( .A1(n332), .A2(n328), .B(n321), .ZN(n320) );
  VHSR_OAI31_2 U359 ( .A1(n332), .A2(n321), .A3(n328), .B(n320), .ZN(n344) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[3]), .A2(a[3]), .ZN(n340) );
  VHSR_CLKNAND2_2 U361 ( .A1(b[1]), .A2(a[1]), .ZN(n407) );
  VHSR_AOI22_2 U362 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n330) );
  VHSR_OAI22_2 U363 ( .A1(n340), .A2(n407), .B1(n334), .B2(n330), .ZN(n331) );
  VHSR_OAI22_2 U364 ( .A1(n332), .A2(n324), .B1(n323), .B2(n322), .ZN(n394) );
  VHSR_IN_2 U365 ( .I(n407), .ZN(n327) );
  VHSR_NOR2_1 U366 ( .A1(n348), .A2(n328), .ZN(n326) );
  VHSR_OAI211_2 U367 ( .A1(n326), .A2(n327), .B(b[2]), .C(a[0]), .ZN(n325) );
  VHSR_OAI22_2 U368 ( .A1(n329), .A2(n328), .B1(n348), .B2(n333), .ZN(n393) );
  VHSR_AOI21_2 U369 ( .A1(n330), .A2(n334), .B(n331), .ZN(n346) );
  VHSR_CLKNAND2_2 U370 ( .A1(n347), .A2(n346), .ZN(n345) );
  VHSR_CLKNAND2_2 U371 ( .A1(n344), .A2(n343), .ZN(n338) );
  VHSR_AOI211_2 U372 ( .A1(n334), .A2(n338), .B(n333), .C(n332), .ZN(n386) );
  VHSR_AD1_1 U373 ( .A(n337), .B(n336), .CI(n335), .CO(n317), .S(n390) );
  VHSR_IN_2 U374 ( .I(n338), .ZN(n342) );
  VHSR_CLKNAND2_2 U375 ( .A1(n342), .A2(n340), .ZN(n339) );
  VHSR_OAI31_2 U376 ( .A1(n341), .A2(n342), .A3(n340), .B(n339), .ZN(n389) );
  VHSR_IAO21_2 U377 ( .A1(n344), .A2(n343), .B(n342), .ZN(n392) );
  VHSR_OAI21_2 U378 ( .A1(n347), .A2(n346), .B(n345), .ZN(n405) );
  VHSR_NOR2_1 U379 ( .A1(n349), .A2(n348), .ZN(n350) );
  VHSR_AOI32_2 U380 ( .A1(b[4]), .A2(n351), .A3(a[0]), .B1(n350), .B2(n351), 
        .ZN(n404) );
  VHSR_AD1_1 U381 ( .A(n354), .B(n353), .CI(n352), .CO(n335), .S(n391) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[6]), .A2(b[7]), .ZN(n358) );
  VHSR_AOI21_2 U383 ( .A1(a[7]), .A2(b[6]), .B(n358), .ZN(n357) );
  VHSR_AOI31_2 U384 ( .A1(a[7]), .A2(n358), .A3(b[6]), .B(n357), .ZN(n359) );
  VHSR_IN_2 U385 ( .I(n359), .ZN(n360) );
  VHSR_MAOI222_2 U386 ( .A(n363), .B(n361), .C(n360), .ZN(n370) );
  VHSR_OAI21_2 U387 ( .A1(n363), .A2(n362), .B(n370), .ZN(n367) );
  VHSR_CLKXOR2_2 U388 ( .A1(n368), .A2(n367), .Z(n364) );
  VHSR_CLKNAND2_2 U389 ( .A1(n365), .A2(n364), .ZN(n400) );
  VHSR_OAI21_2 U390 ( .A1(n365), .A2(n364), .B(n400), .ZN(n366) );
  VHSR_NOR2_1 U391 ( .A1(n368), .A2(n367), .ZN(n369) );
  VHSR_NOR2_1 U392 ( .A1(n399), .A2(n372), .ZN(product[15]) );
  VHSR_AD1_1 U393 ( .A(n397), .B(n396), .CI(n395), .CO(n365), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U394 ( .A1(n399), .A2(n398), .ZN(n402) );
  VHSR_XOR3_2 U395 ( .A1(n402), .A2(n401), .A3(n400), .Z(product[14]) );
  VHSR_AOI21_2 U396 ( .A1(n405), .A2(n404), .B(n403), .ZN(product[4]) );
  VHSR_AOI22_2 U397 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n408) );
  VHSR_AOI21_2 U398 ( .A1(n408), .A2(n407), .B(n406), .ZN(product[2]) );
endmodule


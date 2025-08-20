
module mul8_91 ( a, b, product );
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
         n403;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U205 ( .A1(n351), .B1(n283), .ZN(n287) );
  VHSR_NOR2_1 U206 ( .A1(n250), .A2(n246), .ZN(n238) );
  VHSR_NOR2_1 U207 ( .A1(n396), .A2(n395), .ZN(n394) );
  VHSR_INAND2_2 U208 ( .A1(n324), .B1(n343), .ZN(n339) );
  VHSR_INOR3_2 U209 ( .A1(n374), .B1(n285), .B2(n286), .ZN(n309) );
  VHSR_INOR3_2 U210 ( .A1(n238), .B1(n325), .B2(n281), .ZN(n298) );
  VHSR_NOR2_1 U211 ( .A1(n361), .A2(n360), .ZN(n392) );
  VHSR_IN_2 U212 ( .I(n357), .ZN(product[13]) );
  VHSR_NOR2_2 U213 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_INOR2_1 U214 ( .A1(n359), .B1(n358), .ZN(n361) );
  VHSR_NOR2_2 U215 ( .A1(n347), .A2(n346), .ZN(n358) );
  VHSR_NOR2_2 U216 ( .A1(n295), .A2(n294), .ZN(n293) );
  VHSR_MOAI22_1 U217 ( .A1(n240), .A2(n321), .B1(b[6]), .B2(a[3]), .ZN(n245)
         );
  VHSR_AD1_1 U218 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(product[6])
         );
  VHSR_AD1_1 U219 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U220 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(product[10])
         );
  VHSR_AD1_1 U221 ( .A(n385), .B(n401), .CI(n384), .CO(n345), .S(product[3])
         );
  VHSR_AD1_1 U222 ( .A(n383), .B(n382), .CI(n398), .CO(n379), .S(product[5])
         );
  VHSR_AD1_1 U223 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U224 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(product[9])
         );
  VHSR_AD1_1 U225 ( .A(n366), .B(n365), .CI(n364), .CO(n386), .S(product[11])
         );
  VHSR_IN_2 U226 ( .I(b[0]), .ZN(n320) );
  VHSR_IN_2 U227 ( .I(a[1]), .ZN(n316) );
  VHSR_NOR2_1 U228 ( .A1(n320), .A2(n316), .ZN(product[1]) );
  VHSR_AOI22_2 U229 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n250) );
  VHSR_IN_2 U230 ( .I(b[3]), .ZN(n325) );
  VHSR_CLKNAND2_2 U231 ( .A1(b[2]), .A2(a[4]), .ZN(n271) );
  VHSR_IN_2 U232 ( .I(a[5]), .ZN(n285) );
  VHSR_NOR3_2 U233 ( .A1(n325), .A2(n271), .A3(n285), .ZN(n248) );
  VHSR_CLKNAND2_2 U234 ( .A1(a[6]), .A2(b[1]), .ZN(n225) );
  VHSR_IN_2 U235 ( .I(n225), .ZN(n222) );
  VHSR_IN_2 U236 ( .I(n271), .ZN(n219) );
  VHSR_AOI21_2 U237 ( .A1(a[7]), .A2(b[1]), .B(b[2]), .ZN(n215) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[3]), .A2(a[6]), .ZN(n214) );
  VHSR_NOR4_2 U239 ( .A1(n219), .A2(n215), .A3(n214), .A4(n285), .ZN(n216) );
  VHSR_AOI31_2 U240 ( .A1(a[7]), .A2(b[2]), .A3(n222), .B(n216), .ZN(n228) );
  VHSR_IN_2 U241 ( .I(n228), .ZN(n220) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[3]), .A2(a[5]), .ZN(n218) );
  VHSR_AOI32_2 U243 ( .A1(a[7]), .A2(a[6]), .A3(b[1]), .B1(b[2]), .B2(a[6]), 
        .ZN(n217) );
  VHSR_OAI32_2 U244 ( .A1(n220), .A2(n219), .A3(n218), .B1(n217), .B2(n220), 
        .ZN(n257) );
  VHSR_IN_2 U245 ( .I(b[1]), .ZN(n322) );
  VHSR_CLKNAND2_2 U246 ( .A1(a[4]), .A2(b[0]), .ZN(n396) );
  VHSR_NOR3_2 U247 ( .A1(n285), .A2(n322), .A3(n396), .ZN(n276) );
  VHSR_AOI22_2 U248 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n221) );
  VHSR_NOR2_1 U249 ( .A1(n248), .A2(n221), .ZN(n223) );
  VHSR_MAOI222_2 U250 ( .A(n222), .B(n276), .C(n223), .ZN(n227) );
  VHSR_CLKNAND2_2 U251 ( .A1(a[6]), .A2(b[0]), .ZN(n270) );
  VHSR_NAND3_2 U252 ( .A1(b[1]), .A2(a[5]), .A3(n396), .ZN(n269) );
  VHSR_MAOI222_2 U253 ( .A(n271), .B(n270), .C(n269), .ZN(n268) );
  VHSR_NOR2_1 U254 ( .A1(n276), .A2(n223), .ZN(n226) );
  VHSR_IN_2 U255 ( .I(n227), .ZN(n224) );
  VHSR_AOI21_2 U256 ( .A1(n226), .A2(n225), .B(n224), .ZN(n260) );
  VHSR_CLKNAND2_2 U257 ( .A1(n268), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U258 ( .A1(n227), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U259 ( .A1(n257), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U260 ( .A1(n228), .A2(n255), .ZN(n247) );
  VHSR_IN_2 U261 ( .I(a[7]), .ZN(n281) );
  VHSR_IN_2 U262 ( .I(b[7]), .ZN(n240) );
  VHSR_IN_2 U263 ( .I(a[3]), .ZN(n326) );
  VHSR_IN_2 U264 ( .I(a[2]), .ZN(n321) );
  VHSR_AOI22_2 U265 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n235) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[4]), .A2(a[2]), .ZN(n267) );
  VHSR_NAND3_2 U267 ( .A1(a[3]), .A2(b[5]), .A3(n267), .ZN(n234) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[7]), .A2(a[2]), .ZN(n229) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[1]), .ZN(n231) );
  VHSR_OAI22_2 U270 ( .A1(n235), .A2(n234), .B1(n229), .B2(n231), .ZN(n236) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[4]), .A2(a[0]), .ZN(n395) );
  VHSR_NAND3_2 U272 ( .A1(a[1]), .A2(b[5]), .A3(n395), .ZN(n266) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[6]), .A2(a[0]), .ZN(n265) );
  VHSR_MAOI222_2 U274 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NAND4_2 U275 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n242) );
  VHSR_IN_2 U276 ( .I(b[4]), .ZN(n273) );
  VHSR_IN_2 U277 ( .I(b[5]), .ZN(n286) );
  VHSR_OAI22_2 U278 ( .A1(n273), .A2(n326), .B1(n286), .B2(n321), .ZN(n230) );
  VHSR_AND2_2 U279 ( .A1(n242), .A2(n230), .Z(n233) );
  VHSR_IN_2 U280 ( .I(a[0]), .ZN(n317) );
  VHSR_OAI21_2 U281 ( .A1(n240), .A2(n317), .B(n231), .ZN(n232) );
  VHSR_NOR4_2 U282 ( .A1(n273), .A2(n286), .A3(n316), .A4(n317), .ZN(n274) );
  VHSR_AND2_2 U283 ( .A1(n264), .A2(n263), .Z(n262) );
  VHSR_AD1_1 U284 ( .A(n233), .B(n232), .CI(n274), .CO(n251), .S(n263) );
  VHSR_AOI21_2 U285 ( .A1(n235), .A2(n234), .B(n236), .ZN(n254) );
  VHSR_OAI32_2 U286 ( .A1(n236), .A2(n262), .A3(n251), .B1(n254), .B2(n236), 
        .ZN(n243) );
  VHSR_CLKNAND2_2 U287 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U288 ( .A1(n245), .A2(n241), .ZN(n239) );
  VHSR_NOR3_2 U289 ( .A1(n240), .A2(n326), .A3(n239), .ZN(n297) );
  VHSR_NOR2_1 U290 ( .A1(n325), .A2(n281), .ZN(n237) );
  VHSR_IAO21_2 U291 ( .A1(n238), .A2(n237), .B(n298), .ZN(n301) );
  VHSR_OAI32_2 U292 ( .A1(n297), .A2(n326), .A3(n240), .B1(n239), .B2(n297), 
        .ZN(n300) );
  VHSR_OAI21_2 U293 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U294 ( .A1(n245), .A2(n244), .ZN(n308) );
  VHSR_AOI21_2 U295 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U296 ( .A1(n250), .A2(n249), .ZN(n307) );
  VHSR_NOR2_1 U297 ( .A1(n262), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U298 ( .A1(n262), .A2(n251), .B1(n254), .B2(n253), .ZN(n252) );
  VHSR_OAI21_2 U299 ( .A1(n254), .A2(n253), .B(n252), .ZN(n313) );
  VHSR_OAI21_2 U300 ( .A1(n257), .A2(n256), .B(n255), .ZN(n258) );
  VHSR_IN_2 U301 ( .I(n258), .ZN(n312) );
  VHSR_OAI21_2 U302 ( .A1(n268), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U303 ( .I(n261), .ZN(n330) );
  VHSR_IAO21_2 U304 ( .A1(n264), .A2(n263), .B(n262), .ZN(n329) );
  VHSR_AOI31_2 U305 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n337) );
  VHSR_AOI31_2 U306 ( .A1(n271), .A2(n270), .A3(n269), .B(n268), .ZN(n336) );
  VHSR_CLKNAND2_2 U307 ( .A1(b[5]), .A2(a[0]), .ZN(n272) );
  VHSR_OAI32_2 U308 ( .A1(n274), .A2(n316), .A3(n273), .B1(n272), .B2(n274), 
        .ZN(n342) );
  VHSR_AOI22_2 U309 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n275) );
  VHSR_NOR2_1 U310 ( .A1(n276), .A2(n275), .ZN(n341) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[6]), .ZN(n362) );
  VHSR_IN_2 U312 ( .I(n362), .ZN(n389) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[6]), .A2(b[4]), .ZN(n305) );
  VHSR_NAND3_2 U314 ( .A1(a[7]), .A2(b[5]), .A3(n305), .ZN(n278) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[4]), .A2(b[6]), .ZN(n304) );
  VHSR_NAND3_2 U316 ( .A1(b[7]), .A2(a[5]), .A3(n304), .ZN(n277) );
  VHSR_CLKNAND2_2 U317 ( .A1(n278), .A2(n277), .ZN(n280) );
  VHSR_MAOI222_2 U318 ( .A(n362), .B(n278), .C(n277), .ZN(n346) );
  VHSR_IN_2 U319 ( .I(n346), .ZN(n279) );
  VHSR_OAI21_2 U320 ( .A1(n389), .A2(n280), .B(n279), .ZN(n292) );
  VHSR_AND2_2 U321 ( .A1(a[4]), .A2(b[4]), .Z(n374) );
  VHSR_NOR3_2 U322 ( .A1(n281), .A2(n305), .A3(n286), .ZN(n354) );
  VHSR_AOI22_2 U323 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n282) );
  VHSR_NOR2_1 U324 ( .A1(n354), .A2(n282), .ZN(n288) );
  VHSR_NAND4_2 U325 ( .A1(a[4]), .A2(a[5]), .A3(b[6]), .A4(b[7]), .ZN(n351) );
  VHSR_AOI22_2 U326 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n283) );
  VHSR_IN_2 U327 ( .I(n284), .ZN(n295) );
  VHSR_OR3_2 U328 ( .A1(n374), .A2(n286), .A3(n285), .Z(n303) );
  VHSR_MAOI222_2 U329 ( .A(n305), .B(n304), .C(n303), .ZN(n302) );
  VHSR_IN_2 U330 ( .I(n302), .ZN(n294) );
  VHSR_AD1_1 U331 ( .A(n309), .B(n288), .CI(n287), .CO(n289), .S(n284) );
  VHSR_NOR2_1 U332 ( .A1(n293), .A2(n289), .ZN(n291) );
  VHSR_CLKNAND2_2 U333 ( .A1(n293), .A2(n289), .ZN(n290) );
  VHSR_NOR2_1 U334 ( .A1(n291), .A2(n292), .ZN(n347) );
  VHSR_AOI22_2 U335 ( .A1(n292), .A2(n291), .B1(n290), .B2(n347), .ZN(n387) );
  VHSR_AOI21_2 U336 ( .A1(n295), .A2(n294), .B(n293), .ZN(n366) );
  VHSR_AD1_1 U337 ( .A(n298), .B(n297), .CI(n296), .CO(n388), .S(n365) );
  VHSR_AD1_1 U338 ( .A(n301), .B(n300), .CI(n299), .CO(n296), .S(n369) );
  VHSR_AOI31_2 U339 ( .A1(n305), .A2(n304), .A3(n303), .B(n302), .ZN(n368) );
  VHSR_AD1_1 U340 ( .A(n308), .B(n307), .CI(n306), .CO(n299), .S(n372) );
  VHSR_AOI22_2 U341 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n310) );
  VHSR_NOR2_1 U342 ( .A1(n310), .A2(n309), .ZN(n371) );
  VHSR_AD1_1 U343 ( .A(n313), .B(n312), .CI(n311), .CO(n306), .S(n375) );
  VHSR_IN_2 U344 ( .I(b[2]), .ZN(n319) );
  VHSR_NOR2_1 U345 ( .A1(n319), .A2(n321), .ZN(n334) );
  VHSR_IN_2 U346 ( .I(n334), .ZN(n327) );
  VHSR_NOR2_1 U347 ( .A1(n319), .A2(n326), .ZN(n315) );
  VHSR_OAI21_2 U348 ( .A1(n325), .A2(n321), .B(n315), .ZN(n314) );
  VHSR_OAI31_2 U349 ( .A1(n325), .A2(n315), .A3(n321), .B(n314), .ZN(n340) );
  VHSR_AOI22_2 U350 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n323) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[3]), .A2(a[3]), .ZN(n333) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[1]), .A2(a[1]), .ZN(n403) );
  VHSR_OAI22_2 U353 ( .A1(n327), .A2(n323), .B1(n333), .B2(n403), .ZN(n324) );
  VHSR_OAI22_2 U354 ( .A1(n325), .A2(n317), .B1(n319), .B2(n316), .ZN(n385) );
  VHSR_AOI21_2 U355 ( .A1(n322), .A2(n320), .B(n317), .ZN(product[0]) );
  VHSR_AOI32_2 U356 ( .A1(b[0]), .A2(product[0]), .A3(a[2]), .B1(a[1]), .B2(
        product[0]), .ZN(n318) );
  VHSR_AOI211_2 U357 ( .A1(n322), .A2(n321), .B(n319), .C(n318), .ZN(n401) );
  VHSR_OAI22_2 U358 ( .A1(n322), .A2(n321), .B1(n320), .B2(n326), .ZN(n384) );
  VHSR_AOI21_2 U359 ( .A1(n323), .A2(n327), .B(n324), .ZN(n344) );
  VHSR_CLKNAND2_2 U360 ( .A1(n345), .A2(n344), .ZN(n343) );
  VHSR_CLKNAND2_2 U361 ( .A1(n340), .A2(n339), .ZN(n331) );
  VHSR_AOI211_2 U362 ( .A1(n327), .A2(n331), .B(n326), .C(n325), .ZN(n378) );
  VHSR_AD1_1 U363 ( .A(n330), .B(n329), .CI(n328), .CO(n311), .S(n377) );
  VHSR_IN_2 U364 ( .I(n331), .ZN(n338) );
  VHSR_CLKNAND2_2 U365 ( .A1(n338), .A2(n333), .ZN(n332) );
  VHSR_OAI31_2 U366 ( .A1(n334), .A2(n338), .A3(n333), .B(n332), .ZN(n381) );
  VHSR_AD1_1 U367 ( .A(n337), .B(n336), .CI(n335), .CO(n328), .S(n380) );
  VHSR_IAO21_2 U368 ( .A1(n340), .A2(n339), .B(n338), .ZN(n383) );
  VHSR_AD1_1 U369 ( .A(n342), .B(n394), .CI(n341), .CO(n335), .S(n382) );
  VHSR_OAI21_2 U370 ( .A1(n345), .A2(n344), .B(n343), .ZN(n399) );
  VHSR_AOI211_2 U371 ( .A1(n396), .A2(n395), .B(n394), .C(n399), .ZN(n398) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[6]), .ZN(n349) );
  VHSR_AOI21_2 U373 ( .A1(a[6]), .A2(b[7]), .B(n349), .ZN(n348) );
  VHSR_AOI31_2 U374 ( .A1(a[6]), .A2(n349), .A3(b[7]), .B(n348), .ZN(n350) );
  VHSR_CLKNAND2_2 U375 ( .A1(n351), .A2(n350), .ZN(n353) );
  VHSR_IN_2 U376 ( .I(n354), .ZN(n352) );
  VHSR_MAOI222_2 U377 ( .A(n352), .B(n351), .C(n350), .ZN(n360) );
  VHSR_IAO21_2 U378 ( .A1(n354), .A2(n353), .B(n360), .ZN(n359) );
  VHSR_XNOR2_2 U379 ( .A1(n358), .A2(n359), .ZN(n355) );
  VHSR_CLKNAND2_2 U380 ( .A1(n356), .A2(n355), .ZN(n391) );
  VHSR_OAI21_2 U381 ( .A1(n356), .A2(n355), .B(n391), .ZN(n357) );
  VHSR_CLKNAND2_2 U382 ( .A1(a[7]), .A2(b[7]), .ZN(n390) );
  VHSR_AND3_2 U383 ( .A1(n392), .A2(n362), .A3(n391), .Z(n363) );
  VHSR_NOR2_1 U384 ( .A1(n390), .A2(n363), .ZN(product[15]) );
  VHSR_AD1_1 U385 ( .A(n388), .B(n387), .CI(n386), .CO(n356), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U386 ( .A1(n390), .A2(n389), .ZN(n393) );
  VHSR_XOR3_2 U387 ( .A1(n393), .A2(n392), .A3(n391), .Z(product[14]) );
  VHSR_AOI21_2 U388 ( .A1(n396), .A2(n395), .B(n394), .ZN(n397) );
  VHSR_IN_2 U389 ( .I(n397), .ZN(n400) );
  VHSR_AOI21_2 U390 ( .A1(n400), .A2(n399), .B(n398), .ZN(product[4]) );
  VHSR_AOI22_2 U391 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n402) );
  VHSR_AOI21_2 U392 ( .A1(n403), .A2(n402), .B(n401), .ZN(product[2]) );
endmodule


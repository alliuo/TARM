
module mul8_125 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n216, n217, n218, n219, n220,
         n221, n222, n223, n224, n225, n226, n227, n228, n229, n230, n231,
         n232, n233, n234, n235, n236, n237, n238, n239, n240, n241, n242,
         n243, n244, n245, n246, n247, n248, n249, n250, n251, n252, n253,
         n254, n255, n256, n257, n258, n259, n260, n261, n262, n263, n264,
         n265, n266, n267, n268, n269, n270, n271, n272, n273, n274, n275,
         n276, n277, n278, n279, n280, n281, n282, n283, n284, n285, n286,
         n287, n288, n289, n290, n291, n292, n293, n294, n295, n296, n297,
         n298, n299, n300, n301, n302, n303, n304, n305, n306, n307, n308,
         n309, n310, n311, n312, n313, n314, n315, n316, n317, n318, n319,
         n320, n321, n322, n323, n324, n325, n326, n327, n328, n329, n330,
         n331, n332, n333, n334, n335, n336, n337, n338, n339, n340, n341,
         n342, n343, n344, n345, n346, n347, n348, n349, n350, n351, n352,
         n353, n354, n355, n356, n357, n358, n359, n360, n361, n362, n363,
         n364, n365, n366, n367, n368, n369, n370, n371, n372, n373, n374,
         n375, n376, n377, n378, n379, n380, n381, n382, n383, n384, n385,
         n386, n387, n388, n389, n390, n391, n392, n393, n394, n395, n396,
         n397, n398, n399, n400, n401, n402;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U209 ( .A1(n276), .B1(n221), .ZN(n223) );
  VHSR_INOR2_2 U210 ( .A1(n238), .B1(n256), .ZN(n244) );
  VHSR_INOR2_2 U211 ( .A1(n224), .B1(n262), .ZN(n254) );
  VHSR_INOR2_2 U212 ( .A1(n358), .B1(n357), .ZN(n360) );
  VHSR_NOR2_1 U213 ( .A1(n308), .A2(n337), .ZN(n370) );
  VHSR_IN_2 U214 ( .I(n356), .ZN(product[13]) );
  VHSR_INAND2_1 U215 ( .A1(n277), .B1(n353), .ZN(n280) );
  VHSR_AD1_1 U216 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U217 ( .A(n378), .B(n402), .CI(n377), .CO(n336), .S(product[3])
         );
  VHSR_AD1_1 U218 ( .A(n393), .B(n376), .CI(n375), .CO(n379), .S(product[5])
         );
  VHSR_AD1_1 U219 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U220 ( .A(n368), .B(n367), .CI(n366), .CO(n382), .S(product[9])
         );
  VHSR_AD1_1 U221 ( .A(n365), .B(n364), .CI(n363), .CO(n385), .S(product[11])
         );
  VHSR_IN_2 U222 ( .I(a[7]), .ZN(n240) );
  VHSR_IN_2 U223 ( .I(b[3]), .ZN(n321) );
  VHSR_IN_2 U224 ( .I(a[6]), .ZN(n225) );
  VHSR_IN_2 U225 ( .I(b[2]), .ZN(n400) );
  VHSR_OAI22_2 U226 ( .A1(n225), .A2(n321), .B1(n240), .B2(n400), .ZN(n251) );
  VHSR_CLKNAND2_2 U227 ( .A1(a[7]), .A2(b[1]), .ZN(n218) );
  VHSR_CLKNAND2_2 U228 ( .A1(a[6]), .A2(b[2]), .ZN(n217) );
  VHSR_IN_2 U229 ( .I(a[5]), .ZN(n307) );
  VHSR_IN_2 U230 ( .I(a[4]), .ZN(n337) );
  VHSR_NOR2_1 U231 ( .A1(n337), .A2(n400), .ZN(n271) );
  VHSR_NOR3_2 U232 ( .A1(n321), .A2(n307), .A3(n271), .ZN(n228) );
  VHSR_IN_2 U233 ( .I(n228), .ZN(n216) );
  VHSR_MAOI222_2 U234 ( .A(n218), .B(n217), .C(n216), .ZN(n229) );
  VHSR_IN_2 U235 ( .I(b[1]), .ZN(n398) );
  VHSR_IN_2 U236 ( .I(b[0]), .ZN(n397) );
  VHSR_NOR4_2 U237 ( .A1(n337), .A2(n307), .A3(n398), .A4(n397), .ZN(n276) );
  VHSR_NAND4_2 U238 ( .A1(a[4]), .A2(a[5]), .A3(b[3]), .A4(b[2]), .ZN(n248) );
  VHSR_NOR2_1 U239 ( .A1(n307), .A2(n400), .ZN(n219) );
  VHSR_AOI32_2 U240 ( .A1(b[3]), .A2(n248), .A3(a[4]), .B1(n219), .B2(n248), 
        .ZN(n221) );
  VHSR_IN_2 U241 ( .I(n221), .ZN(n220) );
  VHSR_OAI22_2 U242 ( .A1(n225), .A2(n398), .B1(n240), .B2(n397), .ZN(n222) );
  VHSR_MAOI222_2 U243 ( .A(n276), .B(n220), .C(n222), .ZN(n224) );
  VHSR_AOI211_2 U244 ( .A1(a[4]), .A2(b[0]), .B(n307), .C(n398), .ZN(n270) );
  VHSR_NOR2_1 U245 ( .A1(n225), .A2(n397), .ZN(n269) );
  VHSR_MAOI222_2 U246 ( .A(n271), .B(n270), .C(n269), .ZN(n268) );
  VHSR_OAI21_2 U247 ( .A1(n223), .A2(n222), .B(n224), .ZN(n263) );
  VHSR_NOR2_1 U248 ( .A1(n268), .A2(n263), .ZN(n262) );
  VHSR_OAI22_2 U249 ( .A1(n225), .A2(n400), .B1(n240), .B2(n398), .ZN(n227) );
  VHSR_IN_2 U250 ( .I(n229), .ZN(n226) );
  VHSR_OAI21_2 U251 ( .A1(n228), .A2(n227), .B(n226), .ZN(n253) );
  VHSR_NOR2_1 U252 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_NOR2_1 U253 ( .A1(n229), .A2(n252), .ZN(n249) );
  VHSR_CLKNAND2_2 U254 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U255 ( .A1(n251), .A2(n247), .ZN(n239) );
  VHSR_NOR3_2 U256 ( .A1(n240), .A2(n321), .A3(n239), .ZN(n296) );
  VHSR_IN_2 U257 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U258 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U259 ( .I(b[6]), .ZN(n233) );
  VHSR_IN_2 U260 ( .I(a[2]), .ZN(n322) );
  VHSR_OAI22_2 U261 ( .A1(n233), .A2(n324), .B1(n278), .B2(n322), .ZN(n246) );
  VHSR_NOR2_1 U262 ( .A1(n278), .A2(n322), .ZN(n231) );
  VHSR_IN_2 U263 ( .I(a[1]), .ZN(n396) );
  VHSR_NOR2_1 U264 ( .A1(n233), .A2(n396), .ZN(n230) );
  VHSR_IN_2 U265 ( .I(b[5]), .ZN(n309) );
  VHSR_AOI211_2 U266 ( .A1(a[2]), .A2(b[4]), .B(n309), .C(n324), .ZN(n237) );
  VHSR_OAI22_2 U267 ( .A1(n233), .A2(n322), .B1(n278), .B2(n396), .ZN(n236) );
  VHSR_AOI22_2 U268 ( .A1(n231), .A2(n230), .B1(n237), .B2(n236), .ZN(n238) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[4]), .A2(a[2]), .ZN(n267) );
  VHSR_IN_2 U270 ( .I(b[4]), .ZN(n308) );
  VHSR_IN_2 U271 ( .I(a[0]), .ZN(n401) );
  VHSR_OAI211_2 U272 ( .A1(n308), .A2(n401), .B(b[5]), .C(a[1]), .ZN(n266) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[6]), .A2(a[0]), .ZN(n265) );
  VHSR_MAOI222_2 U274 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NOR4_2 U275 ( .A1(n309), .A2(n308), .A3(n396), .A4(n401), .ZN(n274) );
  VHSR_NAND4_2 U276 ( .A1(b[5]), .A2(b[4]), .A3(a[2]), .A4(a[3]), .ZN(n243) );
  VHSR_OAI22_2 U277 ( .A1(n309), .A2(n322), .B1(n308), .B2(n324), .ZN(n232) );
  VHSR_AND2_2 U278 ( .A1(n243), .A2(n232), .Z(n235) );
  VHSR_OAI22_2 U279 ( .A1(n233), .A2(n396), .B1(n278), .B2(n401), .ZN(n234) );
  VHSR_AND2_2 U280 ( .A1(n264), .A2(n261), .Z(n260) );
  VHSR_AD1_1 U281 ( .A(n274), .B(n235), .CI(n234), .CO(n255), .S(n261) );
  VHSR_NOR2_1 U282 ( .A1(n260), .A2(n255), .ZN(n258) );
  VHSR_OAI21_2 U283 ( .A1(n237), .A2(n236), .B(n238), .ZN(n259) );
  VHSR_NOR2_1 U284 ( .A1(n258), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U285 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_CLKNAND2_2 U286 ( .A1(n246), .A2(n242), .ZN(n241) );
  VHSR_NOR3_2 U287 ( .A1(n278), .A2(n324), .A3(n241), .ZN(n295) );
  VHSR_OAI32_2 U288 ( .A1(n296), .A2(n321), .A3(n240), .B1(n239), .B2(n296), 
        .ZN(n299) );
  VHSR_OAI32_2 U289 ( .A1(n295), .A2(n324), .A3(n278), .B1(n241), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U290 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U291 ( .A1(n246), .A2(n245), .ZN(n306) );
  VHSR_OAI21_2 U292 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_XNOR2_2 U293 ( .A1(n251), .A2(n250), .ZN(n305) );
  VHSR_AOI21_2 U294 ( .A1(n254), .A2(n253), .B(n252), .ZN(n314) );
  VHSR_CLKNAND2_2 U295 ( .A1(n260), .A2(n255), .ZN(n257) );
  VHSR_AOI22_2 U296 ( .A1(n259), .A2(n258), .B1(n257), .B2(n256), .ZN(n313) );
  VHSR_IAO21_2 U297 ( .A1(n264), .A2(n261), .B(n260), .ZN(n317) );
  VHSR_AOI21_2 U298 ( .A1(n268), .A2(n263), .B(n262), .ZN(n316) );
  VHSR_AOI31_2 U299 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n330) );
  VHSR_OAI31_2 U300 ( .A1(n271), .A2(n270), .A3(n269), .B(n268), .ZN(n272) );
  VHSR_IN_2 U301 ( .I(n272), .ZN(n329) );
  VHSR_CLKNAND2_2 U302 ( .A1(b[4]), .A2(a[1]), .ZN(n273) );
  VHSR_OAI32_2 U303 ( .A1(n274), .A2(n401), .A3(n309), .B1(n273), .B2(n274), 
        .ZN(n345) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[5]), .A2(b[0]), .ZN(n275) );
  VHSR_OAI32_2 U305 ( .A1(n276), .A2(n398), .A3(n337), .B1(n275), .B2(n276), 
        .ZN(n344) );
  VHSR_NOR2_1 U306 ( .A1(n397), .A2(n401), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U307 ( .A1(n370), .A2(product[0]), .ZN(n339) );
  VHSR_IN_2 U308 ( .I(n339), .ZN(n343) );
  VHSR_AOI22_2 U309 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n277) );
  VHSR_NAND4_2 U310 ( .A1(a[6]), .A2(a[7]), .A3(b[5]), .A4(b[4]), .ZN(n353) );
  VHSR_NAND3_2 U311 ( .A1(b[5]), .A2(a[5]), .A3(n370), .ZN(n311) );
  VHSR_NAND4_2 U312 ( .A1(b[6]), .A2(b[7]), .A3(a[4]), .A4(a[5]), .ZN(n351) );
  VHSR_NOR2_1 U313 ( .A1(n278), .A2(n337), .ZN(n279) );
  VHSR_AOI32_2 U314 ( .A1(b[6]), .A2(n351), .A3(a[5]), .B1(n279), .B2(n351), 
        .ZN(n282) );
  VHSR_MAOI222_2 U315 ( .A(n280), .B(n311), .C(n282), .ZN(n284) );
  VHSR_AND2_2 U316 ( .A1(n280), .A2(n311), .Z(n281) );
  VHSR_AOI21_2 U317 ( .A1(n282), .A2(n281), .B(n284), .ZN(n283) );
  VHSR_IN_2 U318 ( .I(n283), .ZN(n293) );
  VHSR_CLKNAND2_2 U319 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_CLKNAND2_2 U320 ( .A1(b[6]), .A2(a[4]), .ZN(n302) );
  VHSR_OR3_2 U321 ( .A1(n370), .A2(n307), .A3(n309), .Z(n301) );
  VHSR_MAOI222_2 U322 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_IN_2 U323 ( .I(n300), .ZN(n292) );
  VHSR_NOR2_1 U324 ( .A1(n293), .A2(n292), .ZN(n291) );
  VHSR_NOR2_1 U325 ( .A1(n284), .A2(n291), .ZN(n290) );
  VHSR_CLKNAND2_2 U326 ( .A1(a[6]), .A2(b[6]), .ZN(n361) );
  VHSR_IN_2 U327 ( .I(n361), .ZN(n388) );
  VHSR_NAND3_2 U328 ( .A1(b[5]), .A2(a[7]), .A3(n303), .ZN(n286) );
  VHSR_NAND3_2 U329 ( .A1(a[5]), .A2(b[7]), .A3(n302), .ZN(n285) );
  VHSR_CLKNAND2_2 U330 ( .A1(n286), .A2(n285), .ZN(n288) );
  VHSR_MAOI222_2 U331 ( .A(n361), .B(n286), .C(n285), .ZN(n346) );
  VHSR_IN_2 U332 ( .I(n346), .ZN(n287) );
  VHSR_OAI21_2 U333 ( .A1(n388), .A2(n288), .B(n287), .ZN(n289) );
  VHSR_NOR2_1 U334 ( .A1(n290), .A2(n289), .ZN(n347) );
  VHSR_AOI21_2 U335 ( .A1(n290), .A2(n289), .B(n347), .ZN(n386) );
  VHSR_AOI21_2 U336 ( .A1(n293), .A2(n292), .B(n291), .ZN(n365) );
  VHSR_AD1_1 U337 ( .A(n296), .B(n295), .CI(n294), .CO(n387), .S(n364) );
  VHSR_AD1_1 U338 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n384) );
  VHSR_AOI31_2 U339 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n383) );
  VHSR_AD1_1 U340 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n368) );
  VHSR_OAI22_2 U341 ( .A1(n309), .A2(n337), .B1(n308), .B2(n307), .ZN(n310) );
  VHSR_AND2_2 U342 ( .A1(n311), .A2(n310), .Z(n367) );
  VHSR_AD1_1 U343 ( .A(n314), .B(n313), .CI(n312), .CO(n304), .S(n371) );
  VHSR_AD1_1 U344 ( .A(n317), .B(n316), .CI(n315), .CO(n312), .S(n374) );
  VHSR_NOR2_1 U345 ( .A1(n400), .A2(n322), .ZN(n333) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n318) );
  VHSR_AOI31_2 U347 ( .A1(a[3]), .A2(b[3]), .A3(n333), .B(n318), .ZN(n342) );
  VHSR_NOR2_1 U348 ( .A1(n321), .A2(n396), .ZN(n320) );
  VHSR_NOR2_1 U349 ( .A1(n398), .A2(n324), .ZN(n319) );
  VHSR_MAOI222_2 U350 ( .A(n333), .B(n320), .C(n319), .ZN(n326) );
  VHSR_OAI22_2 U351 ( .A1(n321), .A2(n401), .B1(n400), .B2(n396), .ZN(n378) );
  VHSR_AOI22_2 U352 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n399) );
  VHSR_NOR3_2 U353 ( .A1(n399), .A2(n401), .A3(n400), .ZN(n402) );
  VHSR_OAI22_2 U354 ( .A1(n398), .A2(n322), .B1(n397), .B2(n324), .ZN(n377) );
  VHSR_IN_2 U355 ( .I(n326), .ZN(n325) );
  VHSR_AOI21_2 U356 ( .A1(a[1]), .A2(b[3]), .B(n333), .ZN(n323) );
  VHSR_OAI32_2 U357 ( .A1(n325), .A2(n324), .A3(n398), .B1(n323), .B2(n325), 
        .ZN(n335) );
  VHSR_CLKNAND2_2 U358 ( .A1(n336), .A2(n335), .ZN(n334) );
  VHSR_CLKNAND2_2 U359 ( .A1(n326), .A2(n334), .ZN(n341) );
  VHSR_AND2_2 U360 ( .A1(n342), .A2(n341), .Z(n340) );
  VHSR_OAI211_2 U361 ( .A1(n333), .A2(n340), .B(a[3]), .C(b[3]), .ZN(n327) );
  VHSR_IN_2 U362 ( .I(n327), .ZN(n373) );
  VHSR_AD1_1 U363 ( .A(n330), .B(n329), .CI(n328), .CO(n315), .S(n381) );
  VHSR_CLKNAND2_2 U364 ( .A1(b[3]), .A2(a[3]), .ZN(n332) );
  VHSR_CLKNAND2_2 U365 ( .A1(n340), .A2(n332), .ZN(n331) );
  VHSR_OAI31_2 U366 ( .A1(n333), .A2(n340), .A3(n332), .B(n331), .ZN(n380) );
  VHSR_OAI21_2 U367 ( .A1(n336), .A2(n335), .B(n334), .ZN(n395) );
  VHSR_NOR2_1 U368 ( .A1(n337), .A2(n397), .ZN(n338) );
  VHSR_AOI32_2 U369 ( .A1(b[4]), .A2(n339), .A3(a[0]), .B1(n338), .B2(n339), 
        .ZN(n394) );
  VHSR_NOR2_1 U370 ( .A1(n395), .A2(n394), .ZN(n393) );
  VHSR_IAO21_2 U371 ( .A1(n342), .A2(n341), .B(n340), .ZN(n376) );
  VHSR_AD1_1 U372 ( .A(n345), .B(n344), .CI(n343), .CO(n328), .S(n375) );
  VHSR_NOR2_1 U373 ( .A1(n347), .A2(n346), .ZN(n357) );
  VHSR_CLKNAND2_2 U374 ( .A1(b[6]), .A2(a[7]), .ZN(n349) );
  VHSR_AOI21_2 U375 ( .A1(a[6]), .A2(b[7]), .B(n349), .ZN(n348) );
  VHSR_AOI31_2 U376 ( .A1(a[6]), .A2(n349), .A3(b[7]), .B(n348), .ZN(n350) );
  VHSR_AND2_2 U377 ( .A1(n351), .A2(n350), .Z(n352) );
  VHSR_MAOI222_2 U378 ( .A(n353), .B(n351), .C(n350), .ZN(n359) );
  VHSR_AOI21_2 U379 ( .A1(n353), .A2(n352), .B(n359), .ZN(n358) );
  VHSR_XNOR2_2 U380 ( .A1(n357), .A2(n358), .ZN(n354) );
  VHSR_CLKNAND2_2 U381 ( .A1(n355), .A2(n354), .ZN(n390) );
  VHSR_OAI21_2 U382 ( .A1(n355), .A2(n354), .B(n390), .ZN(n356) );
  VHSR_CLKNAND2_2 U383 ( .A1(a[7]), .A2(b[7]), .ZN(n389) );
  VHSR_NOR2_1 U384 ( .A1(n360), .A2(n359), .ZN(n391) );
  VHSR_AND3_2 U385 ( .A1(n391), .A2(n361), .A3(n390), .Z(n362) );
  VHSR_NOR2_1 U386 ( .A1(n389), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U387 ( .A(n381), .B(n380), .CI(n379), .CO(n372), .S(product[6])
         );
  VHSR_AD1_1 U388 ( .A(n384), .B(n383), .CI(n382), .CO(n363), .S(product[10])
         );
  VHSR_AD1_1 U389 ( .A(n387), .B(n386), .CI(n385), .CO(n355), .S(product[12])
         );
  VHSR_NOR2_1 U390 ( .A1(n389), .A2(n388), .ZN(n392) );
  VHSR_XOR3_2 U391 ( .A1(n392), .A2(n391), .A3(n390), .Z(product[14]) );
  VHSR_AOI21_2 U392 ( .A1(n395), .A2(n394), .B(n393), .ZN(product[4]) );
  VHSR_OAI22_2 U393 ( .A1(n398), .A2(n401), .B1(n397), .B2(n396), .ZN(
        product[1]) );
  VHSR_OAI32_2 U394 ( .A1(n402), .A2(n401), .A3(n400), .B1(n399), .B2(n402), 
        .ZN(product[2]) );
endmodule


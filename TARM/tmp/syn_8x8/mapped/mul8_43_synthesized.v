
module mul8_43 ( a, b, product );
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
         n402;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U204 ( .A1(n400), .A2(n284), .ZN(n218) );
  VHSR_NOR2_1 U205 ( .A1(n246), .A2(n245), .ZN(n244) );
  VHSR_NOR2_1 U206 ( .A1(n248), .A2(n244), .ZN(n237) );
  VHSR_INAND2_2 U207 ( .A1(n319), .B1(n338), .ZN(n334) );
  VHSR_INOR3_2 U208 ( .A1(n237), .B1(n320), .B2(n279), .ZN(n297) );
  VHSR_IOA21_2 U209 ( .A1(n392), .A2(n391), .B(n390), .ZN(n395) );
  VHSR_INOR2_2 U210 ( .A1(n357), .B1(n356), .ZN(n388) );
  VHSR_IN_2 U211 ( .I(n353), .ZN(product[13]) );
  VHSR_NOR2_2 U212 ( .A1(n343), .A2(n342), .ZN(n355) );
  VHSR_INAND2_1 U213 ( .A1(n348), .B1(n346), .ZN(n349) );
  VHSR_MOAI22_1 U214 ( .A1(n278), .A2(n317), .B1(b[4]), .B2(a[3]), .ZN(n229)
         );
  VHSR_AD1_1 U215 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(product[6])
         );
  VHSR_AD1_1 U216 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(product[9])
         );
  VHSR_AD1_1 U217 ( .A(n378), .B(n402), .CI(n377), .CO(n340), .S(product[3])
         );
  VHSR_AD1_1 U218 ( .A(n376), .B(n375), .CI(n393), .CO(n372), .S(product[5])
         );
  VHSR_AD1_1 U219 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U220 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U221 ( .A(n362), .B(n361), .CI(n360), .CO(n379), .S(product[10])
         );
  VHSR_AOI22_2 U222 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n248) );
  VHSR_IN_2 U223 ( .I(b[3]), .ZN(n320) );
  VHSR_IN_2 U224 ( .I(b[2]), .ZN(n400) );
  VHSR_IN_2 U225 ( .I(a[4]), .ZN(n284) );
  VHSR_IN_2 U226 ( .I(n218), .ZN(n269) );
  VHSR_IN_2 U227 ( .I(a[5]), .ZN(n283) );
  VHSR_NOR3_2 U228 ( .A1(n320), .A2(n269), .A3(n283), .ZN(n246) );
  VHSR_CLKNAND2_2 U229 ( .A1(a[6]), .A2(b[1]), .ZN(n224) );
  VHSR_IN_2 U230 ( .I(n224), .ZN(n221) );
  VHSR_AOI21_2 U231 ( .A1(a[7]), .A2(b[1]), .B(b[2]), .ZN(n214) );
  VHSR_CLKNAND2_2 U232 ( .A1(b[3]), .A2(a[6]), .ZN(n213) );
  VHSR_NOR4_2 U233 ( .A1(n218), .A2(n214), .A3(n213), .A4(n283), .ZN(n215) );
  VHSR_AOI31_2 U234 ( .A1(a[7]), .A2(b[2]), .A3(n221), .B(n215), .ZN(n227) );
  VHSR_IN_2 U235 ( .I(n227), .ZN(n219) );
  VHSR_CLKNAND2_2 U236 ( .A1(b[3]), .A2(a[5]), .ZN(n217) );
  VHSR_AOI32_2 U237 ( .A1(a[7]), .A2(a[6]), .A3(b[1]), .B1(b[2]), .B2(a[6]), 
        .ZN(n216) );
  VHSR_OAI32_2 U238 ( .A1(n219), .A2(n218), .A3(n217), .B1(n216), .B2(n219), 
        .ZN(n255) );
  VHSR_IN_2 U239 ( .I(b[1]), .ZN(n398) );
  VHSR_IN_2 U240 ( .I(b[0]), .ZN(n397) );
  VHSR_NOR4_2 U241 ( .A1(n284), .A2(n283), .A3(n398), .A4(n397), .ZN(n273) );
  VHSR_AOI22_2 U242 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n220) );
  VHSR_NOR2_1 U243 ( .A1(n246), .A2(n220), .ZN(n222) );
  VHSR_MAOI222_2 U244 ( .A(n221), .B(n273), .C(n222), .ZN(n226) );
  VHSR_CLKNAND2_2 U245 ( .A1(a[6]), .A2(b[0]), .ZN(n268) );
  VHSR_OAI211_2 U246 ( .A1(n284), .A2(n397), .B(a[5]), .C(b[1]), .ZN(n267) );
  VHSR_MAOI222_2 U247 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_NOR2_1 U248 ( .A1(n273), .A2(n222), .ZN(n225) );
  VHSR_IN_2 U249 ( .I(n226), .ZN(n223) );
  VHSR_AOI21_2 U250 ( .A1(n225), .A2(n224), .B(n223), .ZN(n258) );
  VHSR_CLKNAND2_2 U251 ( .A1(n266), .A2(n258), .ZN(n257) );
  VHSR_CLKNAND2_2 U252 ( .A1(n226), .A2(n257), .ZN(n254) );
  VHSR_CLKNAND2_2 U253 ( .A1(n255), .A2(n254), .ZN(n253) );
  VHSR_CLKNAND2_2 U254 ( .A1(n227), .A2(n253), .ZN(n245) );
  VHSR_IN_2 U255 ( .I(a[7]), .ZN(n279) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n281) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n321) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n282) );
  VHSR_IN_2 U259 ( .I(a[2]), .ZN(n317) );
  VHSR_OAI22_2 U260 ( .A1(n282), .A2(n321), .B1(n281), .B2(n317), .ZN(n243) );
  VHSR_AOI22_2 U261 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n234) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[4]), .A2(a[2]), .ZN(n265) );
  VHSR_NAND3_2 U263 ( .A1(a[3]), .A2(b[5]), .A3(n265), .ZN(n233) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[7]), .A2(a[2]), .ZN(n228) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[1]), .ZN(n230) );
  VHSR_OAI22_2 U266 ( .A1(n234), .A2(n233), .B1(n228), .B2(n230), .ZN(n235) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[4]), .A2(a[0]), .ZN(n391) );
  VHSR_NAND3_2 U268 ( .A1(a[1]), .A2(b[5]), .A3(n391), .ZN(n264) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[0]), .ZN(n263) );
  VHSR_MAOI222_2 U270 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_NAND4_2 U271 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n240) );
  VHSR_IN_2 U272 ( .I(b[5]), .ZN(n278) );
  VHSR_AND2_2 U273 ( .A1(n240), .A2(n229), .Z(n232) );
  VHSR_IN_2 U274 ( .I(a[0]), .ZN(n401) );
  VHSR_OAI21_2 U275 ( .A1(n281), .A2(n401), .B(n230), .ZN(n231) );
  VHSR_IN_2 U276 ( .I(a[1]), .ZN(n396) );
  VHSR_NOR3_2 U277 ( .A1(n278), .A2(n396), .A3(n391), .ZN(n271) );
  VHSR_AND2_2 U278 ( .A1(n262), .A2(n261), .Z(n260) );
  VHSR_AD1_1 U279 ( .A(n232), .B(n231), .CI(n271), .CO(n249), .S(n261) );
  VHSR_AOI21_2 U280 ( .A1(n234), .A2(n233), .B(n235), .ZN(n252) );
  VHSR_OAI32_2 U281 ( .A1(n235), .A2(n260), .A3(n249), .B1(n252), .B2(n235), 
        .ZN(n241) );
  VHSR_CLKNAND2_2 U282 ( .A1(n241), .A2(n240), .ZN(n239) );
  VHSR_CLKNAND2_2 U283 ( .A1(n243), .A2(n239), .ZN(n238) );
  VHSR_NOR3_2 U284 ( .A1(n281), .A2(n321), .A3(n238), .ZN(n296) );
  VHSR_NOR2_1 U285 ( .A1(n320), .A2(n279), .ZN(n236) );
  VHSR_IAO21_2 U286 ( .A1(n237), .A2(n236), .B(n297), .ZN(n300) );
  VHSR_OAI32_2 U287 ( .A1(n296), .A2(n321), .A3(n281), .B1(n238), .B2(n296), 
        .ZN(n299) );
  VHSR_OAI21_2 U288 ( .A1(n241), .A2(n240), .B(n239), .ZN(n242) );
  VHSR_XNOR2_2 U289 ( .A1(n243), .A2(n242), .ZN(n307) );
  VHSR_AOI21_2 U290 ( .A1(n246), .A2(n245), .B(n244), .ZN(n247) );
  VHSR_XNOR2_2 U291 ( .A1(n248), .A2(n247), .ZN(n306) );
  VHSR_NOR2_1 U292 ( .A1(n260), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U293 ( .A1(n260), .A2(n249), .B1(n252), .B2(n251), .ZN(n250) );
  VHSR_OAI21_2 U294 ( .A1(n252), .A2(n251), .B(n250), .ZN(n312) );
  VHSR_OAI21_2 U295 ( .A1(n255), .A2(n254), .B(n253), .ZN(n256) );
  VHSR_IN_2 U296 ( .I(n256), .ZN(n311) );
  VHSR_OAI21_2 U297 ( .A1(n266), .A2(n258), .B(n257), .ZN(n259) );
  VHSR_IN_2 U298 ( .I(n259), .ZN(n325) );
  VHSR_IAO21_2 U299 ( .A1(n262), .A2(n261), .B(n260), .ZN(n324) );
  VHSR_AOI31_2 U300 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n332) );
  VHSR_AOI31_2 U301 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n331) );
  VHSR_AOI22_2 U302 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n270) );
  VHSR_NOR2_1 U303 ( .A1(n271), .A2(n270), .ZN(n337) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[4]), .ZN(n286) );
  VHSR_IN_2 U305 ( .I(n286), .ZN(n367) );
  VHSR_NOR2_1 U306 ( .A1(n397), .A2(n401), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U307 ( .A1(n367), .A2(product[0]), .ZN(n390) );
  VHSR_IN_2 U308 ( .I(n390), .ZN(n341) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[5]), .A2(b[0]), .ZN(n272) );
  VHSR_OAI32_2 U310 ( .A1(n273), .A2(n398), .A3(n284), .B1(n272), .B2(n273), 
        .ZN(n336) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[6]), .ZN(n358) );
  VHSR_IN_2 U312 ( .I(n358), .ZN(n385) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[6]), .A2(b[4]), .ZN(n304) );
  VHSR_NAND3_2 U314 ( .A1(a[7]), .A2(b[5]), .A3(n304), .ZN(n275) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[4]), .A2(b[6]), .ZN(n303) );
  VHSR_NAND3_2 U316 ( .A1(b[7]), .A2(a[5]), .A3(n303), .ZN(n274) );
  VHSR_CLKNAND2_2 U317 ( .A1(n275), .A2(n274), .ZN(n277) );
  VHSR_MAOI222_2 U318 ( .A(n358), .B(n275), .C(n274), .ZN(n342) );
  VHSR_IN_2 U319 ( .I(n342), .ZN(n276) );
  VHSR_OAI21_2 U320 ( .A1(n385), .A2(n277), .B(n276), .ZN(n292) );
  VHSR_NOR3_2 U321 ( .A1(n283), .A2(n278), .A3(n286), .ZN(n308) );
  VHSR_NOR3_2 U322 ( .A1(n279), .A2(n304), .A3(n278), .ZN(n350) );
  VHSR_AOI22_2 U323 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n280) );
  VHSR_NOR2_1 U324 ( .A1(n350), .A2(n280), .ZN(n288) );
  VHSR_NOR4_2 U325 ( .A1(n284), .A2(n283), .A3(n282), .A4(n281), .ZN(n348) );
  VHSR_AOI22_2 U326 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n285) );
  VHSR_NOR2_1 U327 ( .A1(n348), .A2(n285), .ZN(n287) );
  VHSR_NAND3_2 U328 ( .A1(b[5]), .A2(a[5]), .A3(n286), .ZN(n302) );
  VHSR_MAOI222_2 U329 ( .A(n304), .B(n303), .C(n302), .ZN(n301) );
  VHSR_AND2_2 U330 ( .A1(n294), .A2(n301), .Z(n293) );
  VHSR_AD1_1 U331 ( .A(n308), .B(n288), .CI(n287), .CO(n289), .S(n294) );
  VHSR_NOR2_1 U332 ( .A1(n293), .A2(n289), .ZN(n291) );
  VHSR_CLKNAND2_2 U333 ( .A1(n293), .A2(n289), .ZN(n290) );
  VHSR_NOR2_1 U334 ( .A1(n291), .A2(n292), .ZN(n343) );
  VHSR_AOI22_2 U335 ( .A1(n292), .A2(n291), .B1(n290), .B2(n343), .ZN(n383) );
  VHSR_IAO21_2 U336 ( .A1(n294), .A2(n301), .B(n293), .ZN(n381) );
  VHSR_AD1_1 U337 ( .A(n297), .B(n296), .CI(n295), .CO(n384), .S(n380) );
  VHSR_AD1_1 U338 ( .A(n300), .B(n299), .CI(n298), .CO(n295), .S(n362) );
  VHSR_AOI31_2 U339 ( .A1(n304), .A2(n303), .A3(n302), .B(n301), .ZN(n361) );
  VHSR_AD1_1 U340 ( .A(n307), .B(n306), .CI(n305), .CO(n298), .S(n365) );
  VHSR_AOI22_2 U341 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n309) );
  VHSR_NOR2_1 U342 ( .A1(n309), .A2(n308), .ZN(n364) );
  VHSR_AD1_1 U343 ( .A(n312), .B(n311), .CI(n310), .CO(n305), .S(n368) );
  VHSR_NOR2_1 U344 ( .A1(n400), .A2(n317), .ZN(n329) );
  VHSR_IN_2 U345 ( .I(n329), .ZN(n322) );
  VHSR_NOR2_1 U346 ( .A1(n400), .A2(n321), .ZN(n314) );
  VHSR_OAI21_2 U347 ( .A1(n320), .A2(n317), .B(n314), .ZN(n313) );
  VHSR_OAI31_2 U348 ( .A1(n320), .A2(n314), .A3(n317), .B(n313), .ZN(n335) );
  VHSR_AOI22_2 U349 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n318) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[3]), .A2(a[3]), .ZN(n328) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[1]), .A2(a[1]), .ZN(n315) );
  VHSR_OAI22_2 U352 ( .A1(n322), .A2(n318), .B1(n328), .B2(n315), .ZN(n319) );
  VHSR_OAI22_2 U353 ( .A1(n320), .A2(n401), .B1(n400), .B2(n396), .ZN(n378) );
  VHSR_IN_2 U354 ( .I(n315), .ZN(n316) );
  VHSR_AOI21_2 U355 ( .A1(a[2]), .A2(b[0]), .B(n316), .ZN(n399) );
  VHSR_NOR3_2 U356 ( .A1(n399), .A2(n401), .A3(n400), .ZN(n402) );
  VHSR_OAI22_2 U357 ( .A1(n398), .A2(n317), .B1(n397), .B2(n321), .ZN(n377) );
  VHSR_AOI21_2 U358 ( .A1(n318), .A2(n322), .B(n319), .ZN(n339) );
  VHSR_CLKNAND2_2 U359 ( .A1(n340), .A2(n339), .ZN(n338) );
  VHSR_CLKNAND2_2 U360 ( .A1(n335), .A2(n334), .ZN(n326) );
  VHSR_AOI211_2 U361 ( .A1(n322), .A2(n326), .B(n321), .C(n320), .ZN(n371) );
  VHSR_AD1_1 U362 ( .A(n325), .B(n324), .CI(n323), .CO(n310), .S(n370) );
  VHSR_IN_2 U363 ( .I(n326), .ZN(n333) );
  VHSR_CLKNAND2_2 U364 ( .A1(n333), .A2(n328), .ZN(n327) );
  VHSR_OAI31_2 U365 ( .A1(n329), .A2(n333), .A3(n328), .B(n327), .ZN(n374) );
  VHSR_AD1_1 U366 ( .A(n332), .B(n331), .CI(n330), .CO(n323), .S(n373) );
  VHSR_IAO21_2 U367 ( .A1(n335), .A2(n334), .B(n333), .ZN(n376) );
  VHSR_AD1_1 U368 ( .A(n337), .B(n341), .CI(n336), .CO(n330), .S(n375) );
  VHSR_CLKNAND2_2 U369 ( .A1(a[4]), .A2(b[0]), .ZN(n392) );
  VHSR_OAI21_2 U370 ( .A1(n340), .A2(n339), .B(n338), .ZN(n394) );
  VHSR_AOI211_2 U371 ( .A1(n392), .A2(n391), .B(n341), .C(n394), .ZN(n393) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[6]), .ZN(n345) );
  VHSR_AOI21_2 U373 ( .A1(a[6]), .A2(b[7]), .B(n345), .ZN(n344) );
  VHSR_AOI31_2 U374 ( .A1(a[6]), .A2(n345), .A3(b[7]), .B(n344), .ZN(n346) );
  VHSR_IN_2 U375 ( .I(n346), .ZN(n347) );
  VHSR_MAOI222_2 U376 ( .A(n350), .B(n348), .C(n347), .ZN(n357) );
  VHSR_OAI21_2 U377 ( .A1(n350), .A2(n349), .B(n357), .ZN(n354) );
  VHSR_CLKXOR2_2 U378 ( .A1(n355), .A2(n354), .Z(n351) );
  VHSR_CLKNAND2_2 U379 ( .A1(n352), .A2(n351), .ZN(n387) );
  VHSR_OAI21_2 U380 ( .A1(n352), .A2(n351), .B(n387), .ZN(n353) );
  VHSR_CLKNAND2_2 U381 ( .A1(a[7]), .A2(b[7]), .ZN(n386) );
  VHSR_NOR2_1 U382 ( .A1(n355), .A2(n354), .ZN(n356) );
  VHSR_AND3_2 U383 ( .A1(n388), .A2(n358), .A3(n387), .Z(n359) );
  VHSR_NOR2_1 U384 ( .A1(n386), .A2(n359), .ZN(product[15]) );
  VHSR_AD1_1 U385 ( .A(n381), .B(n380), .CI(n379), .CO(n382), .S(product[11])
         );
  VHSR_AD1_1 U386 ( .A(n384), .B(n383), .CI(n382), .CO(n352), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U387 ( .A1(n386), .A2(n385), .ZN(n389) );
  VHSR_XOR3_2 U388 ( .A1(n389), .A2(n388), .A3(n387), .Z(product[14]) );
  VHSR_AOI21_2 U389 ( .A1(n395), .A2(n394), .B(n393), .ZN(product[4]) );
  VHSR_OAI22_2 U390 ( .A1(n398), .A2(n401), .B1(n397), .B2(n396), .ZN(
        product[1]) );
  VHSR_OAI32_2 U391 ( .A1(n402), .A2(n401), .A3(n400), .B1(n399), .B2(n402), 
        .ZN(product[2]) );
endmodule


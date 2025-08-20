
module mul8_17 ( a, b, product );
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

  VHSR_INOR2_2 U204 ( .A1(n239), .B1(n223), .ZN(n224) );
  VHSR_NOR2_1 U205 ( .A1(n316), .A2(n307), .ZN(n261) );
  VHSR_INOR2_2 U206 ( .A1(n373), .B1(n278), .ZN(n282) );
  VHSR_INOR2_2 U207 ( .A1(n227), .B1(n252), .ZN(n249) );
  VHSR_INOR2_2 U208 ( .A1(n346), .B1(n345), .ZN(n358) );
  VHSR_INAND2_2 U209 ( .A1(n322), .B1(n336), .ZN(n334) );
  VHSR_NOR2_1 U210 ( .A1(n289), .A2(n293), .ZN(n288) );
  VHSR_NOR2_1 U211 ( .A1(n232), .A2(n231), .ZN(n291) );
  VHSR_IOA21_2 U212 ( .A1(n319), .A2(n318), .B(n317), .ZN(n400) );
  VHSR_NOR2_1 U213 ( .A1(n307), .A2(n339), .ZN(n373) );
  VHSR_IN_2 U214 ( .I(n356), .ZN(product[13]) );
  VHSR_CLKN_1 U215 ( .I(n361), .ZN(n362) );
  VHSR_INAND3_1 U216 ( .A1(n388), .B1(n391), .B2(n390), .ZN(n361) );
  VHSR_INOR2_1 U217 ( .A1(n360), .B1(n359), .ZN(n391) );
  VHSR_INOR2_1 U218 ( .A1(n229), .B1(n247), .ZN(n240) );
  VHSR_INAND2_1 U219 ( .A1(n351), .B1(n349), .ZN(n352) );
  VHSR_INOR3_1 U220 ( .A1(n280), .B1(n272), .B2(n305), .ZN(n353) );
  VHSR_AD1_1 U221 ( .A(n380), .B(n379), .CI(n378), .CO(n375), .S(product[6])
         );
  VHSR_AD1_1 U222 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U223 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[10])
         );
  VHSR_AD1_1 U224 ( .A(n384), .B(n400), .CI(n383), .CO(n338), .S(product[3])
         );
  VHSR_AD1_1 U225 ( .A(n382), .B(n393), .CI(n381), .CO(n378), .S(product[5])
         );
  VHSR_AD1_1 U226 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U227 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(product[9])
         );
  VHSR_AD1_1 U228 ( .A(n365), .B(n364), .CI(n363), .CO(n385), .S(product[11])
         );
  VHSR_IN_2 U229 ( .I(b[7]), .ZN(n274) );
  VHSR_IN_2 U230 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U231 ( .I(b[6]), .ZN(n275) );
  VHSR_IN_2 U232 ( .I(a[2]), .ZN(n320) );
  VHSR_OAI22_2 U233 ( .A1(n275), .A2(n324), .B1(n274), .B2(n320), .ZN(n237) );
  VHSR_AOI22_2 U234 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n219) );
  VHSR_CLKNAND2_2 U235 ( .A1(b[4]), .A2(a[2]), .ZN(n257) );
  VHSR_NAND3_2 U236 ( .A1(a[3]), .A2(b[5]), .A3(n257), .ZN(n218) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[7]), .A2(a[2]), .ZN(n213) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[6]), .A2(a[1]), .ZN(n215) );
  VHSR_OAI22_2 U239 ( .A1(n219), .A2(n218), .B1(n213), .B2(n215), .ZN(n220) );
  VHSR_CLKNAND2_2 U240 ( .A1(b[6]), .A2(a[0]), .ZN(n256) );
  VHSR_IN_2 U241 ( .I(b[4]), .ZN(n339) );
  VHSR_IN_2 U242 ( .I(a[0]), .ZN(n398) );
  VHSR_OAI211_2 U243 ( .A1(n339), .A2(n398), .B(b[5]), .C(a[1]), .ZN(n255) );
  VHSR_MAOI222_2 U244 ( .A(n257), .B(n256), .C(n255), .ZN(n254) );
  VHSR_IN_2 U245 ( .I(b[5]), .ZN(n305) );
  VHSR_IN_2 U246 ( .I(a[1]), .ZN(n396) );
  VHSR_NOR4_2 U247 ( .A1(n339), .A2(n305), .A3(n396), .A4(n398), .ZN(n264) );
  VHSR_NAND4_2 U248 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n234) );
  VHSR_OAI22_2 U249 ( .A1(n339), .A2(n324), .B1(n305), .B2(n320), .ZN(n214) );
  VHSR_AND2_2 U250 ( .A1(n234), .A2(n214), .Z(n217) );
  VHSR_OAI21_2 U251 ( .A1(n274), .A2(n398), .B(n215), .ZN(n216) );
  VHSR_AND2_2 U252 ( .A1(n254), .A2(n251), .Z(n250) );
  VHSR_AD1_1 U253 ( .A(n264), .B(n217), .CI(n216), .CO(n243), .S(n251) );
  VHSR_AOI21_2 U254 ( .A1(n219), .A2(n218), .B(n220), .ZN(n246) );
  VHSR_OAI32_2 U255 ( .A1(n220), .A2(n250), .A3(n243), .B1(n246), .B2(n220), 
        .ZN(n235) );
  VHSR_CLKNAND2_2 U256 ( .A1(n235), .A2(n234), .ZN(n233) );
  VHSR_CLKNAND2_2 U257 ( .A1(n237), .A2(n233), .ZN(n230) );
  VHSR_NOR3_2 U258 ( .A1(n274), .A2(n324), .A3(n230), .ZN(n292) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[3]), .A2(a[7]), .ZN(n232) );
  VHSR_IN_2 U260 ( .I(b[3]), .ZN(n323) );
  VHSR_IN_2 U261 ( .I(a[6]), .ZN(n267) );
  VHSR_IN_2 U262 ( .I(a[7]), .ZN(n272) );
  VHSR_IN_2 U263 ( .I(b[2]), .ZN(n316) );
  VHSR_OAI22_2 U264 ( .A1(n323), .A2(n267), .B1(n272), .B2(n316), .ZN(n242) );
  VHSR_IN_2 U265 ( .I(a[4]), .ZN(n307) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[3]), .A2(a[5]), .ZN(n221) );
  VHSR_IN_2 U267 ( .I(b[1]), .ZN(n399) );
  VHSR_OAI22_2 U268 ( .A1(n261), .A2(n221), .B1(n272), .B2(n399), .ZN(n228) );
  VHSR_IN_2 U269 ( .I(a[5]), .ZN(n303) );
  VHSR_NOR4_2 U270 ( .A1(n261), .A2(n232), .A3(n303), .A4(n399), .ZN(n222) );
  VHSR_AOI31_2 U271 ( .A1(b[2]), .A2(a[6]), .A3(n228), .B(n222), .ZN(n229) );
  VHSR_IN_2 U272 ( .I(b[0]), .ZN(n397) );
  VHSR_NOR4_2 U273 ( .A1(n307), .A2(n303), .A3(n399), .A4(n397), .ZN(n266) );
  VHSR_NAND3_2 U274 ( .A1(b[3]), .A2(n261), .A3(a[5]), .ZN(n239) );
  VHSR_AOI22_2 U275 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n223) );
  VHSR_OAI22_2 U276 ( .A1(n272), .A2(n397), .B1(n267), .B2(n399), .ZN(n225) );
  VHSR_MAOI222_2 U277 ( .A(n266), .B(n224), .C(n225), .ZN(n227) );
  VHSR_AOI211_2 U278 ( .A1(a[4]), .A2(b[0]), .B(n303), .C(n399), .ZN(n260) );
  VHSR_NOR2_1 U279 ( .A1(n267), .A2(n397), .ZN(n259) );
  VHSR_MAOI222_2 U280 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_OR2_2 U281 ( .A1(n266), .A2(n224), .Z(n226) );
  VHSR_OAI21_2 U282 ( .A1(n226), .A2(n225), .B(n227), .ZN(n253) );
  VHSR_NOR2_1 U283 ( .A1(n258), .A2(n253), .ZN(n252) );
  VHSR_AOI32_2 U284 ( .A1(b[2]), .A2(n229), .A3(a[6]), .B1(n228), .B2(n229), 
        .ZN(n248) );
  VHSR_NOR2_1 U285 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U286 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U287 ( .A1(n242), .A2(n238), .ZN(n231) );
  VHSR_OAI32_2 U288 ( .A1(n292), .A2(n324), .A3(n274), .B1(n230), .B2(n292), 
        .ZN(n299) );
  VHSR_AOI21_2 U289 ( .A1(n232), .A2(n231), .B(n291), .ZN(n298) );
  VHSR_OAI21_2 U290 ( .A1(n235), .A2(n234), .B(n233), .ZN(n236) );
  VHSR_XNOR2_2 U291 ( .A1(n237), .A2(n236), .ZN(n302) );
  VHSR_OAI21_2 U292 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U293 ( .A1(n242), .A2(n241), .ZN(n301) );
  VHSR_NOR2_1 U294 ( .A1(n250), .A2(n243), .ZN(n245) );
  VHSR_AOI22_2 U295 ( .A1(n250), .A2(n243), .B1(n246), .B2(n245), .ZN(n244) );
  VHSR_OAI21_2 U296 ( .A1(n246), .A2(n245), .B(n244), .ZN(n310) );
  VHSR_AOI21_2 U297 ( .A1(n249), .A2(n248), .B(n247), .ZN(n309) );
  VHSR_IAO21_2 U298 ( .A1(n254), .A2(n251), .B(n250), .ZN(n313) );
  VHSR_AOI21_2 U299 ( .A1(n258), .A2(n253), .B(n252), .ZN(n312) );
  VHSR_AOI31_2 U300 ( .A1(n257), .A2(n256), .A3(n255), .B(n254), .ZN(n328) );
  VHSR_OAI31_2 U301 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n262) );
  VHSR_IN_2 U302 ( .I(n262), .ZN(n327) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[5]), .A2(a[0]), .ZN(n263) );
  VHSR_OAI32_2 U304 ( .A1(n264), .A2(n396), .A3(n339), .B1(n263), .B2(n264), 
        .ZN(n344) );
  VHSR_NOR2_1 U305 ( .A1(n397), .A2(n398), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U306 ( .A1(n373), .A2(product[0]), .ZN(n341) );
  VHSR_IN_2 U307 ( .I(n341), .ZN(n343) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[5]), .A2(b[0]), .ZN(n265) );
  VHSR_OAI32_2 U309 ( .A1(n266), .A2(n399), .A3(n307), .B1(n265), .B2(n266), 
        .ZN(n342) );
  VHSR_NOR2_1 U310 ( .A1(n267), .A2(n275), .ZN(n388) );
  VHSR_NOR2_1 U311 ( .A1(n307), .A2(n275), .ZN(n279) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[5]), .A2(b[7]), .ZN(n269) );
  VHSR_NOR2_1 U313 ( .A1(n267), .A2(n339), .ZN(n280) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[7]), .A2(b[5]), .ZN(n268) );
  VHSR_OAI22_2 U315 ( .A1(n279), .A2(n269), .B1(n280), .B2(n268), .ZN(n271) );
  VHSR_OR2_2 U316 ( .A1(n279), .A2(n280), .Z(n294) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[5]), .A2(b[5]), .ZN(n278) );
  VHSR_CLKNAND2_2 U318 ( .A1(a[7]), .A2(b[7]), .ZN(n389) );
  VHSR_NOR3_2 U319 ( .A1(n294), .A2(n278), .A3(n389), .ZN(n270) );
  VHSR_AOI31_2 U320 ( .A1(b[6]), .A2(a[6]), .A3(n271), .B(n270), .ZN(n346) );
  VHSR_OAI21_2 U321 ( .A1(n388), .A2(n271), .B(n346), .ZN(n287) );
  VHSR_AOI22_2 U322 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n273) );
  VHSR_NOR2_1 U323 ( .A1(n353), .A2(n273), .ZN(n283) );
  VHSR_NOR4_2 U324 ( .A1(n307), .A2(n303), .A3(n275), .A4(n274), .ZN(n351) );
  VHSR_AOI22_2 U325 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n276) );
  VHSR_NOR2_1 U326 ( .A1(n351), .A2(n276), .ZN(n281) );
  VHSR_IN_2 U327 ( .I(n277), .ZN(n289) );
  VHSR_NOR2_1 U328 ( .A1(n373), .A2(n278), .ZN(n295) );
  VHSR_AOI22_2 U329 ( .A1(n280), .A2(n279), .B1(n295), .B2(n294), .ZN(n293) );
  VHSR_AD1_1 U330 ( .A(n283), .B(n282), .CI(n281), .CO(n284), .S(n277) );
  VHSR_NOR2_1 U331 ( .A1(n288), .A2(n284), .ZN(n286) );
  VHSR_CLKNAND2_2 U332 ( .A1(n288), .A2(n284), .ZN(n285) );
  VHSR_NOR2_1 U333 ( .A1(n286), .A2(n287), .ZN(n345) );
  VHSR_AOI22_2 U334 ( .A1(n287), .A2(n286), .B1(n285), .B2(n345), .ZN(n386) );
  VHSR_AOI21_2 U335 ( .A1(n293), .A2(n289), .B(n288), .ZN(n365) );
  VHSR_AD1_1 U336 ( .A(n292), .B(n291), .CI(n290), .CO(n387), .S(n364) );
  VHSR_OAI21_2 U337 ( .A1(n295), .A2(n294), .B(n293), .ZN(n296) );
  VHSR_IN_2 U338 ( .I(n296), .ZN(n368) );
  VHSR_AD1_1 U339 ( .A(n299), .B(n298), .CI(n297), .CO(n290), .S(n367) );
  VHSR_AD1_1 U340 ( .A(n302), .B(n301), .CI(n300), .CO(n297), .S(n371) );
  VHSR_NOR2_1 U341 ( .A1(n303), .A2(n339), .ZN(n306) );
  VHSR_OAI21_2 U342 ( .A1(n307), .A2(n305), .B(n306), .ZN(n304) );
  VHSR_OAI31_2 U343 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n370) );
  VHSR_AD1_1 U344 ( .A(n310), .B(n309), .CI(n308), .CO(n300), .S(n374) );
  VHSR_AD1_1 U345 ( .A(n313), .B(n312), .CI(n311), .CO(n308), .S(n377) );
  VHSR_NOR2_1 U346 ( .A1(n316), .A2(n320), .ZN(n332) );
  VHSR_IN_2 U347 ( .I(n332), .ZN(n325) );
  VHSR_NOR2_1 U348 ( .A1(n316), .A2(n324), .ZN(n315) );
  VHSR_OAI21_2 U349 ( .A1(n323), .A2(n320), .B(n315), .ZN(n314) );
  VHSR_OAI31_2 U350 ( .A1(n323), .A2(n315), .A3(n320), .B(n314), .ZN(n335) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[3]), .A2(a[3]), .ZN(n331) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[1]), .A2(a[1]), .ZN(n401) );
  VHSR_AOI22_2 U353 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n321) );
  VHSR_OAI22_2 U354 ( .A1(n331), .A2(n401), .B1(n325), .B2(n321), .ZN(n322) );
  VHSR_OAI22_2 U355 ( .A1(n323), .A2(n398), .B1(n316), .B2(n396), .ZN(n384) );
  VHSR_IN_2 U356 ( .I(n401), .ZN(n319) );
  VHSR_NOR2_1 U357 ( .A1(n397), .A2(n320), .ZN(n318) );
  VHSR_OAI211_2 U358 ( .A1(n318), .A2(n319), .B(b[2]), .C(a[0]), .ZN(n317) );
  VHSR_OAI22_2 U359 ( .A1(n399), .A2(n320), .B1(n397), .B2(n324), .ZN(n383) );
  VHSR_AOI21_2 U360 ( .A1(n321), .A2(n325), .B(n322), .ZN(n337) );
  VHSR_CLKNAND2_2 U361 ( .A1(n338), .A2(n337), .ZN(n336) );
  VHSR_CLKNAND2_2 U362 ( .A1(n335), .A2(n334), .ZN(n329) );
  VHSR_AOI211_2 U363 ( .A1(n325), .A2(n329), .B(n324), .C(n323), .ZN(n376) );
  VHSR_AD1_1 U364 ( .A(n328), .B(n327), .CI(n326), .CO(n311), .S(n380) );
  VHSR_IN_2 U365 ( .I(n329), .ZN(n333) );
  VHSR_CLKNAND2_2 U366 ( .A1(n333), .A2(n331), .ZN(n330) );
  VHSR_OAI31_2 U367 ( .A1(n332), .A2(n333), .A3(n331), .B(n330), .ZN(n379) );
  VHSR_IAO21_2 U368 ( .A1(n335), .A2(n334), .B(n333), .ZN(n382) );
  VHSR_OAI21_2 U369 ( .A1(n338), .A2(n337), .B(n336), .ZN(n395) );
  VHSR_NOR2_1 U370 ( .A1(n339), .A2(n398), .ZN(n340) );
  VHSR_AOI32_2 U371 ( .A1(b[0]), .A2(n341), .A3(a[4]), .B1(n340), .B2(n341), 
        .ZN(n394) );
  VHSR_NOR2_1 U372 ( .A1(n395), .A2(n394), .ZN(n393) );
  VHSR_AD1_1 U373 ( .A(n344), .B(n343), .CI(n342), .CO(n326), .S(n381) );
  VHSR_CLKNAND2_2 U374 ( .A1(a[6]), .A2(b[7]), .ZN(n348) );
  VHSR_AOI21_2 U375 ( .A1(a[7]), .A2(b[6]), .B(n348), .ZN(n347) );
  VHSR_AOI31_2 U376 ( .A1(a[7]), .A2(n348), .A3(b[6]), .B(n347), .ZN(n349) );
  VHSR_IN_2 U377 ( .I(n349), .ZN(n350) );
  VHSR_MAOI222_2 U378 ( .A(n353), .B(n351), .C(n350), .ZN(n360) );
  VHSR_OAI21_2 U379 ( .A1(n353), .A2(n352), .B(n360), .ZN(n357) );
  VHSR_CLKXOR2_2 U380 ( .A1(n358), .A2(n357), .Z(n354) );
  VHSR_CLKNAND2_2 U381 ( .A1(n355), .A2(n354), .ZN(n390) );
  VHSR_OAI21_2 U382 ( .A1(n355), .A2(n354), .B(n390), .ZN(n356) );
  VHSR_NOR2_1 U383 ( .A1(n358), .A2(n357), .ZN(n359) );
  VHSR_NOR2_1 U384 ( .A1(n389), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U385 ( .A(n387), .B(n386), .CI(n385), .CO(n355), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U386 ( .A1(n389), .A2(n388), .ZN(n392) );
  VHSR_XOR3_2 U387 ( .A1(n392), .A2(n391), .A3(n390), .Z(product[14]) );
  VHSR_AOI21_2 U388 ( .A1(n395), .A2(n394), .B(n393), .ZN(product[4]) );
  VHSR_OAI22_2 U389 ( .A1(n399), .A2(n398), .B1(n397), .B2(n396), .ZN(
        product[1]) );
  VHSR_AOI22_2 U390 ( .A1(b[2]), .A2(a[0]), .B1(b[0]), .B2(a[2]), .ZN(n402) );
  VHSR_AOI21_2 U391 ( .A1(n402), .A2(n401), .B(n400), .ZN(product[2]) );
endmodule


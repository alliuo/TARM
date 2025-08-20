
module mul8_133 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , n209, n210, n211, n212, n213,
         n214, n215, n216, n217, n218, n219, n220, n221, n222, n223, n224,
         n225, n226, n227, n228, n229, n230, n231, n232, n233, n234, n235,
         n236, n237, n238, n239, n240, n241, n242, n243, n244, n245, n246,
         n247, n248, n249, n250, n251, n252, n253, n254, n255, n256, n257,
         n258, n259, n260, n261, n262, n263, n264, n265, n266, n267, n268,
         n269, n270, n271, n272, n273, n274, n275, n276, n277, n278, n279,
         n280, n281, n282, n283, n284, n285, n286, n287, n288, n289, n290,
         n291, n292, n293, n294, n295, n296, n297, n298, n299, n300, n301,
         n302, n303, n304, n305, n306, n307, n308, n309, n310, n311, n312,
         n313, n314, n315, n316, n317, n318, n319, n320, n321, n322, n323,
         n324, n325, n326, n327, n328, n329, n330, n331, n332, n333, n334,
         n335, n336, n337, n338, n339, n340, n341, n342, n343, n344, n345,
         n346, n347, n348, n349, n350, n351, n352, n353, n354, n355, n356,
         n357, n358, n359, n360, n361, n362, n363, n364, n365, n366, n367,
         n368, n369, n370, n371, n372, n373, n374, n375, n376, n377, n378,
         n379, n380, n381, n382, n383, n384, n385, n386, n387, n388, n389,
         n390;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;

  VHSR_IN_2 U201 ( .I(n237), .ZN(n211) );
  VHSR_INAND3_2 U202 ( .A1(n259), .B1(a[5]), .B2(b[3]), .ZN(n209) );
  VHSR_NOR2_1 U203 ( .A1(n388), .A2(n275), .ZN(n259) );
  VHSR_INOR2_2 U204 ( .A1(n219), .B1(n245), .ZN(n238) );
  VHSR_NOR2_1 U205 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_NOR2_1 U206 ( .A1(n331), .A2(n330), .ZN(n343) );
  VHSR_NOR2_1 U207 ( .A1(n283), .A2(n284), .ZN(n331) );
  VHSR_IOA21_2 U208 ( .A1(n380), .A2(n379), .B(n378), .ZN(n383) );
  VHSR_INOR2_2 U209 ( .A1(n345), .B1(n344), .ZN(n376) );
  VHSR_IN_2 U210 ( .I(n341), .ZN(product[13]) );
  VHSR_NOR2_2 U211 ( .A1(n229), .A2(n228), .ZN(n289) );
  VHSR_INOR2_1 U212 ( .A1(n217), .B1(n248), .ZN(n247) );
  VHSR_NOR2_2 U213 ( .A1(n285), .A2(n281), .ZN(n283) );
  VHSR_NOR2_2 U214 ( .A1(n265), .A2(n274), .ZN(n373) );
  VHSR_MOAI22_1 U215 ( .A1(n270), .A2(n311), .B1(b[4]), .B2(a[3]), .ZN(n221)
         );
  VHSR_AD1_1 U216 ( .A(n364), .B(n363), .CI(n381), .CO(n360), .S(product[5])
         );
  VHSR_AD1_1 U217 ( .A(n353), .B(n352), .CI(n351), .CO(n348), .S(product[9])
         );
  VHSR_AD1_1 U218 ( .A(n366), .B(n390), .CI(n365), .CO(n328), .S(product[3])
         );
  VHSR_AD1_1 U219 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(product[6])
         );
  VHSR_AD1_1 U220 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(product[7])
         );
  VHSR_AD1_1 U221 ( .A(n356), .B(n355), .CI(n354), .CO(n351), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U222 ( .A(n350), .B(n349), .CI(n348), .CO(n367), .S(product[10])
         );
  VHSR_CLKNAND2_2 U223 ( .A1(b[3]), .A2(a[7]), .ZN(n229) );
  VHSR_IN_2 U224 ( .I(b[3]), .ZN(n315) );
  VHSR_IN_2 U225 ( .I(a[6]), .ZN(n265) );
  VHSR_IN_2 U226 ( .I(a[7]), .ZN(n271) );
  VHSR_IN_2 U227 ( .I(b[2]), .ZN(n388) );
  VHSR_OAI22_2 U228 ( .A1(n315), .A2(n265), .B1(n271), .B2(n388), .ZN(n240) );
  VHSR_IN_2 U229 ( .I(b[1]), .ZN(n386) );
  VHSR_IN_2 U230 ( .I(a[4]), .ZN(n275) );
  VHSR_OAI21_2 U231 ( .A1(n386), .A2(n271), .B(n209), .ZN(n218) );
  VHSR_IN_2 U232 ( .I(a[5]), .ZN(n276) );
  VHSR_NOR4_2 U233 ( .A1(n259), .A2(n276), .A3(n229), .A4(n386), .ZN(n210) );
  VHSR_AOI31_2 U234 ( .A1(b[2]), .A2(a[6]), .A3(n218), .B(n210), .ZN(n219) );
  VHSR_NOR2_1 U235 ( .A1(n265), .A2(n386), .ZN(n214) );
  VHSR_IN_2 U236 ( .I(b[0]), .ZN(n385) );
  VHSR_NOR4_2 U237 ( .A1(n276), .A2(n275), .A3(n386), .A4(n385), .ZN(n264) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[2]), .A2(a[5]), .ZN(n213) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[3]), .A2(a[4]), .ZN(n212) );
  VHSR_NAND4_2 U240 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n237) );
  VHSR_AOI21_2 U241 ( .A1(n213), .A2(n212), .B(n211), .ZN(n215) );
  VHSR_MAOI222_2 U242 ( .A(n214), .B(n264), .C(n215), .ZN(n217) );
  VHSR_AOI211_2 U243 ( .A1(a[4]), .A2(b[0]), .B(n276), .C(n386), .ZN(n258) );
  VHSR_AOI21_2 U244 ( .A1(n271), .A2(n265), .B(n385), .ZN(n257) );
  VHSR_MAOI222_2 U245 ( .A(n259), .B(n258), .C(n257), .ZN(n256) );
  VHSR_OR2_2 U246 ( .A1(n264), .A2(n215), .Z(n216) );
  VHSR_AOI32_2 U247 ( .A1(b[1]), .A2(n217), .A3(a[6]), .B1(n216), .B2(n217), 
        .ZN(n249) );
  VHSR_NOR2_1 U248 ( .A1(n256), .A2(n249), .ZN(n248) );
  VHSR_AOI32_2 U249 ( .A1(b[2]), .A2(n219), .A3(a[6]), .B1(n218), .B2(n219), 
        .ZN(n246) );
  VHSR_CLKNAND2_2 U250 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U251 ( .A1(n240), .A2(n236), .ZN(n228) );
  VHSR_IN_2 U252 ( .I(b[7]), .ZN(n273) );
  VHSR_IN_2 U253 ( .I(a[3]), .ZN(n316) );
  VHSR_IN_2 U254 ( .I(b[6]), .ZN(n274) );
  VHSR_IN_2 U255 ( .I(a[2]), .ZN(n311) );
  VHSR_OAI22_2 U256 ( .A1(n274), .A2(n316), .B1(n273), .B2(n311), .ZN(n235) );
  VHSR_AOI22_2 U257 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n226) );
  VHSR_CLKNAND2_2 U258 ( .A1(b[4]), .A2(a[2]), .ZN(n255) );
  VHSR_NAND3_2 U259 ( .A1(a[3]), .A2(b[5]), .A3(n255), .ZN(n225) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[7]), .A2(a[2]), .ZN(n220) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[6]), .A2(a[1]), .ZN(n222) );
  VHSR_OAI22_2 U262 ( .A1(n226), .A2(n225), .B1(n220), .B2(n222), .ZN(n227) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[4]), .A2(a[0]), .ZN(n379) );
  VHSR_NAND3_2 U264 ( .A1(a[1]), .A2(b[5]), .A3(n379), .ZN(n254) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[0]), .ZN(n253) );
  VHSR_MAOI222_2 U266 ( .A(n255), .B(n254), .C(n253), .ZN(n252) );
  VHSR_NAND4_2 U267 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n232) );
  VHSR_IN_2 U268 ( .I(b[5]), .ZN(n270) );
  VHSR_AND2_2 U269 ( .A1(n232), .A2(n221), .Z(n224) );
  VHSR_IN_2 U270 ( .I(a[0]), .ZN(n389) );
  VHSR_OAI21_2 U271 ( .A1(n273), .A2(n389), .B(n222), .ZN(n223) );
  VHSR_IN_2 U272 ( .I(a[1]), .ZN(n384) );
  VHSR_NOR3_2 U273 ( .A1(n270), .A2(n384), .A3(n379), .ZN(n262) );
  VHSR_AND2_2 U274 ( .A1(n252), .A2(n251), .Z(n250) );
  VHSR_AD1_1 U275 ( .A(n224), .B(n223), .CI(n262), .CO(n241), .S(n251) );
  VHSR_AOI21_2 U276 ( .A1(n226), .A2(n225), .B(n227), .ZN(n244) );
  VHSR_OAI32_2 U277 ( .A1(n227), .A2(n250), .A3(n241), .B1(n244), .B2(n227), 
        .ZN(n233) );
  VHSR_CLKNAND2_2 U278 ( .A1(n233), .A2(n232), .ZN(n231) );
  VHSR_CLKNAND2_2 U279 ( .A1(n235), .A2(n231), .ZN(n230) );
  VHSR_NOR3_2 U280 ( .A1(n273), .A2(n316), .A3(n230), .ZN(n288) );
  VHSR_AOI21_2 U281 ( .A1(n229), .A2(n228), .B(n289), .ZN(n292) );
  VHSR_OAI32_2 U282 ( .A1(n288), .A2(n316), .A3(n273), .B1(n230), .B2(n288), 
        .ZN(n291) );
  VHSR_OAI21_2 U283 ( .A1(n233), .A2(n232), .B(n231), .ZN(n234) );
  VHSR_XNOR2_2 U284 ( .A1(n235), .A2(n234), .ZN(n299) );
  VHSR_OAI21_2 U285 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U286 ( .A1(n240), .A2(n239), .ZN(n298) );
  VHSR_NOR2_1 U287 ( .A1(n250), .A2(n241), .ZN(n243) );
  VHSR_AOI22_2 U288 ( .A1(n250), .A2(n241), .B1(n244), .B2(n243), .ZN(n242) );
  VHSR_OAI21_2 U289 ( .A1(n244), .A2(n243), .B(n242), .ZN(n304) );
  VHSR_AOI21_2 U290 ( .A1(n247), .A2(n246), .B(n245), .ZN(n303) );
  VHSR_AOI21_2 U291 ( .A1(n256), .A2(n249), .B(n248), .ZN(n314) );
  VHSR_IAO21_2 U292 ( .A1(n252), .A2(n251), .B(n250), .ZN(n313) );
  VHSR_AOI31_2 U293 ( .A1(n255), .A2(n254), .A3(n253), .B(n252), .ZN(n319) );
  VHSR_OAI31_2 U294 ( .A1(n259), .A2(n258), .A3(n257), .B(n256), .ZN(n260) );
  VHSR_IN_2 U295 ( .I(n260), .ZN(n318) );
  VHSR_AOI22_2 U296 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n261) );
  VHSR_NOR2_1 U297 ( .A1(n262), .A2(n261), .ZN(n325) );
  VHSR_CLKNAND2_2 U298 ( .A1(a[4]), .A2(b[4]), .ZN(n278) );
  VHSR_IN_2 U299 ( .I(n278), .ZN(n355) );
  VHSR_NOR2_1 U300 ( .A1(n385), .A2(n389), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U301 ( .A1(n355), .A2(product[0]), .ZN(n378) );
  VHSR_IN_2 U302 ( .I(n378), .ZN(n329) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[1]), .ZN(n263) );
  VHSR_OAI32_2 U304 ( .A1(n264), .A2(n385), .A3(n276), .B1(n263), .B2(n264), 
        .ZN(n324) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[6]), .ZN(n295) );
  VHSR_NAND3_2 U306 ( .A1(b[7]), .A2(a[5]), .A3(n295), .ZN(n267) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[6]), .A2(b[4]), .ZN(n296) );
  VHSR_NAND3_2 U308 ( .A1(a[7]), .A2(b[5]), .A3(n296), .ZN(n266) );
  VHSR_CLKNAND2_2 U309 ( .A1(n267), .A2(n266), .ZN(n269) );
  VHSR_IN_2 U310 ( .I(n373), .ZN(n346) );
  VHSR_MAOI222_2 U311 ( .A(n346), .B(n267), .C(n266), .ZN(n330) );
  VHSR_IN_2 U312 ( .I(n330), .ZN(n268) );
  VHSR_OAI21_2 U313 ( .A1(n373), .A2(n269), .B(n268), .ZN(n284) );
  VHSR_NOR3_2 U314 ( .A1(n276), .A2(n270), .A3(n278), .ZN(n300) );
  VHSR_NOR3_2 U315 ( .A1(n271), .A2(n296), .A3(n270), .ZN(n338) );
  VHSR_AOI22_2 U316 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n272) );
  VHSR_NOR2_1 U317 ( .A1(n338), .A2(n272), .ZN(n280) );
  VHSR_NOR4_2 U318 ( .A1(n276), .A2(n275), .A3(n274), .A4(n273), .ZN(n336) );
  VHSR_AOI22_2 U319 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n277) );
  VHSR_NOR2_1 U320 ( .A1(n336), .A2(n277), .ZN(n279) );
  VHSR_NAND3_2 U321 ( .A1(b[5]), .A2(a[5]), .A3(n278), .ZN(n294) );
  VHSR_MAOI222_2 U322 ( .A(n296), .B(n295), .C(n294), .ZN(n293) );
  VHSR_AND2_2 U323 ( .A1(n286), .A2(n293), .Z(n285) );
  VHSR_AD1_1 U324 ( .A(n300), .B(n280), .CI(n279), .CO(n281), .S(n286) );
  VHSR_CLKNAND2_2 U325 ( .A1(n285), .A2(n281), .ZN(n282) );
  VHSR_AOI22_2 U326 ( .A1(n284), .A2(n283), .B1(n282), .B2(n331), .ZN(n371) );
  VHSR_IAO21_2 U327 ( .A1(n286), .A2(n293), .B(n285), .ZN(n369) );
  VHSR_AD1_1 U328 ( .A(n289), .B(n288), .CI(n287), .CO(n372), .S(n368) );
  VHSR_AD1_1 U329 ( .A(n292), .B(n291), .CI(n290), .CO(n287), .S(n350) );
  VHSR_AOI31_2 U330 ( .A1(n296), .A2(n295), .A3(n294), .B(n293), .ZN(n349) );
  VHSR_AD1_1 U331 ( .A(n299), .B(n298), .CI(n297), .CO(n290), .S(n353) );
  VHSR_AOI22_2 U332 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n301) );
  VHSR_NOR2_1 U333 ( .A1(n301), .A2(n300), .ZN(n352) );
  VHSR_AD1_1 U334 ( .A(n304), .B(n303), .CI(n302), .CO(n297), .S(n356) );
  VHSR_NOR2_1 U335 ( .A1(n315), .A2(n384), .ZN(n307) );
  VHSR_NOR2_1 U336 ( .A1(n386), .A2(n316), .ZN(n306) );
  VHSR_NOR2_1 U337 ( .A1(n388), .A2(n311), .ZN(n305) );
  VHSR_MAOI222_2 U338 ( .A(n307), .B(n306), .C(n305), .ZN(n310) );
  VHSR_OAI22_2 U339 ( .A1(n315), .A2(n389), .B1(n388), .B2(n384), .ZN(n366) );
  VHSR_AOI22_2 U340 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n387) );
  VHSR_NOR3_2 U341 ( .A1(n387), .A2(n389), .A3(n388), .ZN(n390) );
  VHSR_OAI22_2 U342 ( .A1(n386), .A2(n311), .B1(n385), .B2(n316), .ZN(n365) );
  VHSR_IN_2 U343 ( .I(n310), .ZN(n309) );
  VHSR_AOI22_2 U344 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n308) );
  VHSR_OAI32_2 U345 ( .A1(n309), .A2(n311), .A3(n388), .B1(n308), .B2(n309), 
        .ZN(n327) );
  VHSR_CLKNAND2_2 U346 ( .A1(n328), .A2(n327), .ZN(n326) );
  VHSR_CLKNAND2_2 U347 ( .A1(n310), .A2(n326), .ZN(n321) );
  VHSR_OAI22_2 U348 ( .A1(n315), .A2(n311), .B1(n388), .B2(n316), .ZN(n322) );
  VHSR_CLKNAND2_2 U349 ( .A1(n321), .A2(n322), .ZN(n320) );
  VHSR_NOR3_2 U350 ( .A1(n315), .A2(n316), .A3(n320), .ZN(n359) );
  VHSR_AD1_1 U351 ( .A(n314), .B(n313), .CI(n312), .CO(n302), .S(n358) );
  VHSR_OAI32_2 U352 ( .A1(n359), .A2(n316), .A3(n315), .B1(n320), .B2(n359), 
        .ZN(n362) );
  VHSR_AD1_1 U353 ( .A(n319), .B(n318), .CI(n317), .CO(n312), .S(n361) );
  VHSR_OAI21_2 U354 ( .A1(n322), .A2(n321), .B(n320), .ZN(n323) );
  VHSR_IN_2 U355 ( .I(n323), .ZN(n364) );
  VHSR_AD1_1 U356 ( .A(n325), .B(n329), .CI(n324), .CO(n317), .S(n363) );
  VHSR_CLKNAND2_2 U357 ( .A1(a[4]), .A2(b[0]), .ZN(n380) );
  VHSR_OAI21_2 U358 ( .A1(n328), .A2(n327), .B(n326), .ZN(n382) );
  VHSR_AOI211_2 U359 ( .A1(n380), .A2(n379), .B(n329), .C(n382), .ZN(n381) );
  VHSR_CLKNAND2_2 U360 ( .A1(a[6]), .A2(b[7]), .ZN(n333) );
  VHSR_AOI21_2 U361 ( .A1(a[7]), .A2(b[6]), .B(n333), .ZN(n332) );
  VHSR_AOI31_2 U362 ( .A1(a[7]), .A2(n333), .A3(b[6]), .B(n332), .ZN(n334) );
  VHSR_IN_2 U363 ( .I(n334), .ZN(n335) );
  VHSR_OR2_2 U364 ( .A1(n336), .A2(n335), .Z(n337) );
  VHSR_MAOI222_2 U365 ( .A(n338), .B(n336), .C(n335), .ZN(n345) );
  VHSR_OAI21_2 U366 ( .A1(n338), .A2(n337), .B(n345), .ZN(n342) );
  VHSR_CLKXOR2_2 U367 ( .A1(n343), .A2(n342), .Z(n339) );
  VHSR_CLKNAND2_2 U368 ( .A1(n340), .A2(n339), .ZN(n375) );
  VHSR_OAI21_2 U369 ( .A1(n340), .A2(n339), .B(n375), .ZN(n341) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[7]), .A2(b[7]), .ZN(n374) );
  VHSR_NOR2_1 U371 ( .A1(n343), .A2(n342), .ZN(n344) );
  VHSR_AND3_2 U372 ( .A1(n376), .A2(n346), .A3(n375), .Z(n347) );
  VHSR_NOR2_1 U373 ( .A1(n374), .A2(n347), .ZN(product[15]) );
  VHSR_AD1_1 U374 ( .A(n369), .B(n368), .CI(n367), .CO(n370), .S(product[11])
         );
  VHSR_AD1_1 U375 ( .A(n372), .B(n371), .CI(n370), .CO(n340), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U376 ( .A1(n374), .A2(n373), .ZN(n377) );
  VHSR_XOR3_2 U377 ( .A1(n377), .A2(n376), .A3(n375), .Z(product[14]) );
  VHSR_AOI21_2 U378 ( .A1(n383), .A2(n382), .B(n381), .ZN(product[4]) );
  VHSR_OAI22_2 U379 ( .A1(n386), .A2(n389), .B1(n385), .B2(n384), .ZN(
        product[1]) );
  VHSR_OAI32_2 U380 ( .A1(n390), .A2(n389), .A3(n388), .B1(n387), .B2(n390), 
        .ZN(product[2]) );
endmodule



module mul8_131 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , n207, n208, n209, n210, n211,
         n212, n213, n214, n215, n216, n217, n218, n219, n220, n221, n222,
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
         n377, n378, n379, n380, n381, n382, n383, n384, n385, n386, n387;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;

  VHSR_INAND3_2 U199 ( .A1(n256), .B1(a[5]), .B2(b[3]), .ZN(n207) );
  VHSR_NOR2_1 U200 ( .A1(n385), .A2(n272), .ZN(n256) );
  VHSR_INOR2_2 U201 ( .A1(n216), .B1(n242), .ZN(n235) );
  VHSR_NOR2_1 U202 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_NOR2_1 U203 ( .A1(n328), .A2(n327), .ZN(n340) );
  VHSR_NOR2_1 U204 ( .A1(n280), .A2(n281), .ZN(n328) );
  VHSR_IOA21_2 U205 ( .A1(n377), .A2(n376), .B(n375), .ZN(n380) );
  VHSR_INOR2_2 U206 ( .A1(n342), .B1(n341), .ZN(n373) );
  VHSR_IN_2 U207 ( .I(n338), .ZN(product[13]) );
  VHSR_NOR2_2 U208 ( .A1(n226), .A2(n225), .ZN(n286) );
  VHSR_INOR2_1 U209 ( .A1(n214), .B1(n245), .ZN(n244) );
  VHSR_NOR2_2 U210 ( .A1(n282), .A2(n278), .ZN(n280) );
  VHSR_NOR2_2 U211 ( .A1(n262), .A2(n271), .ZN(n370) );
  VHSR_MOAI22_1 U212 ( .A1(n267), .A2(n308), .B1(b[4]), .B2(a[3]), .ZN(n218)
         );
  VHSR_AD1_1 U213 ( .A(n361), .B(n360), .CI(n378), .CO(n357), .S(product[5])
         );
  VHSR_AD1_1 U214 ( .A(n350), .B(n349), .CI(n348), .CO(n345), .S(product[9])
         );
  VHSR_AD1_1 U215 ( .A(n363), .B(n387), .CI(n362), .CO(n325), .S(product[3])
         );
  VHSR_AD1_1 U216 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(product[6])
         );
  VHSR_AD1_1 U217 ( .A(n356), .B(n355), .CI(n354), .CO(n351), .S(product[7])
         );
  VHSR_AD1_1 U218 ( .A(n353), .B(n352), .CI(n351), .CO(n348), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U219 ( .A(n347), .B(n346), .CI(n345), .CO(n364), .S(product[10])
         );
  VHSR_CLKNAND2_2 U220 ( .A1(b[3]), .A2(a[7]), .ZN(n226) );
  VHSR_IN_2 U221 ( .I(b[3]), .ZN(n312) );
  VHSR_IN_2 U222 ( .I(a[6]), .ZN(n262) );
  VHSR_IN_2 U223 ( .I(a[7]), .ZN(n268) );
  VHSR_IN_2 U224 ( .I(b[2]), .ZN(n385) );
  VHSR_OAI22_2 U225 ( .A1(n312), .A2(n262), .B1(n268), .B2(n385), .ZN(n237) );
  VHSR_IN_2 U226 ( .I(b[1]), .ZN(n383) );
  VHSR_IN_2 U227 ( .I(a[4]), .ZN(n272) );
  VHSR_OAI21_2 U228 ( .A1(n383), .A2(n268), .B(n207), .ZN(n215) );
  VHSR_IN_2 U229 ( .I(a[5]), .ZN(n273) );
  VHSR_NOR4_2 U230 ( .A1(n256), .A2(n273), .A3(n226), .A4(n383), .ZN(n208) );
  VHSR_AOI31_2 U231 ( .A1(b[2]), .A2(a[6]), .A3(n215), .B(n208), .ZN(n216) );
  VHSR_IN_2 U232 ( .I(b[0]), .ZN(n382) );
  VHSR_NOR4_2 U233 ( .A1(n273), .A2(n272), .A3(n383), .A4(n382), .ZN(n261) );
  VHSR_NAND4_2 U234 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n234) );
  VHSR_NOR2_1 U235 ( .A1(n385), .A2(n273), .ZN(n209) );
  VHSR_AOI32_2 U236 ( .A1(b[3]), .A2(n234), .A3(a[4]), .B1(n209), .B2(n234), 
        .ZN(n210) );
  VHSR_IN_2 U237 ( .I(n210), .ZN(n211) );
  VHSR_OAI22_2 U238 ( .A1(n268), .A2(n382), .B1(n262), .B2(n383), .ZN(n212) );
  VHSR_MAOI222_2 U239 ( .A(n261), .B(n211), .C(n212), .ZN(n214) );
  VHSR_NOR2_1 U240 ( .A1(n262), .A2(n382), .ZN(n255) );
  VHSR_AOI211_2 U241 ( .A1(a[4]), .A2(b[0]), .B(n273), .C(n383), .ZN(n254) );
  VHSR_MAOI222_2 U242 ( .A(n256), .B(n255), .C(n254), .ZN(n253) );
  VHSR_OR2_2 U243 ( .A1(n261), .A2(n211), .Z(n213) );
  VHSR_OAI21_2 U244 ( .A1(n213), .A2(n212), .B(n214), .ZN(n246) );
  VHSR_NOR2_1 U245 ( .A1(n253), .A2(n246), .ZN(n245) );
  VHSR_AOI32_2 U246 ( .A1(b[2]), .A2(n216), .A3(a[6]), .B1(n215), .B2(n216), 
        .ZN(n243) );
  VHSR_CLKNAND2_2 U247 ( .A1(n235), .A2(n234), .ZN(n233) );
  VHSR_CLKNAND2_2 U248 ( .A1(n237), .A2(n233), .ZN(n225) );
  VHSR_IN_2 U249 ( .I(b[7]), .ZN(n270) );
  VHSR_IN_2 U250 ( .I(a[3]), .ZN(n313) );
  VHSR_IN_2 U251 ( .I(b[6]), .ZN(n271) );
  VHSR_IN_2 U252 ( .I(a[2]), .ZN(n308) );
  VHSR_OAI22_2 U253 ( .A1(n271), .A2(n313), .B1(n270), .B2(n308), .ZN(n232) );
  VHSR_AOI22_2 U254 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n223) );
  VHSR_CLKNAND2_2 U255 ( .A1(b[4]), .A2(a[2]), .ZN(n252) );
  VHSR_NAND3_2 U256 ( .A1(a[3]), .A2(b[5]), .A3(n252), .ZN(n222) );
  VHSR_CLKNAND2_2 U257 ( .A1(b[7]), .A2(a[2]), .ZN(n217) );
  VHSR_CLKNAND2_2 U258 ( .A1(b[6]), .A2(a[1]), .ZN(n219) );
  VHSR_OAI22_2 U259 ( .A1(n223), .A2(n222), .B1(n217), .B2(n219), .ZN(n224) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[4]), .A2(a[0]), .ZN(n376) );
  VHSR_NAND3_2 U261 ( .A1(a[1]), .A2(b[5]), .A3(n376), .ZN(n251) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[6]), .A2(a[0]), .ZN(n250) );
  VHSR_MAOI222_2 U263 ( .A(n252), .B(n251), .C(n250), .ZN(n249) );
  VHSR_NAND4_2 U264 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n229) );
  VHSR_IN_2 U265 ( .I(b[5]), .ZN(n267) );
  VHSR_AND2_2 U266 ( .A1(n229), .A2(n218), .Z(n221) );
  VHSR_IN_2 U267 ( .I(a[0]), .ZN(n386) );
  VHSR_OAI21_2 U268 ( .A1(n270), .A2(n386), .B(n219), .ZN(n220) );
  VHSR_IN_2 U269 ( .I(a[1]), .ZN(n381) );
  VHSR_NOR3_2 U270 ( .A1(n267), .A2(n381), .A3(n376), .ZN(n259) );
  VHSR_AND2_2 U271 ( .A1(n249), .A2(n248), .Z(n247) );
  VHSR_AD1_1 U272 ( .A(n221), .B(n220), .CI(n259), .CO(n238), .S(n248) );
  VHSR_AOI21_2 U273 ( .A1(n223), .A2(n222), .B(n224), .ZN(n241) );
  VHSR_OAI32_2 U274 ( .A1(n224), .A2(n247), .A3(n238), .B1(n241), .B2(n224), 
        .ZN(n230) );
  VHSR_CLKNAND2_2 U275 ( .A1(n230), .A2(n229), .ZN(n228) );
  VHSR_CLKNAND2_2 U276 ( .A1(n232), .A2(n228), .ZN(n227) );
  VHSR_NOR3_2 U277 ( .A1(n270), .A2(n313), .A3(n227), .ZN(n285) );
  VHSR_AOI21_2 U278 ( .A1(n226), .A2(n225), .B(n286), .ZN(n289) );
  VHSR_OAI32_2 U279 ( .A1(n285), .A2(n313), .A3(n270), .B1(n227), .B2(n285), 
        .ZN(n288) );
  VHSR_OAI21_2 U280 ( .A1(n230), .A2(n229), .B(n228), .ZN(n231) );
  VHSR_XNOR2_2 U281 ( .A1(n232), .A2(n231), .ZN(n296) );
  VHSR_OAI21_2 U282 ( .A1(n235), .A2(n234), .B(n233), .ZN(n236) );
  VHSR_XNOR2_2 U283 ( .A1(n237), .A2(n236), .ZN(n295) );
  VHSR_NOR2_1 U284 ( .A1(n247), .A2(n238), .ZN(n240) );
  VHSR_AOI22_2 U285 ( .A1(n247), .A2(n238), .B1(n241), .B2(n240), .ZN(n239) );
  VHSR_OAI21_2 U286 ( .A1(n241), .A2(n240), .B(n239), .ZN(n301) );
  VHSR_AOI21_2 U287 ( .A1(n244), .A2(n243), .B(n242), .ZN(n300) );
  VHSR_AOI21_2 U288 ( .A1(n253), .A2(n246), .B(n245), .ZN(n311) );
  VHSR_IAO21_2 U289 ( .A1(n249), .A2(n248), .B(n247), .ZN(n310) );
  VHSR_AOI31_2 U290 ( .A1(n252), .A2(n251), .A3(n250), .B(n249), .ZN(n316) );
  VHSR_OAI31_2 U291 ( .A1(n256), .A2(n255), .A3(n254), .B(n253), .ZN(n257) );
  VHSR_IN_2 U292 ( .I(n257), .ZN(n315) );
  VHSR_AOI22_2 U293 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n258) );
  VHSR_NOR2_1 U294 ( .A1(n259), .A2(n258), .ZN(n322) );
  VHSR_CLKNAND2_2 U295 ( .A1(a[4]), .A2(b[4]), .ZN(n275) );
  VHSR_IN_2 U296 ( .I(n275), .ZN(n352) );
  VHSR_NOR2_1 U297 ( .A1(n382), .A2(n386), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U298 ( .A1(n352), .A2(product[0]), .ZN(n375) );
  VHSR_IN_2 U299 ( .I(n375), .ZN(n326) );
  VHSR_CLKNAND2_2 U300 ( .A1(a[4]), .A2(b[1]), .ZN(n260) );
  VHSR_OAI32_2 U301 ( .A1(n261), .A2(n382), .A3(n273), .B1(n260), .B2(n261), 
        .ZN(n321) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[6]), .A2(b[4]), .ZN(n293) );
  VHSR_NAND3_2 U303 ( .A1(a[7]), .A2(b[5]), .A3(n293), .ZN(n264) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[6]), .ZN(n292) );
  VHSR_NAND3_2 U305 ( .A1(b[7]), .A2(a[5]), .A3(n292), .ZN(n263) );
  VHSR_CLKNAND2_2 U306 ( .A1(n264), .A2(n263), .ZN(n266) );
  VHSR_IN_2 U307 ( .I(n370), .ZN(n343) );
  VHSR_MAOI222_2 U308 ( .A(n343), .B(n264), .C(n263), .ZN(n327) );
  VHSR_IN_2 U309 ( .I(n327), .ZN(n265) );
  VHSR_OAI21_2 U310 ( .A1(n370), .A2(n266), .B(n265), .ZN(n281) );
  VHSR_NOR3_2 U311 ( .A1(n273), .A2(n267), .A3(n275), .ZN(n297) );
  VHSR_NOR3_2 U312 ( .A1(n268), .A2(n293), .A3(n267), .ZN(n335) );
  VHSR_AOI22_2 U313 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n269) );
  VHSR_NOR2_1 U314 ( .A1(n335), .A2(n269), .ZN(n277) );
  VHSR_NOR4_2 U315 ( .A1(n273), .A2(n272), .A3(n271), .A4(n270), .ZN(n333) );
  VHSR_AOI22_2 U316 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n274) );
  VHSR_NOR2_1 U317 ( .A1(n333), .A2(n274), .ZN(n276) );
  VHSR_NAND3_2 U318 ( .A1(b[5]), .A2(a[5]), .A3(n275), .ZN(n291) );
  VHSR_MAOI222_2 U319 ( .A(n293), .B(n292), .C(n291), .ZN(n290) );
  VHSR_AND2_2 U320 ( .A1(n283), .A2(n290), .Z(n282) );
  VHSR_AD1_1 U321 ( .A(n297), .B(n277), .CI(n276), .CO(n278), .S(n283) );
  VHSR_CLKNAND2_2 U322 ( .A1(n282), .A2(n278), .ZN(n279) );
  VHSR_AOI22_2 U323 ( .A1(n281), .A2(n280), .B1(n279), .B2(n328), .ZN(n368) );
  VHSR_IAO21_2 U324 ( .A1(n283), .A2(n290), .B(n282), .ZN(n366) );
  VHSR_AD1_1 U325 ( .A(n286), .B(n285), .CI(n284), .CO(n369), .S(n365) );
  VHSR_AD1_1 U326 ( .A(n289), .B(n288), .CI(n287), .CO(n284), .S(n347) );
  VHSR_AOI31_2 U327 ( .A1(n293), .A2(n292), .A3(n291), .B(n290), .ZN(n346) );
  VHSR_AD1_1 U328 ( .A(n296), .B(n295), .CI(n294), .CO(n287), .S(n350) );
  VHSR_AOI22_2 U329 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n298) );
  VHSR_NOR2_1 U330 ( .A1(n298), .A2(n297), .ZN(n349) );
  VHSR_AD1_1 U331 ( .A(n301), .B(n300), .CI(n299), .CO(n294), .S(n353) );
  VHSR_NOR2_1 U332 ( .A1(n312), .A2(n381), .ZN(n304) );
  VHSR_NOR2_1 U333 ( .A1(n383), .A2(n313), .ZN(n303) );
  VHSR_NOR2_1 U334 ( .A1(n385), .A2(n308), .ZN(n302) );
  VHSR_MAOI222_2 U335 ( .A(n304), .B(n303), .C(n302), .ZN(n307) );
  VHSR_OAI22_2 U336 ( .A1(n312), .A2(n386), .B1(n385), .B2(n381), .ZN(n363) );
  VHSR_AOI22_2 U337 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n384) );
  VHSR_NOR3_2 U338 ( .A1(n384), .A2(n386), .A3(n385), .ZN(n387) );
  VHSR_OAI22_2 U339 ( .A1(n383), .A2(n308), .B1(n382), .B2(n313), .ZN(n362) );
  VHSR_IN_2 U340 ( .I(n307), .ZN(n306) );
  VHSR_AOI22_2 U341 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n305) );
  VHSR_OAI32_2 U342 ( .A1(n306), .A2(n308), .A3(n385), .B1(n305), .B2(n306), 
        .ZN(n324) );
  VHSR_CLKNAND2_2 U343 ( .A1(n325), .A2(n324), .ZN(n323) );
  VHSR_CLKNAND2_2 U344 ( .A1(n307), .A2(n323), .ZN(n318) );
  VHSR_OAI22_2 U345 ( .A1(n312), .A2(n308), .B1(n385), .B2(n313), .ZN(n319) );
  VHSR_CLKNAND2_2 U346 ( .A1(n318), .A2(n319), .ZN(n317) );
  VHSR_NOR3_2 U347 ( .A1(n312), .A2(n313), .A3(n317), .ZN(n356) );
  VHSR_AD1_1 U348 ( .A(n311), .B(n310), .CI(n309), .CO(n299), .S(n355) );
  VHSR_OAI32_2 U349 ( .A1(n356), .A2(n313), .A3(n312), .B1(n317), .B2(n356), 
        .ZN(n359) );
  VHSR_AD1_1 U350 ( .A(n316), .B(n315), .CI(n314), .CO(n309), .S(n358) );
  VHSR_OAI21_2 U351 ( .A1(n319), .A2(n318), .B(n317), .ZN(n320) );
  VHSR_IN_2 U352 ( .I(n320), .ZN(n361) );
  VHSR_AD1_1 U353 ( .A(n322), .B(n326), .CI(n321), .CO(n314), .S(n360) );
  VHSR_CLKNAND2_2 U354 ( .A1(a[4]), .A2(b[0]), .ZN(n377) );
  VHSR_OAI21_2 U355 ( .A1(n325), .A2(n324), .B(n323), .ZN(n379) );
  VHSR_AOI211_2 U356 ( .A1(n377), .A2(n376), .B(n326), .C(n379), .ZN(n378) );
  VHSR_CLKNAND2_2 U357 ( .A1(a[6]), .A2(b[7]), .ZN(n330) );
  VHSR_AOI21_2 U358 ( .A1(a[7]), .A2(b[6]), .B(n330), .ZN(n329) );
  VHSR_AOI31_2 U359 ( .A1(a[7]), .A2(n330), .A3(b[6]), .B(n329), .ZN(n331) );
  VHSR_IN_2 U360 ( .I(n331), .ZN(n332) );
  VHSR_OR2_2 U361 ( .A1(n333), .A2(n332), .Z(n334) );
  VHSR_MAOI222_2 U362 ( .A(n335), .B(n333), .C(n332), .ZN(n342) );
  VHSR_OAI21_2 U363 ( .A1(n335), .A2(n334), .B(n342), .ZN(n339) );
  VHSR_CLKXOR2_2 U364 ( .A1(n340), .A2(n339), .Z(n336) );
  VHSR_CLKNAND2_2 U365 ( .A1(n337), .A2(n336), .ZN(n372) );
  VHSR_OAI21_2 U366 ( .A1(n337), .A2(n336), .B(n372), .ZN(n338) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[7]), .A2(b[7]), .ZN(n371) );
  VHSR_NOR2_1 U368 ( .A1(n340), .A2(n339), .ZN(n341) );
  VHSR_AND3_2 U369 ( .A1(n373), .A2(n343), .A3(n372), .Z(n344) );
  VHSR_NOR2_1 U370 ( .A1(n371), .A2(n344), .ZN(product[15]) );
  VHSR_AD1_1 U371 ( .A(n366), .B(n365), .CI(n364), .CO(n367), .S(product[11])
         );
  VHSR_AD1_1 U372 ( .A(n369), .B(n368), .CI(n367), .CO(n337), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U373 ( .A1(n371), .A2(n370), .ZN(n374) );
  VHSR_XOR3_2 U374 ( .A1(n374), .A2(n373), .A3(n372), .Z(product[14]) );
  VHSR_AOI21_2 U375 ( .A1(n380), .A2(n379), .B(n378), .ZN(product[4]) );
  VHSR_OAI22_2 U376 ( .A1(n383), .A2(n386), .B1(n382), .B2(n381), .ZN(
        product[1]) );
  VHSR_OAI32_2 U377 ( .A1(n387), .A2(n386), .A3(n385), .B1(n384), .B2(n387), 
        .ZN(product[2]) );
endmodule


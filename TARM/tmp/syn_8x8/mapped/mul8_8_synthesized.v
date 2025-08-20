
module mul8_8 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n210, n211,
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
         n377, n378, n379, n380, n381, n382, n383, n384, n385, n386, n387,
         n388, n389, n390, n391, n392;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U202 ( .A1(n211), .A2(n390), .ZN(n219) );
  VHSR_INOR2_2 U203 ( .A1(n341), .B1(n273), .ZN(n277) );
  VHSR_NOR2_1 U204 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_NOR2_1 U205 ( .A1(n337), .A2(n336), .ZN(n348) );
  VHSR_NOR2_1 U206 ( .A1(n335), .A2(n384), .ZN(n334) );
  VHSR_INOR3_2 U207 ( .A1(n364), .B1(n275), .B2(n276), .ZN(n299) );
  VHSR_NOR2_1 U208 ( .A1(n281), .A2(n282), .ZN(n337) );
  VHSR_NOR2_1 U209 ( .A1(n351), .A2(n350), .ZN(n382) );
  VHSR_IN_2 U210 ( .I(n347), .ZN(product[13]) );
  VHSR_INOR2_1 U211 ( .A1(n232), .B1(n271), .ZN(n288) );
  VHSR_INOR2_1 U212 ( .A1(n349), .B1(n348), .ZN(n351) );
  VHSR_INOR2_1 U213 ( .A1(n223), .B1(n248), .ZN(n243) );
  VHSR_INAND2_1 U214 ( .A1(n314), .B1(n331), .ZN(n327) );
  VHSR_NOR2_2 U215 ( .A1(n283), .A2(n279), .ZN(n281) );
  VHSR_NOR2_2 U216 ( .A1(n285), .A2(n284), .ZN(n283) );
  VHSR_MOAI22_1 U217 ( .A1(n217), .A2(n216), .B1(a[6]), .B2(n266), .ZN(n218)
         );
  VHSR_MOAI22_1 U218 ( .A1(n276), .A2(n311), .B1(b[4]), .B2(a[3]), .ZN(n225)
         );
  VHSR_MOAI22_1 U219 ( .A1(n235), .A2(n311), .B1(b[6]), .B2(a[3]), .ZN(n240)
         );
  VHSR_AD1_1 U220 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(product[6])
         );
  VHSR_AD1_1 U221 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U222 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(product[10])
         );
  VHSR_AD1_1 U223 ( .A(n375), .B(n392), .CI(n374), .CO(n333), .S(product[3])
         );
  VHSR_AD1_1 U224 ( .A(n373), .B(n372), .CI(n386), .CO(n369), .S(product[5])
         );
  VHSR_AD1_1 U225 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U226 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(product[9])
         );
  VHSR_AD1_1 U227 ( .A(n356), .B(n355), .CI(n354), .CO(n376), .S(product[11])
         );
  VHSR_IN_2 U228 ( .I(b[0]), .ZN(n310) );
  VHSR_IN_2 U229 ( .I(a[1]), .ZN(n305) );
  VHSR_NOR2_1 U230 ( .A1(n310), .A2(n305), .ZN(product[1]) );
  VHSR_IN_2 U231 ( .I(b[1]), .ZN(n312) );
  VHSR_IN_2 U232 ( .I(a[0]), .ZN(n391) );
  VHSR_NOR2_1 U233 ( .A1(n312), .A2(n391), .ZN(product[0]) );
  VHSR_IN_2 U234 ( .I(b[3]), .ZN(n306) );
  VHSR_IN_2 U235 ( .I(a[5]), .ZN(n275) );
  VHSR_AOI211_2 U236 ( .A1(a[4]), .A2(b[2]), .B(n306), .C(n275), .ZN(n222) );
  VHSR_IN_2 U237 ( .I(a[6]), .ZN(n211) );
  VHSR_IN_2 U238 ( .I(b[2]), .ZN(n390) );
  VHSR_IN_2 U239 ( .I(a[7]), .ZN(n271) );
  VHSR_NOR3_2 U240 ( .A1(n271), .A2(n211), .A3(n312), .ZN(n220) );
  VHSR_MAOI222_2 U241 ( .A(n222), .B(n219), .C(n220), .ZN(n223) );
  VHSR_CLKNAND2_2 U242 ( .A1(a[4]), .A2(b[0]), .ZN(n335) );
  VHSR_IN_2 U243 ( .I(n335), .ZN(n385) );
  VHSR_NOR3_2 U244 ( .A1(n385), .A2(n312), .A3(n275), .ZN(n261) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[2]), .A2(a[4]), .ZN(n210) );
  VHSR_OAI21_2 U246 ( .A1(n211), .A2(n310), .B(n210), .ZN(n260) );
  VHSR_AOI22_2 U247 ( .A1(n219), .A2(n385), .B1(n261), .B2(n260), .ZN(n259) );
  VHSR_NOR2_1 U248 ( .A1(n275), .A2(n335), .ZN(n212) );
  VHSR_OAI21_2 U249 ( .A1(a[6]), .A2(n212), .B(b[1]), .ZN(n217) );
  VHSR_CLKNAND2_2 U250 ( .A1(b[2]), .A2(a[5]), .ZN(n214) );
  VHSR_AOI21_2 U251 ( .A1(b[3]), .A2(a[4]), .B(n214), .ZN(n213) );
  VHSR_AOI31_2 U252 ( .A1(b[3]), .A2(n214), .A3(a[4]), .B(n213), .ZN(n216) );
  VHSR_NOR3_2 U253 ( .A1(n275), .A2(n312), .A3(n335), .ZN(n266) );
  VHSR_AOI22_2 U254 ( .A1(n266), .A2(a[6]), .B1(n217), .B2(n216), .ZN(n215) );
  VHSR_OAI21_2 U255 ( .A1(n217), .A2(n216), .B(n215), .ZN(n254) );
  VHSR_NOR2_1 U256 ( .A1(n259), .A2(n254), .ZN(n253) );
  VHSR_NOR2_1 U257 ( .A1(n253), .A2(n218), .ZN(n250) );
  VHSR_OR2_2 U258 ( .A1(n220), .A2(n219), .Z(n221) );
  VHSR_OAI21_2 U259 ( .A1(n222), .A2(n221), .B(n223), .ZN(n249) );
  VHSR_NAND4_2 U260 ( .A1(b[3]), .A2(b[2]), .A3(a[4]), .A4(a[5]), .ZN(n241) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[3]), .A2(a[6]), .ZN(n242) );
  VHSR_AOI21_2 U262 ( .A1(n243), .A2(n241), .B(n242), .ZN(n232) );
  VHSR_IN_2 U263 ( .I(b[7]), .ZN(n235) );
  VHSR_IN_2 U264 ( .I(a[3]), .ZN(n309) );
  VHSR_IN_2 U265 ( .I(a[2]), .ZN(n311) );
  VHSR_AOI22_2 U266 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n230) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[4]), .A2(a[2]), .ZN(n258) );
  VHSR_NAND3_2 U268 ( .A1(a[3]), .A2(b[5]), .A3(n258), .ZN(n229) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[7]), .A2(a[2]), .ZN(n224) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[1]), .ZN(n226) );
  VHSR_OAI22_2 U271 ( .A1(n230), .A2(n229), .B1(n224), .B2(n226), .ZN(n231) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[6]), .A2(a[0]), .ZN(n257) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[4]), .A2(a[0]), .ZN(n384) );
  VHSR_NAND3_2 U274 ( .A1(a[1]), .A2(b[5]), .A3(n384), .ZN(n256) );
  VHSR_MAOI222_2 U275 ( .A(n258), .B(n257), .C(n256), .ZN(n255) );
  VHSR_IN_2 U276 ( .I(b[5]), .ZN(n276) );
  VHSR_NOR3_2 U277 ( .A1(n276), .A2(n305), .A3(n384), .ZN(n263) );
  VHSR_NAND4_2 U278 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_AND2_2 U279 ( .A1(n237), .A2(n225), .Z(n228) );
  VHSR_OAI21_2 U280 ( .A1(n235), .A2(n391), .B(n226), .ZN(n227) );
  VHSR_AND2_2 U281 ( .A1(n255), .A2(n252), .Z(n251) );
  VHSR_AD1_1 U282 ( .A(n263), .B(n228), .CI(n227), .CO(n244), .S(n252) );
  VHSR_AOI21_2 U283 ( .A1(n230), .A2(n229), .B(n231), .ZN(n247) );
  VHSR_OAI32_2 U284 ( .A1(n231), .A2(n251), .A3(n244), .B1(n247), .B2(n231), 
        .ZN(n238) );
  VHSR_CLKNAND2_2 U285 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U286 ( .A1(n240), .A2(n236), .ZN(n234) );
  VHSR_NOR3_2 U287 ( .A1(n235), .A2(n309), .A3(n234), .ZN(n287) );
  VHSR_NOR2_1 U288 ( .A1(n306), .A2(n271), .ZN(n233) );
  VHSR_IAO21_2 U289 ( .A1(n233), .A2(n232), .B(n288), .ZN(n291) );
  VHSR_OAI32_2 U290 ( .A1(n287), .A2(n309), .A3(n235), .B1(n234), .B2(n287), 
        .ZN(n290) );
  VHSR_OAI21_2 U291 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U292 ( .A1(n240), .A2(n239), .ZN(n298) );
  VHSR_XNOR3_2 U293 ( .A1(n243), .A2(n242), .A3(n241), .ZN(n297) );
  VHSR_NOR2_1 U294 ( .A1(n251), .A2(n244), .ZN(n246) );
  VHSR_AOI22_2 U295 ( .A1(n251), .A2(n244), .B1(n247), .B2(n246), .ZN(n245) );
  VHSR_OAI21_2 U296 ( .A1(n247), .A2(n246), .B(n245), .ZN(n303) );
  VHSR_AOI21_2 U297 ( .A1(n250), .A2(n249), .B(n248), .ZN(n302) );
  VHSR_IAO21_2 U298 ( .A1(n255), .A2(n252), .B(n251), .ZN(n318) );
  VHSR_AOI21_2 U299 ( .A1(n259), .A2(n254), .B(n253), .ZN(n317) );
  VHSR_AOI31_2 U300 ( .A1(n258), .A2(n257), .A3(n256), .B(n255), .ZN(n325) );
  VHSR_OAI21_2 U301 ( .A1(n261), .A2(n260), .B(n259), .ZN(n262) );
  VHSR_IN_2 U302 ( .I(n262), .ZN(n324) );
  VHSR_AOI22_2 U303 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n264) );
  VHSR_NOR2_1 U304 ( .A1(n264), .A2(n263), .ZN(n330) );
  VHSR_AOI22_2 U305 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n265) );
  VHSR_NOR2_1 U306 ( .A1(n266), .A2(n265), .ZN(n329) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[6]), .A2(b[6]), .ZN(n352) );
  VHSR_IN_2 U308 ( .I(n352), .ZN(n379) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[4]), .ZN(n295) );
  VHSR_NAND3_2 U310 ( .A1(a[7]), .A2(b[5]), .A3(n295), .ZN(n268) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[4]), .A2(b[6]), .ZN(n294) );
  VHSR_NAND3_2 U312 ( .A1(b[7]), .A2(a[5]), .A3(n294), .ZN(n267) );
  VHSR_CLKNAND2_2 U313 ( .A1(n268), .A2(n267), .ZN(n270) );
  VHSR_MAOI222_2 U314 ( .A(n352), .B(n268), .C(n267), .ZN(n336) );
  VHSR_IN_2 U315 ( .I(n336), .ZN(n269) );
  VHSR_OAI21_2 U316 ( .A1(n379), .A2(n270), .B(n269), .ZN(n282) );
  VHSR_AND2_2 U317 ( .A1(a[4]), .A2(b[4]), .Z(n364) );
  VHSR_NOR3_2 U318 ( .A1(n271), .A2(n295), .A3(n276), .ZN(n344) );
  VHSR_AOI22_2 U319 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n272) );
  VHSR_NOR2_1 U320 ( .A1(n344), .A2(n272), .ZN(n278) );
  VHSR_NAND4_2 U321 ( .A1(a[4]), .A2(a[5]), .A3(b[6]), .A4(b[7]), .ZN(n341) );
  VHSR_AOI22_2 U322 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n273) );
  VHSR_IN_2 U323 ( .I(n274), .ZN(n285) );
  VHSR_OR3_2 U324 ( .A1(n364), .A2(n276), .A3(n275), .Z(n293) );
  VHSR_MAOI222_2 U325 ( .A(n295), .B(n294), .C(n293), .ZN(n292) );
  VHSR_IN_2 U326 ( .I(n292), .ZN(n284) );
  VHSR_AD1_1 U327 ( .A(n299), .B(n278), .CI(n277), .CO(n279), .S(n274) );
  VHSR_CLKNAND2_2 U328 ( .A1(n283), .A2(n279), .ZN(n280) );
  VHSR_AOI22_2 U329 ( .A1(n282), .A2(n281), .B1(n280), .B2(n337), .ZN(n377) );
  VHSR_AOI21_2 U330 ( .A1(n285), .A2(n284), .B(n283), .ZN(n356) );
  VHSR_AD1_1 U331 ( .A(n288), .B(n287), .CI(n286), .CO(n378), .S(n355) );
  VHSR_AD1_1 U332 ( .A(n291), .B(n290), .CI(n289), .CO(n286), .S(n359) );
  VHSR_AOI31_2 U333 ( .A1(n295), .A2(n294), .A3(n293), .B(n292), .ZN(n358) );
  VHSR_AD1_1 U334 ( .A(n298), .B(n297), .CI(n296), .CO(n289), .S(n362) );
  VHSR_AOI22_2 U335 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n300) );
  VHSR_NOR2_1 U336 ( .A1(n300), .A2(n299), .ZN(n361) );
  VHSR_AD1_1 U337 ( .A(n303), .B(n302), .CI(n301), .CO(n296), .S(n365) );
  VHSR_NOR2_1 U338 ( .A1(n390), .A2(n311), .ZN(n322) );
  VHSR_IN_2 U339 ( .I(n322), .ZN(n315) );
  VHSR_CLKNAND2_2 U340 ( .A1(b[3]), .A2(a[3]), .ZN(n321) );
  VHSR_AOI22_2 U341 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n304) );
  VHSR_IAO21_2 U342 ( .A1(n315), .A2(n321), .B(n304), .ZN(n328) );
  VHSR_AOI22_2 U343 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n313) );
  VHSR_CLKNAND2_2 U344 ( .A1(b[1]), .A2(a[1]), .ZN(n307) );
  VHSR_OAI22_2 U345 ( .A1(n315), .A2(n313), .B1(n321), .B2(n307), .ZN(n314) );
  VHSR_OAI22_2 U346 ( .A1(n306), .A2(n391), .B1(n390), .B2(n305), .ZN(n375) );
  VHSR_OAI21_2 U347 ( .A1(n311), .A2(n310), .B(n307), .ZN(n308) );
  VHSR_IN_2 U348 ( .I(n308), .ZN(n389) );
  VHSR_NOR3_2 U349 ( .A1(n389), .A2(n391), .A3(n390), .ZN(n392) );
  VHSR_OAI22_2 U350 ( .A1(n312), .A2(n311), .B1(n310), .B2(n309), .ZN(n374) );
  VHSR_AOI21_2 U351 ( .A1(n313), .A2(n315), .B(n314), .ZN(n332) );
  VHSR_CLKNAND2_2 U352 ( .A1(n333), .A2(n332), .ZN(n331) );
  VHSR_CLKNAND2_2 U353 ( .A1(n328), .A2(n327), .ZN(n319) );
  VHSR_AOI21_2 U354 ( .A1(n315), .A2(n319), .B(n321), .ZN(n368) );
  VHSR_AD1_1 U355 ( .A(n318), .B(n317), .CI(n316), .CO(n301), .S(n367) );
  VHSR_IN_2 U356 ( .I(n319), .ZN(n326) );
  VHSR_CLKNAND2_2 U357 ( .A1(n326), .A2(n321), .ZN(n320) );
  VHSR_OAI31_2 U358 ( .A1(n322), .A2(n326), .A3(n321), .B(n320), .ZN(n371) );
  VHSR_AD1_1 U359 ( .A(n325), .B(n324), .CI(n323), .CO(n316), .S(n370) );
  VHSR_IAO21_2 U360 ( .A1(n328), .A2(n327), .B(n326), .ZN(n373) );
  VHSR_AD1_1 U361 ( .A(n330), .B(n334), .CI(n329), .CO(n323), .S(n372) );
  VHSR_OAI21_2 U362 ( .A1(n333), .A2(n332), .B(n331), .ZN(n387) );
  VHSR_AOI211_2 U363 ( .A1(n335), .A2(n384), .B(n334), .C(n387), .ZN(n386) );
  VHSR_CLKNAND2_2 U364 ( .A1(a[6]), .A2(b[7]), .ZN(n339) );
  VHSR_AOI21_2 U365 ( .A1(a[7]), .A2(b[6]), .B(n339), .ZN(n338) );
  VHSR_AOI31_2 U366 ( .A1(a[7]), .A2(n339), .A3(b[6]), .B(n338), .ZN(n340) );
  VHSR_CLKNAND2_2 U367 ( .A1(n341), .A2(n340), .ZN(n343) );
  VHSR_IN_2 U368 ( .I(n344), .ZN(n342) );
  VHSR_MAOI222_2 U369 ( .A(n342), .B(n341), .C(n340), .ZN(n350) );
  VHSR_IAO21_2 U370 ( .A1(n344), .A2(n343), .B(n350), .ZN(n349) );
  VHSR_XNOR2_2 U371 ( .A1(n348), .A2(n349), .ZN(n345) );
  VHSR_CLKNAND2_2 U372 ( .A1(n346), .A2(n345), .ZN(n381) );
  VHSR_OAI21_2 U373 ( .A1(n346), .A2(n345), .B(n381), .ZN(n347) );
  VHSR_CLKNAND2_2 U374 ( .A1(a[7]), .A2(b[7]), .ZN(n380) );
  VHSR_AND3_2 U375 ( .A1(n382), .A2(n352), .A3(n381), .Z(n353) );
  VHSR_NOR2_1 U376 ( .A1(n380), .A2(n353), .ZN(product[15]) );
  VHSR_AD1_1 U377 ( .A(n378), .B(n377), .CI(n376), .CO(n346), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U378 ( .A1(n380), .A2(n379), .ZN(n383) );
  VHSR_XOR3_2 U379 ( .A1(n383), .A2(n382), .A3(n381), .Z(product[14]) );
  VHSR_IAO22_2 U380 ( .B1(n385), .B2(n384), .A1(n384), .A2(n385), .ZN(n388) );
  VHSR_AOI21_2 U381 ( .A1(n388), .A2(n387), .B(n386), .ZN(product[4]) );
  VHSR_OAI32_2 U382 ( .A1(n392), .A2(n391), .A3(n390), .B1(n389), .B2(n392), 
        .ZN(product[2]) );
endmodule


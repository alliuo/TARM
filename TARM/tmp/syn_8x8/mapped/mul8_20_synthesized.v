
module mul8_20 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236, n237, n238,
         n239, n240, n241, n242, n243, n244, n245, n246, n247, n248, n249,
         n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, n260,
         n261, n262, n263, n264, n265, n266, n267, n268, n269, n270, n271,
         n272, n273, n274, n275, n276, n277, n278, n279, n280, n281, n282,
         n283, n284, n285, n286, n287, n288, n289, n290, n291, n292, n293,
         n294, n295, n296, n297, n298, n299, n300, n301, n302, n303, n304,
         n305, n306, n307, n308, n309, n310, n311, n312, n313, n314, n315,
         n316, n317, n318, n319, n320, n321, n322, n323, n324, n325, n326,
         n327, n328, n329, n330, n331, n332, n333, n334, n335, n336, n337,
         n338, n339, n340, n341, n342, n343, n344, n345, n346, n347, n348,
         n349, n350, n351, n352, n353, n354, n355, n356, n357, n358, n359,
         n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, n370,
         n371, n372, n373, n374, n375, n376, n377, n378, n379, n380, n381,
         n382, n383, n384, n385, n386, n387, n388, n389, n390, n391, n392,
         n393, n394, n395, n396, n397, n398, n399, n400;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U205 ( .A1(n264), .B1(a[5]), .B2(b[3]), .ZN(n215) );
  VHSR_NOR2_1 U206 ( .A1(n312), .A2(n280), .ZN(n264) );
  VHSR_INOR2_2 U207 ( .A1(n224), .B1(n250), .ZN(n243) );
  VHSR_INAND2_2 U208 ( .A1(n318), .B1(n317), .ZN(n319) );
  VHSR_INOR2_2 U209 ( .A1(n222), .B1(n253), .ZN(n252) );
  VHSR_NOR2_1 U210 ( .A1(n341), .A2(n340), .ZN(n353) );
  VHSR_NOR2_1 U211 ( .A1(n312), .A2(n313), .ZN(n328) );
  VHSR_NOR2_1 U212 ( .A1(n234), .A2(n233), .ZN(n294) );
  VHSR_IOA21_2 U213 ( .A1(n390), .A2(n389), .B(n388), .ZN(n392) );
  VHSR_INOR2_2 U214 ( .A1(n355), .B1(n354), .ZN(n386) );
  VHSR_IN_2 U215 ( .I(n351), .ZN(product[13]) );
  VHSR_NOR2_2 U216 ( .A1(n270), .A2(n279), .ZN(n383) );
  VHSR_MOAI22_1 U217 ( .A1(n275), .A2(n313), .B1(b[4]), .B2(a[3]), .ZN(n226)
         );
  VHSR_AD1_1 U218 ( .A(n363), .B(n362), .CI(n361), .CO(n358), .S(product[9])
         );
  VHSR_AD1_1 U219 ( .A(n370), .B(n398), .CI(n369), .CO(n333), .S(product[3])
         );
  VHSR_AD1_1 U220 ( .A(n391), .B(n368), .CI(n367), .CO(n371), .S(product[5])
         );
  VHSR_AD1_1 U221 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U222 ( .A(n360), .B(n359), .CI(n358), .CO(n377), .S(product[10])
         );
  VHSR_CLKNAND2_2 U223 ( .A1(b[3]), .A2(a[7]), .ZN(n234) );
  VHSR_IN_2 U224 ( .I(b[3]), .ZN(n315) );
  VHSR_IN_2 U225 ( .I(a[6]), .ZN(n270) );
  VHSR_IN_2 U226 ( .I(a[7]), .ZN(n276) );
  VHSR_IN_2 U227 ( .I(b[2]), .ZN(n312) );
  VHSR_OAI22_2 U228 ( .A1(n315), .A2(n270), .B1(n276), .B2(n312), .ZN(n245) );
  VHSR_IN_2 U229 ( .I(b[1]), .ZN(n397) );
  VHSR_IN_2 U230 ( .I(a[4]), .ZN(n280) );
  VHSR_OAI21_2 U231 ( .A1(n397), .A2(n276), .B(n215), .ZN(n223) );
  VHSR_IN_2 U232 ( .I(a[5]), .ZN(n281) );
  VHSR_NOR4_2 U233 ( .A1(n264), .A2(n281), .A3(n234), .A4(n397), .ZN(n216) );
  VHSR_AOI31_2 U234 ( .A1(b[2]), .A2(a[6]), .A3(n223), .B(n216), .ZN(n224) );
  VHSR_IN_2 U235 ( .I(b[0]), .ZN(n395) );
  VHSR_NOR4_2 U236 ( .A1(n281), .A2(n280), .A3(n397), .A4(n395), .ZN(n269) );
  VHSR_NAND4_2 U237 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n242) );
  VHSR_NOR2_1 U238 ( .A1(n312), .A2(n281), .ZN(n217) );
  VHSR_AOI32_2 U239 ( .A1(b[3]), .A2(n242), .A3(a[4]), .B1(n217), .B2(n242), 
        .ZN(n218) );
  VHSR_IN_2 U240 ( .I(n218), .ZN(n219) );
  VHSR_OAI22_2 U241 ( .A1(n276), .A2(n395), .B1(n270), .B2(n397), .ZN(n220) );
  VHSR_MAOI222_2 U242 ( .A(n269), .B(n219), .C(n220), .ZN(n222) );
  VHSR_NOR2_1 U243 ( .A1(n270), .A2(n395), .ZN(n263) );
  VHSR_AOI211_2 U244 ( .A1(a[4]), .A2(b[0]), .B(n281), .C(n397), .ZN(n262) );
  VHSR_MAOI222_2 U245 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_OR2_2 U246 ( .A1(n269), .A2(n219), .Z(n221) );
  VHSR_OAI21_2 U247 ( .A1(n221), .A2(n220), .B(n222), .ZN(n254) );
  VHSR_NOR2_1 U248 ( .A1(n261), .A2(n254), .ZN(n253) );
  VHSR_AOI32_2 U249 ( .A1(b[2]), .A2(n224), .A3(a[6]), .B1(n223), .B2(n224), 
        .ZN(n251) );
  VHSR_NOR2_1 U250 ( .A1(n252), .A2(n251), .ZN(n250) );
  VHSR_CLKNAND2_2 U251 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U252 ( .A1(n245), .A2(n241), .ZN(n233) );
  VHSR_IN_2 U253 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U254 ( .I(a[3]), .ZN(n314) );
  VHSR_IN_2 U255 ( .I(b[6]), .ZN(n279) );
  VHSR_IN_2 U256 ( .I(a[2]), .ZN(n313) );
  VHSR_OAI22_2 U257 ( .A1(n279), .A2(n314), .B1(n278), .B2(n313), .ZN(n240) );
  VHSR_AOI22_2 U258 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n231) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[4]), .A2(a[2]), .ZN(n260) );
  VHSR_NAND3_2 U260 ( .A1(a[3]), .A2(b[5]), .A3(n260), .ZN(n230) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[7]), .A2(a[2]), .ZN(n225) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[6]), .A2(a[1]), .ZN(n227) );
  VHSR_OAI22_2 U263 ( .A1(n231), .A2(n230), .B1(n225), .B2(n227), .ZN(n232) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[4]), .A2(a[0]), .ZN(n389) );
  VHSR_NAND3_2 U265 ( .A1(a[1]), .A2(b[5]), .A3(n389), .ZN(n259) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[6]), .A2(a[0]), .ZN(n258) );
  VHSR_MAOI222_2 U267 ( .A(n260), .B(n259), .C(n258), .ZN(n257) );
  VHSR_NAND4_2 U268 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_IN_2 U269 ( .I(b[5]), .ZN(n275) );
  VHSR_AND2_2 U270 ( .A1(n237), .A2(n226), .Z(n229) );
  VHSR_IN_2 U271 ( .I(a[0]), .ZN(n396) );
  VHSR_OAI21_2 U272 ( .A1(n278), .A2(n396), .B(n227), .ZN(n228) );
  VHSR_IN_2 U273 ( .I(a[1]), .ZN(n394) );
  VHSR_NOR3_2 U274 ( .A1(n275), .A2(n394), .A3(n389), .ZN(n267) );
  VHSR_AND2_2 U275 ( .A1(n257), .A2(n256), .Z(n255) );
  VHSR_AD1_1 U276 ( .A(n229), .B(n228), .CI(n267), .CO(n246), .S(n256) );
  VHSR_AOI21_2 U277 ( .A1(n231), .A2(n230), .B(n232), .ZN(n249) );
  VHSR_OAI32_2 U278 ( .A1(n232), .A2(n255), .A3(n246), .B1(n249), .B2(n232), 
        .ZN(n238) );
  VHSR_CLKNAND2_2 U279 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U280 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U281 ( .A1(n278), .A2(n314), .A3(n235), .ZN(n293) );
  VHSR_AOI21_2 U282 ( .A1(n234), .A2(n233), .B(n294), .ZN(n297) );
  VHSR_OAI32_2 U283 ( .A1(n293), .A2(n314), .A3(n278), .B1(n235), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U284 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U285 ( .A1(n240), .A2(n239), .ZN(n304) );
  VHSR_OAI21_2 U286 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U287 ( .A1(n245), .A2(n244), .ZN(n303) );
  VHSR_NOR2_1 U288 ( .A1(n255), .A2(n246), .ZN(n248) );
  VHSR_AOI22_2 U289 ( .A1(n255), .A2(n246), .B1(n249), .B2(n248), .ZN(n247) );
  VHSR_OAI21_2 U290 ( .A1(n249), .A2(n248), .B(n247), .ZN(n309) );
  VHSR_AOI21_2 U291 ( .A1(n252), .A2(n251), .B(n250), .ZN(n308) );
  VHSR_AOI21_2 U292 ( .A1(n261), .A2(n254), .B(n253), .ZN(n324) );
  VHSR_IAO21_2 U293 ( .A1(n257), .A2(n256), .B(n255), .ZN(n323) );
  VHSR_AOI31_2 U294 ( .A1(n260), .A2(n259), .A3(n258), .B(n257), .ZN(n331) );
  VHSR_OAI31_2 U295 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n265) );
  VHSR_IN_2 U296 ( .I(n265), .ZN(n330) );
  VHSR_AOI22_2 U297 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n266) );
  VHSR_NOR2_1 U298 ( .A1(n267), .A2(n266), .ZN(n336) );
  VHSR_CLKNAND2_2 U299 ( .A1(a[4]), .A2(b[4]), .ZN(n283) );
  VHSR_IN_2 U300 ( .I(n283), .ZN(n365) );
  VHSR_NOR2_1 U301 ( .A1(n395), .A2(n396), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U302 ( .A1(n365), .A2(product[0]), .ZN(n388) );
  VHSR_IN_2 U303 ( .I(n388), .ZN(n335) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[1]), .ZN(n268) );
  VHSR_OAI32_2 U305 ( .A1(n269), .A2(n395), .A3(n281), .B1(n268), .B2(n269), 
        .ZN(n334) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[6]), .A2(b[4]), .ZN(n301) );
  VHSR_NAND3_2 U307 ( .A1(a[7]), .A2(b[5]), .A3(n301), .ZN(n272) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[6]), .ZN(n300) );
  VHSR_NAND3_2 U309 ( .A1(b[7]), .A2(a[5]), .A3(n300), .ZN(n271) );
  VHSR_CLKNAND2_2 U310 ( .A1(n272), .A2(n271), .ZN(n274) );
  VHSR_IN_2 U311 ( .I(n383), .ZN(n356) );
  VHSR_MAOI222_2 U312 ( .A(n356), .B(n272), .C(n271), .ZN(n340) );
  VHSR_IN_2 U313 ( .I(n340), .ZN(n273) );
  VHSR_OAI21_2 U314 ( .A1(n383), .A2(n274), .B(n273), .ZN(n289) );
  VHSR_NOR3_2 U315 ( .A1(n281), .A2(n275), .A3(n283), .ZN(n305) );
  VHSR_NOR3_2 U316 ( .A1(n276), .A2(n301), .A3(n275), .ZN(n348) );
  VHSR_AOI22_2 U317 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n277) );
  VHSR_NOR2_1 U318 ( .A1(n348), .A2(n277), .ZN(n285) );
  VHSR_NOR4_2 U319 ( .A1(n281), .A2(n280), .A3(n279), .A4(n278), .ZN(n346) );
  VHSR_AOI22_2 U320 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n282) );
  VHSR_NOR2_1 U321 ( .A1(n346), .A2(n282), .ZN(n284) );
  VHSR_NAND3_2 U322 ( .A1(b[5]), .A2(a[5]), .A3(n283), .ZN(n299) );
  VHSR_MAOI222_2 U323 ( .A(n301), .B(n300), .C(n299), .ZN(n298) );
  VHSR_AND2_2 U324 ( .A1(n291), .A2(n298), .Z(n290) );
  VHSR_AD1_1 U325 ( .A(n305), .B(n285), .CI(n284), .CO(n286), .S(n291) );
  VHSR_NOR2_1 U326 ( .A1(n290), .A2(n286), .ZN(n288) );
  VHSR_CLKNAND2_2 U327 ( .A1(n290), .A2(n286), .ZN(n287) );
  VHSR_NOR2_1 U328 ( .A1(n288), .A2(n289), .ZN(n341) );
  VHSR_AOI22_2 U329 ( .A1(n289), .A2(n288), .B1(n287), .B2(n341), .ZN(n381) );
  VHSR_IAO21_2 U330 ( .A1(n291), .A2(n298), .B(n290), .ZN(n379) );
  VHSR_AD1_1 U331 ( .A(n294), .B(n293), .CI(n292), .CO(n382), .S(n378) );
  VHSR_AD1_1 U332 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n360) );
  VHSR_AOI31_2 U333 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n359) );
  VHSR_AD1_1 U334 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n363) );
  VHSR_AOI22_2 U335 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n306) );
  VHSR_NOR2_1 U336 ( .A1(n306), .A2(n305), .ZN(n362) );
  VHSR_AD1_1 U337 ( .A(n309), .B(n308), .CI(n307), .CO(n302), .S(n366) );
  VHSR_NOR4_2 U338 ( .A1(n315), .A2(n312), .A3(n394), .A4(n396), .ZN(n339) );
  VHSR_CLKNAND2_2 U339 ( .A1(b[3]), .A2(a[3]), .ZN(n326) );
  VHSR_IN_2 U340 ( .I(n328), .ZN(n317) );
  VHSR_AOI22_2 U341 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n310) );
  VHSR_IAO21_2 U342 ( .A1(n326), .A2(n317), .B(n310), .ZN(n338) );
  VHSR_CLKNAND2_2 U343 ( .A1(b[2]), .A2(a[1]), .ZN(n311) );
  VHSR_OAI32_2 U344 ( .A1(n339), .A2(n396), .A3(n315), .B1(n311), .B2(n339), 
        .ZN(n370) );
  VHSR_AOI22_2 U345 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n400) );
  VHSR_NOR3_2 U346 ( .A1(n400), .A2(n396), .A3(n312), .ZN(n398) );
  VHSR_OAI22_2 U347 ( .A1(n397), .A2(n313), .B1(n395), .B2(n314), .ZN(n369) );
  VHSR_IN_2 U348 ( .I(n333), .ZN(n321) );
  VHSR_NOR2_1 U349 ( .A1(n397), .A2(n314), .ZN(n316) );
  VHSR_AOI211_2 U350 ( .A1(b[2]), .A2(a[0]), .B(n315), .C(n394), .ZN(n318) );
  VHSR_MAOI222_2 U351 ( .A(n316), .B(n328), .C(n318), .ZN(n320) );
  VHSR_AOI32_2 U352 ( .A1(a[3]), .A2(n320), .A3(b[1]), .B1(n319), .B2(n320), 
        .ZN(n332) );
  VHSR_OAI21_2 U353 ( .A1(n321), .A2(n332), .B(n320), .ZN(n337) );
  VHSR_IAO21_2 U354 ( .A1(n328), .A2(n327), .B(n326), .ZN(n376) );
  VHSR_AD1_1 U355 ( .A(n324), .B(n323), .CI(n322), .CO(n307), .S(n375) );
  VHSR_OAI21_2 U356 ( .A1(n328), .A2(n326), .B(n327), .ZN(n325) );
  VHSR_OAI31_2 U357 ( .A1(n328), .A2(n327), .A3(n326), .B(n325), .ZN(n373) );
  VHSR_AD1_1 U358 ( .A(n331), .B(n330), .CI(n329), .CO(n322), .S(n372) );
  VHSR_CLKNAND2_2 U359 ( .A1(a[4]), .A2(b[0]), .ZN(n390) );
  VHSR_CLKXOR2_2 U360 ( .A1(n333), .A2(n332), .Z(n393) );
  VHSR_AOI211_2 U361 ( .A1(n390), .A2(n389), .B(n335), .C(n393), .ZN(n391) );
  VHSR_AD1_1 U362 ( .A(n336), .B(n335), .CI(n334), .CO(n329), .S(n368) );
  VHSR_AD1_1 U363 ( .A(n339), .B(n338), .CI(n337), .CO(n327), .S(n367) );
  VHSR_CLKNAND2_2 U364 ( .A1(a[6]), .A2(b[7]), .ZN(n343) );
  VHSR_AOI21_2 U365 ( .A1(a[7]), .A2(b[6]), .B(n343), .ZN(n342) );
  VHSR_AOI31_2 U366 ( .A1(a[7]), .A2(n343), .A3(b[6]), .B(n342), .ZN(n344) );
  VHSR_IN_2 U367 ( .I(n344), .ZN(n345) );
  VHSR_OR2_2 U368 ( .A1(n346), .A2(n345), .Z(n347) );
  VHSR_MAOI222_2 U369 ( .A(n348), .B(n346), .C(n345), .ZN(n355) );
  VHSR_OAI21_2 U370 ( .A1(n348), .A2(n347), .B(n355), .ZN(n352) );
  VHSR_CLKXOR2_2 U371 ( .A1(n353), .A2(n352), .Z(n349) );
  VHSR_CLKNAND2_2 U372 ( .A1(n350), .A2(n349), .ZN(n385) );
  VHSR_OAI21_2 U373 ( .A1(n350), .A2(n349), .B(n385), .ZN(n351) );
  VHSR_CLKNAND2_2 U374 ( .A1(a[7]), .A2(b[7]), .ZN(n384) );
  VHSR_NOR2_1 U375 ( .A1(n353), .A2(n352), .ZN(n354) );
  VHSR_AND3_2 U376 ( .A1(n386), .A2(n356), .A3(n385), .Z(n357) );
  VHSR_NOR2_1 U377 ( .A1(n384), .A2(n357), .ZN(product[15]) );
  VHSR_AD1_1 U378 ( .A(n373), .B(n372), .CI(n371), .CO(n374), .S(product[6])
         );
  VHSR_AD1_1 U379 ( .A(n376), .B(n375), .CI(n374), .CO(n364), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U380 ( .A(n379), .B(n378), .CI(n377), .CO(n380), .S(product[11])
         );
  VHSR_AD1_1 U381 ( .A(n382), .B(n381), .CI(n380), .CO(n350), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U382 ( .A1(n384), .A2(n383), .ZN(n387) );
  VHSR_XOR3_2 U383 ( .A1(n387), .A2(n386), .A3(n385), .Z(product[14]) );
  VHSR_AOI21_2 U384 ( .A1(n393), .A2(n392), .B(n391), .ZN(product[4]) );
  VHSR_OAI22_2 U385 ( .A1(n397), .A2(n396), .B1(n395), .B2(n394), .ZN(
        product[1]) );
  VHSR_CLKNAND2_2 U386 ( .A1(b[2]), .A2(a[0]), .ZN(n399) );
  VHSR_AOI21_2 U387 ( .A1(n400), .A2(n399), .B(n398), .ZN(product[2]) );
endmodule


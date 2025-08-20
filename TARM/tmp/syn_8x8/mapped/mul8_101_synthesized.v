
module mul8_101 ( a, b, product );
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
         n391, n392, n393, n394, n395, n396, n397, n398;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U204 ( .A1(n265), .B1(a[5]), .B2(b[3]), .ZN(n221) );
  VHSR_INAND2_2 U205 ( .A1(n270), .B1(n216), .ZN(n217) );
  VHSR_INOR2_2 U206 ( .A1(n233), .B1(n250), .ZN(n238) );
  VHSR_INOR2_2 U207 ( .A1(n219), .B1(n256), .ZN(n248) );
  VHSR_NOR2_1 U208 ( .A1(n276), .A2(n234), .ZN(n294) );
  VHSR_INOR2_2 U209 ( .A1(n356), .B1(n355), .ZN(n387) );
  VHSR_IN_2 U210 ( .I(n352), .ZN(product[13]) );
  VHSR_INAND2_1 U211 ( .A1(n347), .B1(n345), .ZN(n348) );
  VHSR_AD1_1 U212 ( .A(n364), .B(n363), .CI(n362), .CO(n359), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U213 ( .A(n371), .B(n398), .CI(n370), .CO(n331), .S(product[3])
         );
  VHSR_AD1_1 U214 ( .A(n389), .B(n369), .CI(n368), .CO(n372), .S(product[5])
         );
  VHSR_AD1_1 U215 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U216 ( .A(n361), .B(n360), .CI(n359), .CO(n375), .S(product[9])
         );
  VHSR_IN_2 U217 ( .I(a[7]), .ZN(n276) );
  VHSR_IN_2 U218 ( .I(b[2]), .ZN(n396) );
  VHSR_IN_2 U219 ( .I(a[4]), .ZN(n280) );
  VHSR_NOR2_1 U220 ( .A1(n396), .A2(n280), .ZN(n265) );
  VHSR_NAND3_2 U221 ( .A1(a[6]), .A2(a[7]), .A3(b[1]), .ZN(n220) );
  VHSR_OAI21_2 U222 ( .A1(a[6]), .A2(a[7]), .B(b[2]), .ZN(n223) );
  VHSR_MAOI222_2 U223 ( .A(n221), .B(n220), .C(n223), .ZN(n225) );
  VHSR_IN_2 U224 ( .I(a[6]), .ZN(n215) );
  VHSR_IN_2 U225 ( .I(b[1]), .ZN(n394) );
  VHSR_NOR2_1 U226 ( .A1(n215), .A2(n394), .ZN(n218) );
  VHSR_IN_2 U227 ( .I(a[5]), .ZN(n281) );
  VHSR_IN_2 U228 ( .I(b[0]), .ZN(n393) );
  VHSR_NOR4_2 U229 ( .A1(n281), .A2(n280), .A3(n394), .A4(n393), .ZN(n270) );
  VHSR_NAND4_2 U230 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n243) );
  VHSR_IN_2 U231 ( .I(b[3]), .ZN(n316) );
  VHSR_NOR2_1 U232 ( .A1(n316), .A2(n280), .ZN(n213) );
  VHSR_AOI32_2 U233 ( .A1(a[5]), .A2(n243), .A3(b[2]), .B1(n213), .B2(n243), 
        .ZN(n216) );
  VHSR_IN_2 U234 ( .I(n216), .ZN(n214) );
  VHSR_MAOI222_2 U235 ( .A(n218), .B(n270), .C(n214), .ZN(n219) );
  VHSR_AOI211_2 U236 ( .A1(a[4]), .A2(b[0]), .B(n281), .C(n394), .ZN(n264) );
  VHSR_NOR2_1 U237 ( .A1(n215), .A2(n393), .ZN(n263) );
  VHSR_MAOI222_2 U238 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_OAI21_2 U239 ( .A1(n218), .A2(n217), .B(n219), .ZN(n257) );
  VHSR_NOR2_1 U240 ( .A1(n262), .A2(n257), .ZN(n256) );
  VHSR_AND2_2 U241 ( .A1(n221), .A2(n220), .Z(n222) );
  VHSR_AOI21_2 U242 ( .A1(n223), .A2(n222), .B(n225), .ZN(n224) );
  VHSR_IN_2 U243 ( .I(n224), .ZN(n247) );
  VHSR_NOR2_1 U244 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_NOR2_1 U245 ( .A1(n225), .A2(n246), .ZN(n242) );
  VHSR_CLKNAND2_2 U246 ( .A1(n242), .A2(n243), .ZN(n241) );
  VHSR_NAND3_2 U247 ( .A1(a[6]), .A2(b[3]), .A3(n241), .ZN(n234) );
  VHSR_IN_2 U248 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U249 ( .I(a[3]), .ZN(n319) );
  VHSR_IN_2 U250 ( .I(b[6]), .ZN(n279) );
  VHSR_IN_2 U251 ( .I(a[2]), .ZN(n317) );
  VHSR_OAI22_2 U252 ( .A1(n279), .A2(n319), .B1(n278), .B2(n317), .ZN(n240) );
  VHSR_NOR2_1 U253 ( .A1(n278), .A2(n317), .ZN(n227) );
  VHSR_IN_2 U254 ( .I(a[1]), .ZN(n392) );
  VHSR_NOR2_1 U255 ( .A1(n279), .A2(n392), .ZN(n226) );
  VHSR_IN_2 U256 ( .I(b[5]), .ZN(n275) );
  VHSR_AOI211_2 U257 ( .A1(b[4]), .A2(a[2]), .B(n275), .C(n319), .ZN(n232) );
  VHSR_OAI22_2 U258 ( .A1(n279), .A2(n317), .B1(n278), .B2(n392), .ZN(n231) );
  VHSR_AOI22_2 U259 ( .A1(n227), .A2(n226), .B1(n232), .B2(n231), .ZN(n233) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[4]), .A2(a[2]), .ZN(n261) );
  VHSR_IN_2 U261 ( .I(b[4]), .ZN(n332) );
  VHSR_IN_2 U262 ( .I(a[0]), .ZN(n397) );
  VHSR_OAI211_2 U263 ( .A1(n332), .A2(n397), .B(b[5]), .C(a[1]), .ZN(n260) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[6]), .A2(a[0]), .ZN(n259) );
  VHSR_MAOI222_2 U265 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_NOR4_2 U266 ( .A1(n332), .A2(n275), .A3(n392), .A4(n397), .ZN(n268) );
  VHSR_NAND4_2 U267 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_OAI22_2 U268 ( .A1(n332), .A2(n319), .B1(n275), .B2(n317), .ZN(n228) );
  VHSR_AND2_2 U269 ( .A1(n237), .A2(n228), .Z(n230) );
  VHSR_OAI22_2 U270 ( .A1(n279), .A2(n392), .B1(n278), .B2(n397), .ZN(n229) );
  VHSR_AND2_2 U271 ( .A1(n258), .A2(n255), .Z(n254) );
  VHSR_AD1_1 U272 ( .A(n268), .B(n230), .CI(n229), .CO(n249), .S(n255) );
  VHSR_NOR2_1 U273 ( .A1(n254), .A2(n249), .ZN(n252) );
  VHSR_OAI21_2 U274 ( .A1(n232), .A2(n231), .B(n233), .ZN(n253) );
  VHSR_NOR2_1 U275 ( .A1(n252), .A2(n253), .ZN(n250) );
  VHSR_CLKNAND2_2 U276 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U277 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U278 ( .A1(n278), .A2(n319), .A3(n235), .ZN(n293) );
  VHSR_OAI32_2 U279 ( .A1(n294), .A2(n276), .A3(n316), .B1(n234), .B2(n294), 
        .ZN(n297) );
  VHSR_OAI32_2 U280 ( .A1(n293), .A2(n319), .A3(n278), .B1(n235), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U281 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U282 ( .A1(n240), .A2(n239), .ZN(n304) );
  VHSR_OAI21_2 U283 ( .A1(n243), .A2(n242), .B(n241), .ZN(n245) );
  VHSR_CLKNAND2_2 U284 ( .A1(b[3]), .A2(a[6]), .ZN(n244) );
  VHSR_CLKXOR2_2 U285 ( .A1(n245), .A2(n244), .Z(n303) );
  VHSR_AOI21_2 U286 ( .A1(n248), .A2(n247), .B(n246), .ZN(n309) );
  VHSR_CLKNAND2_2 U287 ( .A1(n254), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U288 ( .A1(n253), .A2(n252), .B1(n251), .B2(n250), .ZN(n308) );
  VHSR_IAO21_2 U289 ( .A1(n258), .A2(n255), .B(n254), .ZN(n312) );
  VHSR_AOI21_2 U290 ( .A1(n262), .A2(n257), .B(n256), .ZN(n311) );
  VHSR_AOI31_2 U291 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n325) );
  VHSR_OAI31_2 U292 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n266) );
  VHSR_IN_2 U293 ( .I(n266), .ZN(n324) );
  VHSR_CLKNAND2_2 U294 ( .A1(b[5]), .A2(a[0]), .ZN(n267) );
  VHSR_OAI32_2 U295 ( .A1(n268), .A2(n392), .A3(n332), .B1(n267), .B2(n268), 
        .ZN(n340) );
  VHSR_CLKNAND2_2 U296 ( .A1(a[4]), .A2(b[1]), .ZN(n269) );
  VHSR_OAI32_2 U297 ( .A1(n270), .A2(n393), .A3(n281), .B1(n269), .B2(n270), 
        .ZN(n339) );
  VHSR_CLKNAND2_2 U298 ( .A1(a[4]), .A2(b[4]), .ZN(n283) );
  VHSR_IN_2 U299 ( .I(n283), .ZN(n363) );
  VHSR_NOR2_1 U300 ( .A1(n393), .A2(n397), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U301 ( .A1(n363), .A2(product[0]), .ZN(n334) );
  VHSR_IN_2 U302 ( .I(n334), .ZN(n338) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[6]), .A2(b[6]), .ZN(n357) );
  VHSR_IN_2 U304 ( .I(n357), .ZN(n384) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[6]), .A2(b[4]), .ZN(n301) );
  VHSR_NAND3_2 U306 ( .A1(a[7]), .A2(b[5]), .A3(n301), .ZN(n272) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[4]), .A2(b[6]), .ZN(n300) );
  VHSR_NAND3_2 U308 ( .A1(b[7]), .A2(a[5]), .A3(n300), .ZN(n271) );
  VHSR_CLKNAND2_2 U309 ( .A1(n272), .A2(n271), .ZN(n274) );
  VHSR_MAOI222_2 U310 ( .A(n357), .B(n272), .C(n271), .ZN(n341) );
  VHSR_IN_2 U311 ( .I(n341), .ZN(n273) );
  VHSR_OAI21_2 U312 ( .A1(n384), .A2(n274), .B(n273), .ZN(n289) );
  VHSR_NOR3_2 U313 ( .A1(n281), .A2(n275), .A3(n283), .ZN(n305) );
  VHSR_NOR3_2 U314 ( .A1(n276), .A2(n301), .A3(n275), .ZN(n349) );
  VHSR_AOI22_2 U315 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n277) );
  VHSR_NOR2_1 U316 ( .A1(n349), .A2(n277), .ZN(n285) );
  VHSR_NOR4_2 U317 ( .A1(n281), .A2(n280), .A3(n279), .A4(n278), .ZN(n347) );
  VHSR_AOI22_2 U318 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n282) );
  VHSR_NOR2_1 U319 ( .A1(n347), .A2(n282), .ZN(n284) );
  VHSR_NAND3_2 U320 ( .A1(b[5]), .A2(a[5]), .A3(n283), .ZN(n299) );
  VHSR_MAOI222_2 U321 ( .A(n301), .B(n300), .C(n299), .ZN(n298) );
  VHSR_AND2_2 U322 ( .A1(n291), .A2(n298), .Z(n290) );
  VHSR_AD1_1 U323 ( .A(n305), .B(n285), .CI(n284), .CO(n286), .S(n291) );
  VHSR_NOR2_1 U324 ( .A1(n290), .A2(n286), .ZN(n288) );
  VHSR_CLKNAND2_2 U325 ( .A1(n290), .A2(n286), .ZN(n287) );
  VHSR_NOR2_1 U326 ( .A1(n288), .A2(n289), .ZN(n342) );
  VHSR_AOI22_2 U327 ( .A1(n289), .A2(n288), .B1(n287), .B2(n342), .ZN(n382) );
  VHSR_IAO21_2 U328 ( .A1(n291), .A2(n298), .B(n290), .ZN(n380) );
  VHSR_AD1_1 U329 ( .A(n294), .B(n293), .CI(n292), .CO(n383), .S(n379) );
  VHSR_AD1_1 U330 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n377) );
  VHSR_AOI31_2 U331 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n376) );
  VHSR_AD1_1 U332 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n361) );
  VHSR_AOI22_2 U333 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n306) );
  VHSR_NOR2_1 U334 ( .A1(n306), .A2(n305), .ZN(n360) );
  VHSR_AD1_1 U335 ( .A(n309), .B(n308), .CI(n307), .CO(n302), .S(n364) );
  VHSR_AD1_1 U336 ( .A(n312), .B(n311), .CI(n310), .CO(n307), .S(n367) );
  VHSR_NOR2_1 U337 ( .A1(n396), .A2(n317), .ZN(n328) );
  VHSR_AOI22_2 U338 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n313) );
  VHSR_AOI31_2 U339 ( .A1(a[3]), .A2(b[3]), .A3(n328), .B(n313), .ZN(n337) );
  VHSR_NOR2_1 U340 ( .A1(n394), .A2(n319), .ZN(n315) );
  VHSR_NOR2_1 U341 ( .A1(n316), .A2(n392), .ZN(n314) );
  VHSR_MAOI222_2 U342 ( .A(n328), .B(n315), .C(n314), .ZN(n321) );
  VHSR_OAI22_2 U343 ( .A1(n316), .A2(n397), .B1(n396), .B2(n392), .ZN(n371) );
  VHSR_AOI22_2 U344 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n395) );
  VHSR_NOR3_2 U345 ( .A1(n395), .A2(n397), .A3(n396), .ZN(n398) );
  VHSR_OAI22_2 U346 ( .A1(n394), .A2(n317), .B1(n393), .B2(n319), .ZN(n370) );
  VHSR_IN_2 U347 ( .I(n321), .ZN(n320) );
  VHSR_AOI21_2 U348 ( .A1(a[1]), .A2(b[3]), .B(n328), .ZN(n318) );
  VHSR_OAI32_2 U349 ( .A1(n320), .A2(n319), .A3(n394), .B1(n318), .B2(n320), 
        .ZN(n330) );
  VHSR_CLKNAND2_2 U350 ( .A1(n331), .A2(n330), .ZN(n329) );
  VHSR_CLKNAND2_2 U351 ( .A1(n321), .A2(n329), .ZN(n336) );
  VHSR_AND2_2 U352 ( .A1(n337), .A2(n336), .Z(n335) );
  VHSR_OAI211_2 U353 ( .A1(n328), .A2(n335), .B(a[3]), .C(b[3]), .ZN(n322) );
  VHSR_IN_2 U354 ( .I(n322), .ZN(n366) );
  VHSR_AD1_1 U355 ( .A(n325), .B(n324), .CI(n323), .CO(n310), .S(n374) );
  VHSR_CLKNAND2_2 U356 ( .A1(b[3]), .A2(a[3]), .ZN(n327) );
  VHSR_CLKNAND2_2 U357 ( .A1(n335), .A2(n327), .ZN(n326) );
  VHSR_OAI31_2 U358 ( .A1(n328), .A2(n335), .A3(n327), .B(n326), .ZN(n373) );
  VHSR_OAI21_2 U359 ( .A1(n331), .A2(n330), .B(n329), .ZN(n391) );
  VHSR_NOR2_1 U360 ( .A1(n332), .A2(n397), .ZN(n333) );
  VHSR_AOI32_2 U361 ( .A1(b[0]), .A2(n334), .A3(a[4]), .B1(n333), .B2(n334), 
        .ZN(n390) );
  VHSR_NOR2_1 U362 ( .A1(n391), .A2(n390), .ZN(n389) );
  VHSR_IAO21_2 U363 ( .A1(n337), .A2(n336), .B(n335), .ZN(n369) );
  VHSR_AD1_1 U364 ( .A(n340), .B(n339), .CI(n338), .CO(n323), .S(n368) );
  VHSR_NOR2_1 U365 ( .A1(n342), .A2(n341), .ZN(n354) );
  VHSR_CLKNAND2_2 U366 ( .A1(a[7]), .A2(b[6]), .ZN(n344) );
  VHSR_AOI21_2 U367 ( .A1(a[6]), .A2(b[7]), .B(n344), .ZN(n343) );
  VHSR_AOI31_2 U368 ( .A1(a[6]), .A2(n344), .A3(b[7]), .B(n343), .ZN(n345) );
  VHSR_IN_2 U369 ( .I(n345), .ZN(n346) );
  VHSR_MAOI222_2 U370 ( .A(n349), .B(n347), .C(n346), .ZN(n356) );
  VHSR_OAI21_2 U371 ( .A1(n349), .A2(n348), .B(n356), .ZN(n353) );
  VHSR_CLKXOR2_2 U372 ( .A1(n354), .A2(n353), .Z(n350) );
  VHSR_CLKNAND2_2 U373 ( .A1(n351), .A2(n350), .ZN(n386) );
  VHSR_OAI21_2 U374 ( .A1(n351), .A2(n350), .B(n386), .ZN(n352) );
  VHSR_CLKNAND2_2 U375 ( .A1(a[7]), .A2(b[7]), .ZN(n385) );
  VHSR_NOR2_1 U376 ( .A1(n354), .A2(n353), .ZN(n355) );
  VHSR_AND3_2 U377 ( .A1(n387), .A2(n357), .A3(n386), .Z(n358) );
  VHSR_NOR2_1 U378 ( .A1(n385), .A2(n358), .ZN(product[15]) );
  VHSR_AD1_1 U379 ( .A(n374), .B(n373), .CI(n372), .CO(n365), .S(product[6])
         );
  VHSR_AD1_1 U380 ( .A(n377), .B(n376), .CI(n375), .CO(n378), .S(product[10])
         );
  VHSR_AD1_1 U381 ( .A(n380), .B(n379), .CI(n378), .CO(n381), .S(product[11])
         );
  VHSR_AD1_1 U382 ( .A(n383), .B(n382), .CI(n381), .CO(n351), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U383 ( .A1(n385), .A2(n384), .ZN(n388) );
  VHSR_XOR3_2 U384 ( .A1(n388), .A2(n387), .A3(n386), .Z(product[14]) );
  VHSR_AOI21_2 U385 ( .A1(n391), .A2(n390), .B(n389), .ZN(product[4]) );
  VHSR_OAI22_2 U386 ( .A1(n394), .A2(n397), .B1(n393), .B2(n392), .ZN(
        product[1]) );
  VHSR_OAI32_2 U387 ( .A1(n398), .A2(n397), .A3(n396), .B1(n395), .B2(n398), 
        .ZN(product[2]) );
endmodule


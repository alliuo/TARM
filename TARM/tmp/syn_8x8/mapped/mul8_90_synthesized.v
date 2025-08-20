
module mul8_90 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n211, n212,
         n213, n214, n215, n216, n217, n218, n219, n220, n221, n222, n223,
         n224, n225, n226, n227, n228, n229, n230, n231, n232, n233, n234,
         n235, n236, n237, n238, n239, n240, n241, n242, n243, n244, n245,
         n246, n247, n248, n249, n250, n251, n252, n253, n254, n255, n256,
         n257, n258, n259, n260, n261, n262, n263, n264, n265, n266, n267,
         n268, n269, n270, n271, n272, n273, n274, n275, n276, n277, n278,
         n279, n280, n281, n282, n283, n284, n285, n286, n287, n288, n289,
         n290, n291, n292, n293, n294, n295, n296, n297, n298, n299, n300,
         n301, n302, n303, n304, n305, n306, n307, n308, n309, n310, n311,
         n312, n313, n314, n315, n316, n317, n318, n319, n320, n321, n322,
         n323, n324, n325, n326, n327, n328, n329, n330, n331, n332, n333,
         n334, n335, n336, n337, n338, n339, n340, n341, n342, n343, n344,
         n345, n346, n347, n348, n349, n350, n351, n352, n353, n354, n355,
         n356, n357, n358, n359, n360, n361, n362, n363, n364, n365, n366,
         n367, n368, n369, n370, n371, n372, n373, n374, n375, n376, n377,
         n378, n379, n380, n381, n382, n383, n384, n385, n386, n387, n388,
         n389, n390, n391, n392, n393, n394, n395, n396;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U203 ( .A1(n225), .B1(n249), .ZN(n243) );
  VHSR_INOR2_2 U204 ( .A1(n220), .B1(n252), .ZN(n251) );
  VHSR_INAND2_2 U205 ( .A1(n318), .B1(n335), .ZN(n331) );
  VHSR_INOR3_2 U206 ( .A1(n360), .B1(n280), .B2(n281), .ZN(n303) );
  VHSR_NOR2_1 U207 ( .A1(n234), .A2(n274), .ZN(n292) );
  VHSR_INOR2_2 U208 ( .A1(n353), .B1(n352), .ZN(n384) );
  VHSR_IN_2 U209 ( .I(n349), .ZN(product[13]) );
  VHSR_INAND2_1 U210 ( .A1(n344), .B1(n342), .ZN(n345) );
  VHSR_AD1_1 U211 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(product[6])
         );
  VHSR_AD1_1 U212 ( .A(n361), .B(n360), .CI(n359), .CO(n356), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U213 ( .A(n371), .B(n396), .CI(n370), .CO(n337), .S(product[3])
         );
  VHSR_AD1_1 U214 ( .A(n369), .B(n368), .CI(n390), .CO(n365), .S(product[5])
         );
  VHSR_AD1_1 U215 ( .A(n364), .B(n363), .CI(n362), .CO(n359), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U216 ( .A(n358), .B(n357), .CI(n356), .CO(n372), .S(product[9])
         );
  VHSR_IN_2 U217 ( .I(a[1]), .ZN(n310) );
  VHSR_IN_2 U218 ( .I(b[0]), .ZN(n313) );
  VHSR_NOR2_1 U219 ( .A1(n310), .A2(n313), .ZN(product[1]) );
  VHSR_IN_2 U220 ( .I(a[0]), .ZN(n394) );
  VHSR_IN_2 U221 ( .I(b[1]), .ZN(n315) );
  VHSR_NOR2_1 U222 ( .A1(n394), .A2(n315), .ZN(product[0]) );
  VHSR_IN_2 U223 ( .I(a[7]), .ZN(n274) );
  VHSR_NOR2_1 U224 ( .A1(n274), .A2(n315), .ZN(n212) );
  VHSR_IN_2 U225 ( .I(a[6]), .ZN(n223) );
  VHSR_IN_2 U226 ( .I(b[2]), .ZN(n395) );
  VHSR_NOR2_1 U227 ( .A1(n223), .A2(n395), .ZN(n211) );
  VHSR_IN_2 U228 ( .I(a[5]), .ZN(n281) );
  VHSR_IN_2 U229 ( .I(b[3]), .ZN(n309) );
  VHSR_AOI211_2 U230 ( .A1(a[4]), .A2(b[2]), .B(n281), .C(n309), .ZN(n221) );
  VHSR_MAOI222_2 U231 ( .A(n212), .B(n211), .C(n221), .ZN(n225) );
  VHSR_IN_2 U232 ( .I(a[4]), .ZN(n276) );
  VHSR_NOR4_2 U233 ( .A1(n276), .A2(n281), .A3(n315), .A4(n313), .ZN(n268) );
  VHSR_NOR2_1 U234 ( .A1(n223), .A2(n315), .ZN(n216) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[5]), .A2(b[2]), .ZN(n214) );
  VHSR_AOI21_2 U236 ( .A1(a[4]), .A2(b[3]), .B(n214), .ZN(n213) );
  VHSR_AOI31_2 U237 ( .A1(a[4]), .A2(n214), .A3(b[3]), .B(n213), .ZN(n219) );
  VHSR_IN_2 U238 ( .I(n219), .ZN(n215) );
  VHSR_MAOI222_2 U239 ( .A(n268), .B(n216), .C(n215), .ZN(n220) );
  VHSR_NOR2_1 U240 ( .A1(n276), .A2(n395), .ZN(n263) );
  VHSR_AOI211_2 U241 ( .A1(a[4]), .A2(b[0]), .B(n281), .C(n315), .ZN(n262) );
  VHSR_AOI21_2 U242 ( .A1(n223), .A2(n274), .B(n313), .ZN(n261) );
  VHSR_MAOI222_2 U243 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NOR2_1 U244 ( .A1(n268), .A2(n216), .ZN(n218) );
  VHSR_AOI22_2 U245 ( .A1(n268), .A2(n216), .B1(n219), .B2(n218), .ZN(n217) );
  VHSR_OAI21_2 U246 ( .A1(n219), .A2(n218), .B(n217), .ZN(n253) );
  VHSR_NOR2_1 U247 ( .A1(n260), .A2(n253), .ZN(n252) );
  VHSR_IN_2 U248 ( .I(n221), .ZN(n222) );
  VHSR_OAI21_2 U249 ( .A1(n395), .A2(n223), .B(n222), .ZN(n224) );
  VHSR_AOI32_2 U250 ( .A1(b[1]), .A2(n225), .A3(a[7]), .B1(n224), .B2(n225), 
        .ZN(n250) );
  VHSR_NOR2_1 U251 ( .A1(n251), .A2(n250), .ZN(n249) );
  VHSR_NAND3_2 U252 ( .A1(a[5]), .A2(n263), .A3(b[3]), .ZN(n242) );
  VHSR_CLKNAND2_2 U253 ( .A1(a[6]), .A2(b[3]), .ZN(n241) );
  VHSR_IN_2 U254 ( .I(b[7]), .ZN(n277) );
  VHSR_IN_2 U255 ( .I(a[3]), .ZN(n314) );
  VHSR_IN_2 U256 ( .I(b[6]), .ZN(n278) );
  VHSR_IN_2 U257 ( .I(a[2]), .ZN(n316) );
  VHSR_OAI22_2 U258 ( .A1(n278), .A2(n314), .B1(n277), .B2(n316), .ZN(n240) );
  VHSR_AOI22_2 U259 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[4]), .A2(a[2]), .ZN(n259) );
  VHSR_NAND3_2 U261 ( .A1(a[3]), .A2(b[5]), .A3(n259), .ZN(n231) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[7]), .A2(a[2]), .ZN(n226) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[6]), .A2(a[1]), .ZN(n228) );
  VHSR_OAI22_2 U264 ( .A1(n232), .A2(n231), .B1(n226), .B2(n228), .ZN(n233) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[0]), .ZN(n388) );
  VHSR_NAND3_2 U266 ( .A1(a[1]), .A2(b[5]), .A3(n388), .ZN(n258) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[0]), .ZN(n257) );
  VHSR_MAOI222_2 U268 ( .A(n259), .B(n258), .C(n257), .ZN(n256) );
  VHSR_NAND4_2 U269 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_IN_2 U270 ( .I(b[4]), .ZN(n273) );
  VHSR_IN_2 U271 ( .I(b[5]), .ZN(n280) );
  VHSR_OAI22_2 U272 ( .A1(n273), .A2(n314), .B1(n280), .B2(n316), .ZN(n227) );
  VHSR_AND2_2 U273 ( .A1(n237), .A2(n227), .Z(n230) );
  VHSR_OAI21_2 U274 ( .A1(n277), .A2(n394), .B(n228), .ZN(n229) );
  VHSR_NOR3_2 U275 ( .A1(n280), .A2(n310), .A3(n388), .ZN(n266) );
  VHSR_AND2_2 U276 ( .A1(n256), .A2(n255), .Z(n254) );
  VHSR_AD1_1 U277 ( .A(n230), .B(n229), .CI(n266), .CO(n245), .S(n255) );
  VHSR_AOI21_2 U278 ( .A1(n232), .A2(n231), .B(n233), .ZN(n248) );
  VHSR_OAI32_2 U279 ( .A1(n233), .A2(n254), .A3(n245), .B1(n248), .B2(n233), 
        .ZN(n238) );
  VHSR_CLKNAND2_2 U280 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U281 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U282 ( .A1(n277), .A2(n314), .A3(n235), .ZN(n291) );
  VHSR_OAI32_2 U283 ( .A1(n292), .A2(n309), .A3(n274), .B1(n234), .B2(n292), 
        .ZN(n295) );
  VHSR_OAI32_2 U284 ( .A1(n291), .A2(n314), .A3(n277), .B1(n235), .B2(n291), 
        .ZN(n294) );
  VHSR_OAI21_2 U285 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U286 ( .A1(n240), .A2(n239), .ZN(n302) );
  VHSR_AD1_1 U287 ( .A(n243), .B(n242), .CI(n241), .CO(n234), .S(n244) );
  VHSR_IN_2 U288 ( .I(n244), .ZN(n301) );
  VHSR_NOR2_1 U289 ( .A1(n254), .A2(n245), .ZN(n247) );
  VHSR_AOI22_2 U290 ( .A1(n254), .A2(n245), .B1(n248), .B2(n247), .ZN(n246) );
  VHSR_OAI21_2 U291 ( .A1(n248), .A2(n247), .B(n246), .ZN(n307) );
  VHSR_AOI21_2 U292 ( .A1(n251), .A2(n250), .B(n249), .ZN(n306) );
  VHSR_AOI21_2 U293 ( .A1(n260), .A2(n253), .B(n252), .ZN(n322) );
  VHSR_IAO21_2 U294 ( .A1(n256), .A2(n255), .B(n254), .ZN(n321) );
  VHSR_AOI31_2 U295 ( .A1(n259), .A2(n258), .A3(n257), .B(n256), .ZN(n329) );
  VHSR_OAI31_2 U296 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n264) );
  VHSR_IN_2 U297 ( .I(n264), .ZN(n328) );
  VHSR_AOI22_2 U298 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n265) );
  VHSR_NOR2_1 U299 ( .A1(n266), .A2(n265), .ZN(n334) );
  VHSR_CLKNAND2_2 U300 ( .A1(a[4]), .A2(b[0]), .ZN(n387) );
  VHSR_NOR2_1 U301 ( .A1(n388), .A2(n387), .ZN(n386) );
  VHSR_AOI22_2 U302 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n267) );
  VHSR_NOR2_1 U303 ( .A1(n268), .A2(n267), .ZN(n333) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[6]), .A2(b[6]), .ZN(n354) );
  VHSR_IN_2 U305 ( .I(n354), .ZN(n381) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[6]), .A2(b[4]), .ZN(n299) );
  VHSR_NAND3_2 U307 ( .A1(a[7]), .A2(b[5]), .A3(n299), .ZN(n270) );
  VHSR_CLKNAND2_2 U308 ( .A1(b[6]), .A2(a[4]), .ZN(n298) );
  VHSR_NAND3_2 U309 ( .A1(b[7]), .A2(a[5]), .A3(n298), .ZN(n269) );
  VHSR_CLKNAND2_2 U310 ( .A1(n270), .A2(n269), .ZN(n272) );
  VHSR_MAOI222_2 U311 ( .A(n354), .B(n270), .C(n269), .ZN(n338) );
  VHSR_IN_2 U312 ( .I(n338), .ZN(n271) );
  VHSR_OAI21_2 U313 ( .A1(n381), .A2(n272), .B(n271), .ZN(n287) );
  VHSR_NOR2_1 U314 ( .A1(n273), .A2(n276), .ZN(n360) );
  VHSR_NOR3_2 U315 ( .A1(n274), .A2(n299), .A3(n280), .ZN(n346) );
  VHSR_AOI22_2 U316 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n275) );
  VHSR_NOR2_1 U317 ( .A1(n346), .A2(n275), .ZN(n283) );
  VHSR_NOR4_2 U318 ( .A1(n278), .A2(n277), .A3(n276), .A4(n281), .ZN(n344) );
  VHSR_AOI22_2 U319 ( .A1(b[6]), .A2(a[5]), .B1(b[7]), .B2(a[4]), .ZN(n279) );
  VHSR_NOR2_1 U320 ( .A1(n344), .A2(n279), .ZN(n282) );
  VHSR_OR3_2 U321 ( .A1(n360), .A2(n281), .A3(n280), .Z(n297) );
  VHSR_MAOI222_2 U322 ( .A(n299), .B(n298), .C(n297), .ZN(n296) );
  VHSR_AND2_2 U323 ( .A1(n289), .A2(n296), .Z(n288) );
  VHSR_AD1_1 U324 ( .A(n303), .B(n283), .CI(n282), .CO(n284), .S(n289) );
  VHSR_NOR2_1 U325 ( .A1(n288), .A2(n284), .ZN(n286) );
  VHSR_CLKNAND2_2 U326 ( .A1(n288), .A2(n284), .ZN(n285) );
  VHSR_NOR2_1 U327 ( .A1(n286), .A2(n287), .ZN(n339) );
  VHSR_AOI22_2 U328 ( .A1(n287), .A2(n286), .B1(n285), .B2(n339), .ZN(n379) );
  VHSR_IAO21_2 U329 ( .A1(n289), .A2(n296), .B(n288), .ZN(n377) );
  VHSR_AD1_1 U330 ( .A(n292), .B(n291), .CI(n290), .CO(n380), .S(n376) );
  VHSR_AD1_1 U331 ( .A(n295), .B(n294), .CI(n293), .CO(n290), .S(n374) );
  VHSR_AOI31_2 U332 ( .A1(n299), .A2(n298), .A3(n297), .B(n296), .ZN(n373) );
  VHSR_AD1_1 U333 ( .A(n302), .B(n301), .CI(n300), .CO(n293), .S(n358) );
  VHSR_AOI22_2 U334 ( .A1(b[4]), .A2(a[5]), .B1(b[5]), .B2(a[4]), .ZN(n304) );
  VHSR_NOR2_1 U335 ( .A1(n304), .A2(n303), .ZN(n357) );
  VHSR_AD1_1 U336 ( .A(n307), .B(n306), .CI(n305), .CO(n300), .S(n361) );
  VHSR_NOR2_1 U337 ( .A1(n316), .A2(n395), .ZN(n326) );
  VHSR_IN_2 U338 ( .I(n326), .ZN(n319) );
  VHSR_CLKNAND2_2 U339 ( .A1(a[3]), .A2(b[3]), .ZN(n325) );
  VHSR_AOI22_2 U340 ( .A1(a[2]), .A2(b[3]), .B1(a[3]), .B2(b[2]), .ZN(n308) );
  VHSR_IAO21_2 U341 ( .A1(n319), .A2(n325), .B(n308), .ZN(n332) );
  VHSR_AOI22_2 U342 ( .A1(a[3]), .A2(b[1]), .B1(a[1]), .B2(b[3]), .ZN(n317) );
  VHSR_CLKNAND2_2 U343 ( .A1(a[1]), .A2(b[1]), .ZN(n311) );
  VHSR_OAI22_2 U344 ( .A1(n319), .A2(n317), .B1(n325), .B2(n311), .ZN(n318) );
  VHSR_OAI22_2 U345 ( .A1(n310), .A2(n395), .B1(n394), .B2(n309), .ZN(n371) );
  VHSR_IN_2 U346 ( .I(n311), .ZN(n312) );
  VHSR_AOI21_2 U347 ( .A1(b[0]), .A2(a[2]), .B(n312), .ZN(n393) );
  VHSR_NOR3_2 U348 ( .A1(n393), .A2(n395), .A3(n394), .ZN(n396) );
  VHSR_OAI22_2 U349 ( .A1(n316), .A2(n315), .B1(n314), .B2(n313), .ZN(n370) );
  VHSR_AOI21_2 U350 ( .A1(n317), .A2(n319), .B(n318), .ZN(n336) );
  VHSR_CLKNAND2_2 U351 ( .A1(n337), .A2(n336), .ZN(n335) );
  VHSR_CLKNAND2_2 U352 ( .A1(n332), .A2(n331), .ZN(n323) );
  VHSR_AOI21_2 U353 ( .A1(n319), .A2(n323), .B(n325), .ZN(n364) );
  VHSR_AD1_1 U354 ( .A(n322), .B(n321), .CI(n320), .CO(n305), .S(n363) );
  VHSR_IN_2 U355 ( .I(n323), .ZN(n330) );
  VHSR_CLKNAND2_2 U356 ( .A1(n330), .A2(n325), .ZN(n324) );
  VHSR_OAI31_2 U357 ( .A1(n326), .A2(n330), .A3(n325), .B(n324), .ZN(n367) );
  VHSR_AD1_1 U358 ( .A(n329), .B(n328), .CI(n327), .CO(n320), .S(n366) );
  VHSR_IAO21_2 U359 ( .A1(n332), .A2(n331), .B(n330), .ZN(n369) );
  VHSR_AD1_1 U360 ( .A(n334), .B(n386), .CI(n333), .CO(n327), .S(n368) );
  VHSR_OAI21_2 U361 ( .A1(n337), .A2(n336), .B(n335), .ZN(n391) );
  VHSR_AOI211_2 U362 ( .A1(n388), .A2(n387), .B(n386), .C(n391), .ZN(n390) );
  VHSR_NOR2_1 U363 ( .A1(n339), .A2(n338), .ZN(n351) );
  VHSR_CLKNAND2_2 U364 ( .A1(b[6]), .A2(a[7]), .ZN(n341) );
  VHSR_AOI21_2 U365 ( .A1(a[6]), .A2(b[7]), .B(n341), .ZN(n340) );
  VHSR_AOI31_2 U366 ( .A1(a[6]), .A2(n341), .A3(b[7]), .B(n340), .ZN(n342) );
  VHSR_IN_2 U367 ( .I(n342), .ZN(n343) );
  VHSR_MAOI222_2 U368 ( .A(n346), .B(n344), .C(n343), .ZN(n353) );
  VHSR_OAI21_2 U369 ( .A1(n346), .A2(n345), .B(n353), .ZN(n350) );
  VHSR_CLKXOR2_2 U370 ( .A1(n351), .A2(n350), .Z(n347) );
  VHSR_CLKNAND2_2 U371 ( .A1(n348), .A2(n347), .ZN(n383) );
  VHSR_OAI21_2 U372 ( .A1(n348), .A2(n347), .B(n383), .ZN(n349) );
  VHSR_CLKNAND2_2 U373 ( .A1(a[7]), .A2(b[7]), .ZN(n382) );
  VHSR_NOR2_1 U374 ( .A1(n351), .A2(n350), .ZN(n352) );
  VHSR_AND3_2 U375 ( .A1(n384), .A2(n354), .A3(n383), .Z(n355) );
  VHSR_NOR2_1 U376 ( .A1(n382), .A2(n355), .ZN(product[15]) );
  VHSR_AD1_1 U377 ( .A(n374), .B(n373), .CI(n372), .CO(n375), .S(product[10])
         );
  VHSR_AD1_1 U378 ( .A(n377), .B(n376), .CI(n375), .CO(n378), .S(product[11])
         );
  VHSR_AD1_1 U379 ( .A(n380), .B(n379), .CI(n378), .CO(n348), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U380 ( .A1(n382), .A2(n381), .ZN(n385) );
  VHSR_XOR3_2 U381 ( .A1(n385), .A2(n384), .A3(n383), .Z(product[14]) );
  VHSR_AOI21_2 U382 ( .A1(n388), .A2(n387), .B(n386), .ZN(n389) );
  VHSR_IN_2 U383 ( .I(n389), .ZN(n392) );
  VHSR_AOI21_2 U384 ( .A1(n392), .A2(n391), .B(n390), .ZN(product[4]) );
  VHSR_OAI32_2 U385 ( .A1(n396), .A2(n395), .A3(n394), .B1(n393), .B2(n396), 
        .ZN(product[2]) );
endmodule


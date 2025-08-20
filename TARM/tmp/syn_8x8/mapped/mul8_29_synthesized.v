
module mul8_29 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n212, n213,
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
         n390, n391, n392, n393, n394, n395, n396, n397, n398, n399, n400,
         n401, n402;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U203 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_NOR2_1 U204 ( .A1(n247), .A2(n243), .ZN(n236) );
  VHSR_INAND2_2 U205 ( .A1(n321), .B1(n339), .ZN(n334) );
  VHSR_NOR2_1 U206 ( .A1(n290), .A2(n291), .ZN(n346) );
  VHSR_NOR2_1 U207 ( .A1(n395), .A2(n394), .ZN(n393) );
  VHSR_INOR2_2 U208 ( .A1(n360), .B1(n359), .ZN(n391) );
  VHSR_IN_2 U209 ( .I(n356), .ZN(product[13]) );
  VHSR_INOR3_1 U210 ( .A1(n236), .B1(n314), .B2(n278), .ZN(n296) );
  VHSR_NOR2_2 U211 ( .A1(n346), .A2(n345), .ZN(n358) );
  VHSR_NOR2_2 U212 ( .A1(n292), .A2(n288), .ZN(n290) );
  VHSR_INAND2_1 U213 ( .A1(n351), .B1(n349), .ZN(n352) );
  VHSR_AD1_1 U214 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(product[6])
         );
  VHSR_AD1_1 U215 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[9])
         );
  VHSR_AD1_1 U216 ( .A(n381), .B(n400), .CI(n380), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n379), .B(n378), .CI(n393), .CO(n375), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U219 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U220 ( .A(n365), .B(n364), .CI(n363), .CO(n382), .S(product[10])
         );
  VHSR_AOI22_2 U221 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n247) );
  VHSR_IN_2 U222 ( .I(b[3]), .ZN(n314) );
  VHSR_IN_2 U223 ( .I(b[2]), .ZN(n313) );
  VHSR_IN_2 U224 ( .I(a[5]), .ZN(n283) );
  VHSR_IN_2 U225 ( .I(a[4]), .ZN(n282) );
  VHSR_NOR4_2 U226 ( .A1(n314), .A2(n313), .A3(n283), .A4(n282), .ZN(n245) );
  VHSR_IN_2 U227 ( .I(a[7]), .ZN(n278) );
  VHSR_IN_2 U228 ( .I(b[1]), .ZN(n399) );
  VHSR_NOR2_1 U229 ( .A1(n278), .A2(n399), .ZN(n213) );
  VHSR_AOI211_2 U230 ( .A1(b[2]), .A2(a[4]), .B(n314), .C(n283), .ZN(n214) );
  VHSR_CLKNAND2_2 U231 ( .A1(a[6]), .A2(b[2]), .ZN(n216) );
  VHSR_IN_2 U232 ( .I(n216), .ZN(n212) );
  VHSR_MAOI222_2 U233 ( .A(n213), .B(n214), .C(n212), .ZN(n226) );
  VHSR_AOI21_2 U234 ( .A1(b[1]), .A2(a[7]), .B(n214), .ZN(n217) );
  VHSR_IN_2 U235 ( .I(n226), .ZN(n215) );
  VHSR_AOI21_2 U236 ( .A1(n217), .A2(n216), .B(n215), .ZN(n254) );
  VHSR_CLKNAND2_2 U237 ( .A1(a[6]), .A2(b[1]), .ZN(n223) );
  VHSR_IN_2 U238 ( .I(n223), .ZN(n220) );
  VHSR_IN_2 U239 ( .I(b[0]), .ZN(n397) );
  VHSR_NOR4_2 U240 ( .A1(n283), .A2(n282), .A3(n399), .A4(n397), .ZN(n272) );
  VHSR_CLKNAND2_2 U241 ( .A1(b[2]), .A2(a[5]), .ZN(n219) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[3]), .A2(a[4]), .ZN(n218) );
  VHSR_AOI21_2 U243 ( .A1(n219), .A2(n218), .B(n245), .ZN(n221) );
  VHSR_MAOI222_2 U244 ( .A(n220), .B(n272), .C(n221), .ZN(n225) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[2]), .A2(a[4]), .ZN(n268) );
  VHSR_OAI21_2 U246 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n267) );
  VHSR_OAI211_2 U247 ( .A1(n282), .A2(n397), .B(a[5]), .C(b[1]), .ZN(n266) );
  VHSR_MAOI222_2 U248 ( .A(n268), .B(n267), .C(n266), .ZN(n265) );
  VHSR_NOR2_1 U249 ( .A1(n272), .A2(n221), .ZN(n224) );
  VHSR_IN_2 U250 ( .I(n225), .ZN(n222) );
  VHSR_AOI21_2 U251 ( .A1(n224), .A2(n223), .B(n222), .ZN(n257) );
  VHSR_CLKNAND2_2 U252 ( .A1(n265), .A2(n257), .ZN(n256) );
  VHSR_CLKNAND2_2 U253 ( .A1(n225), .A2(n256), .ZN(n253) );
  VHSR_CLKNAND2_2 U254 ( .A1(n254), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U255 ( .A1(n226), .A2(n252), .ZN(n244) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n318) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n281) );
  VHSR_IN_2 U259 ( .I(a[2]), .ZN(n319) );
  VHSR_OAI22_2 U260 ( .A1(n281), .A2(n318), .B1(n280), .B2(n319), .ZN(n242) );
  VHSR_AOI22_2 U261 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n233) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[4]), .A2(a[2]), .ZN(n264) );
  VHSR_NAND3_2 U263 ( .A1(a[3]), .A2(b[5]), .A3(n264), .ZN(n232) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[7]), .A2(a[2]), .ZN(n227) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[1]), .ZN(n229) );
  VHSR_OAI22_2 U266 ( .A1(n233), .A2(n232), .B1(n227), .B2(n229), .ZN(n234) );
  VHSR_IN_2 U267 ( .I(b[4]), .ZN(n342) );
  VHSR_IN_2 U268 ( .I(a[0]), .ZN(n398) );
  VHSR_OAI211_2 U269 ( .A1(n342), .A2(n398), .B(b[5]), .C(a[1]), .ZN(n263) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[0]), .ZN(n262) );
  VHSR_MAOI222_2 U271 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_NAND4_2 U272 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_IN_2 U273 ( .I(b[5]), .ZN(n277) );
  VHSR_OAI22_2 U274 ( .A1(n342), .A2(n318), .B1(n277), .B2(n319), .ZN(n228) );
  VHSR_AND2_2 U275 ( .A1(n239), .A2(n228), .Z(n231) );
  VHSR_OAI21_2 U276 ( .A1(n280), .A2(n398), .B(n229), .ZN(n230) );
  VHSR_IN_2 U277 ( .I(a[1]), .ZN(n396) );
  VHSR_NOR4_2 U278 ( .A1(n342), .A2(n277), .A3(n396), .A4(n398), .ZN(n270) );
  VHSR_AND2_2 U279 ( .A1(n261), .A2(n260), .Z(n259) );
  VHSR_AD1_1 U280 ( .A(n231), .B(n230), .CI(n270), .CO(n248), .S(n260) );
  VHSR_AOI21_2 U281 ( .A1(n233), .A2(n232), .B(n234), .ZN(n251) );
  VHSR_OAI32_2 U282 ( .A1(n234), .A2(n259), .A3(n248), .B1(n251), .B2(n234), 
        .ZN(n240) );
  VHSR_CLKNAND2_2 U283 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U284 ( .A1(n242), .A2(n238), .ZN(n237) );
  VHSR_NOR3_2 U285 ( .A1(n280), .A2(n318), .A3(n237), .ZN(n295) );
  VHSR_NOR2_1 U286 ( .A1(n314), .A2(n278), .ZN(n235) );
  VHSR_IAO21_2 U287 ( .A1(n236), .A2(n235), .B(n296), .ZN(n299) );
  VHSR_OAI32_2 U288 ( .A1(n295), .A2(n318), .A3(n280), .B1(n237), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U289 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U290 ( .A1(n242), .A2(n241), .ZN(n306) );
  VHSR_AOI21_2 U291 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U292 ( .A1(n247), .A2(n246), .ZN(n305) );
  VHSR_NOR2_1 U293 ( .A1(n259), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U294 ( .A1(n259), .A2(n248), .B1(n251), .B2(n250), .ZN(n249) );
  VHSR_OAI21_2 U295 ( .A1(n251), .A2(n250), .B(n249), .ZN(n311) );
  VHSR_OAI21_2 U296 ( .A1(n254), .A2(n253), .B(n252), .ZN(n255) );
  VHSR_IN_2 U297 ( .I(n255), .ZN(n310) );
  VHSR_OAI21_2 U298 ( .A1(n265), .A2(n257), .B(n256), .ZN(n258) );
  VHSR_IN_2 U299 ( .I(n258), .ZN(n325) );
  VHSR_IAO21_2 U300 ( .A1(n261), .A2(n260), .B(n259), .ZN(n324) );
  VHSR_AOI31_2 U301 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n332) );
  VHSR_AOI31_2 U302 ( .A1(n268), .A2(n267), .A3(n266), .B(n265), .ZN(n331) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[5]), .A2(a[0]), .ZN(n269) );
  VHSR_OAI32_2 U304 ( .A1(n270), .A2(n396), .A3(n342), .B1(n269), .B2(n270), 
        .ZN(n338) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[4]), .ZN(n285) );
  VHSR_IN_2 U306 ( .I(n285), .ZN(n370) );
  VHSR_NOR2_1 U307 ( .A1(n397), .A2(n398), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U308 ( .A1(n370), .A2(product[0]), .ZN(n344) );
  VHSR_IN_2 U309 ( .I(n344), .ZN(n337) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[4]), .A2(b[1]), .ZN(n271) );
  VHSR_OAI32_2 U311 ( .A1(n272), .A2(n397), .A3(n283), .B1(n271), .B2(n272), 
        .ZN(n336) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[6]), .ZN(n361) );
  VHSR_IN_2 U313 ( .I(n361), .ZN(n388) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_NAND3_2 U315 ( .A1(a[7]), .A2(b[5]), .A3(n303), .ZN(n274) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[4]), .A2(b[6]), .ZN(n302) );
  VHSR_NAND3_2 U317 ( .A1(b[7]), .A2(a[5]), .A3(n302), .ZN(n273) );
  VHSR_CLKNAND2_2 U318 ( .A1(n274), .A2(n273), .ZN(n276) );
  VHSR_MAOI222_2 U319 ( .A(n361), .B(n274), .C(n273), .ZN(n345) );
  VHSR_IN_2 U320 ( .I(n345), .ZN(n275) );
  VHSR_OAI21_2 U321 ( .A1(n388), .A2(n276), .B(n275), .ZN(n291) );
  VHSR_NOR3_2 U322 ( .A1(n283), .A2(n277), .A3(n285), .ZN(n307) );
  VHSR_NOR3_2 U323 ( .A1(n278), .A2(n303), .A3(n277), .ZN(n353) );
  VHSR_AOI22_2 U324 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n279) );
  VHSR_NOR2_1 U325 ( .A1(n353), .A2(n279), .ZN(n287) );
  VHSR_NOR4_2 U326 ( .A1(n283), .A2(n282), .A3(n281), .A4(n280), .ZN(n351) );
  VHSR_AOI22_2 U327 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n284) );
  VHSR_NOR2_1 U328 ( .A1(n351), .A2(n284), .ZN(n286) );
  VHSR_NAND3_2 U329 ( .A1(b[5]), .A2(a[5]), .A3(n285), .ZN(n301) );
  VHSR_MAOI222_2 U330 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_AND2_2 U331 ( .A1(n293), .A2(n300), .Z(n292) );
  VHSR_AD1_1 U332 ( .A(n307), .B(n287), .CI(n286), .CO(n288), .S(n293) );
  VHSR_CLKNAND2_2 U333 ( .A1(n292), .A2(n288), .ZN(n289) );
  VHSR_AOI22_2 U334 ( .A1(n291), .A2(n290), .B1(n289), .B2(n346), .ZN(n386) );
  VHSR_IAO21_2 U335 ( .A1(n293), .A2(n300), .B(n292), .ZN(n384) );
  VHSR_AD1_1 U336 ( .A(n296), .B(n295), .CI(n294), .CO(n387), .S(n383) );
  VHSR_AD1_1 U337 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n365) );
  VHSR_AOI31_2 U338 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n364) );
  VHSR_AD1_1 U339 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n368) );
  VHSR_AOI22_2 U340 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n308) );
  VHSR_NOR2_1 U341 ( .A1(n308), .A2(n307), .ZN(n367) );
  VHSR_AD1_1 U342 ( .A(n311), .B(n310), .CI(n309), .CO(n304), .S(n371) );
  VHSR_NOR2_1 U343 ( .A1(n313), .A2(n319), .ZN(n329) );
  VHSR_IN_2 U344 ( .I(n329), .ZN(n322) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[3]), .A2(a[3]), .ZN(n328) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n312) );
  VHSR_IAO21_2 U347 ( .A1(n322), .A2(n328), .B(n312), .ZN(n335) );
  VHSR_AOI22_2 U348 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n320) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[1]), .A2(a[1]), .ZN(n401) );
  VHSR_OAI22_2 U350 ( .A1(n322), .A2(n320), .B1(n328), .B2(n401), .ZN(n321) );
  VHSR_OAI22_2 U351 ( .A1(n314), .A2(n398), .B1(n313), .B2(n396), .ZN(n381) );
  VHSR_IN_2 U352 ( .I(n401), .ZN(n317) );
  VHSR_NOR2_1 U353 ( .A1(n397), .A2(n319), .ZN(n316) );
  VHSR_OAI211_2 U354 ( .A1(n316), .A2(n317), .B(b[2]), .C(a[0]), .ZN(n315) );
  VHSR_IOA21_2 U355 ( .A1(n317), .A2(n316), .B(n315), .ZN(n400) );
  VHSR_OAI22_2 U356 ( .A1(n399), .A2(n319), .B1(n397), .B2(n318), .ZN(n380) );
  VHSR_AOI21_2 U357 ( .A1(n320), .A2(n322), .B(n321), .ZN(n340) );
  VHSR_CLKNAND2_2 U358 ( .A1(n341), .A2(n340), .ZN(n339) );
  VHSR_CLKNAND2_2 U359 ( .A1(n335), .A2(n334), .ZN(n326) );
  VHSR_AOI21_2 U360 ( .A1(n322), .A2(n326), .B(n328), .ZN(n374) );
  VHSR_AD1_1 U361 ( .A(n325), .B(n324), .CI(n323), .CO(n309), .S(n373) );
  VHSR_IN_2 U362 ( .I(n326), .ZN(n333) );
  VHSR_CLKNAND2_2 U363 ( .A1(n333), .A2(n328), .ZN(n327) );
  VHSR_OAI31_2 U364 ( .A1(n329), .A2(n333), .A3(n328), .B(n327), .ZN(n377) );
  VHSR_AD1_1 U365 ( .A(n332), .B(n331), .CI(n330), .CO(n323), .S(n376) );
  VHSR_IAO21_2 U366 ( .A1(n335), .A2(n334), .B(n333), .ZN(n379) );
  VHSR_AD1_1 U367 ( .A(n338), .B(n337), .CI(n336), .CO(n330), .S(n378) );
  VHSR_OAI21_2 U368 ( .A1(n341), .A2(n340), .B(n339), .ZN(n395) );
  VHSR_NOR2_1 U369 ( .A1(n342), .A2(n398), .ZN(n343) );
  VHSR_AOI32_2 U370 ( .A1(b[0]), .A2(n344), .A3(a[4]), .B1(n343), .B2(n344), 
        .ZN(n394) );
  VHSR_CLKNAND2_2 U371 ( .A1(a[7]), .A2(b[6]), .ZN(n348) );
  VHSR_AOI21_2 U372 ( .A1(a[6]), .A2(b[7]), .B(n348), .ZN(n347) );
  VHSR_AOI31_2 U373 ( .A1(a[6]), .A2(n348), .A3(b[7]), .B(n347), .ZN(n349) );
  VHSR_IN_2 U374 ( .I(n349), .ZN(n350) );
  VHSR_MAOI222_2 U375 ( .A(n353), .B(n351), .C(n350), .ZN(n360) );
  VHSR_OAI21_2 U376 ( .A1(n353), .A2(n352), .B(n360), .ZN(n357) );
  VHSR_CLKXOR2_2 U377 ( .A1(n358), .A2(n357), .Z(n354) );
  VHSR_CLKNAND2_2 U378 ( .A1(n355), .A2(n354), .ZN(n390) );
  VHSR_OAI21_2 U379 ( .A1(n355), .A2(n354), .B(n390), .ZN(n356) );
  VHSR_CLKNAND2_2 U380 ( .A1(a[7]), .A2(b[7]), .ZN(n389) );
  VHSR_NOR2_1 U381 ( .A1(n358), .A2(n357), .ZN(n359) );
  VHSR_AND3_2 U382 ( .A1(n391), .A2(n361), .A3(n390), .Z(n362) );
  VHSR_NOR2_1 U383 ( .A1(n389), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U384 ( .A(n384), .B(n383), .CI(n382), .CO(n385), .S(product[11])
         );
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


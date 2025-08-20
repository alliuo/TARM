
module mul8_100 ( a, b, product );
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
         n389, n390, n391, n392, n393, n394, n395, n396, n397, n398, n399;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U202 ( .A1(n232), .B1(n251), .ZN(n238) );
  VHSR_NOR2_1 U203 ( .A1(n245), .A2(n241), .ZN(n234) );
  VHSR_NOR2_1 U204 ( .A1(n397), .A2(n318), .ZN(n329) );
  VHSR_INOR3_2 U205 ( .A1(n234), .B1(n317), .B2(n277), .ZN(n295) );
  VHSR_NOR2_1 U206 ( .A1(n392), .A2(n391), .ZN(n390) );
  VHSR_INOR2_2 U207 ( .A1(n357), .B1(n356), .ZN(n388) );
  VHSR_IN_2 U208 ( .I(n353), .ZN(product[13]) );
  VHSR_NOR2_2 U209 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_NOR2_2 U210 ( .A1(n343), .A2(n342), .ZN(n355) );
  VHSR_INAND2_1 U211 ( .A1(n348), .B1(n346), .ZN(n349) );
  VHSR_MOAI22_1 U212 ( .A1(n277), .A2(n395), .B1(a[6]), .B2(b[2]), .ZN(n214)
         );
  VHSR_AD1_1 U213 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(product[9])
         );
  VHSR_AD1_1 U214 ( .A(n375), .B(n399), .CI(n374), .CO(n332), .S(product[3])
         );
  VHSR_AD1_1 U215 ( .A(n390), .B(n373), .CI(n372), .CO(n376), .S(product[5])
         );
  VHSR_AD1_1 U216 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U217 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U218 ( .A(n362), .B(n361), .CI(n360), .CO(n379), .S(product[10])
         );
  VHSR_AOI22_2 U219 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n245) );
  VHSR_IN_2 U220 ( .I(b[3]), .ZN(n317) );
  VHSR_IN_2 U221 ( .I(b[2]), .ZN(n397) );
  VHSR_IN_2 U222 ( .I(a[5]), .ZN(n282) );
  VHSR_IN_2 U223 ( .I(a[4]), .ZN(n281) );
  VHSR_NOR4_2 U224 ( .A1(n317), .A2(n397), .A3(n282), .A4(n281), .ZN(n243) );
  VHSR_IN_2 U225 ( .I(a[7]), .ZN(n277) );
  VHSR_IN_2 U226 ( .I(b[1]), .ZN(n395) );
  VHSR_NOR2_1 U227 ( .A1(n277), .A2(n395), .ZN(n212) );
  VHSR_AND2_2 U228 ( .A1(a[6]), .A2(b[2]), .Z(n211) );
  VHSR_AOI211_2 U229 ( .A1(b[2]), .A2(a[4]), .B(n317), .C(n282), .ZN(n213) );
  VHSR_MAOI222_2 U230 ( .A(n212), .B(n211), .C(n213), .ZN(n224) );
  VHSR_OAI21_2 U231 ( .A1(n214), .A2(n213), .B(n224), .ZN(n215) );
  VHSR_IN_2 U232 ( .I(n215), .ZN(n248) );
  VHSR_IN_2 U233 ( .I(b[0]), .ZN(n394) );
  VHSR_NOR4_2 U234 ( .A1(n282), .A2(n281), .A3(n395), .A4(n394), .ZN(n271) );
  VHSR_CLKNAND2_2 U235 ( .A1(b[2]), .A2(a[5]), .ZN(n217) );
  VHSR_CLKNAND2_2 U236 ( .A1(b[3]), .A2(a[4]), .ZN(n216) );
  VHSR_AOI21_2 U237 ( .A1(n217), .A2(n216), .B(n243), .ZN(n219) );
  VHSR_AOI22_2 U238 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n221) );
  VHSR_IN_2 U239 ( .I(n221), .ZN(n218) );
  VHSR_MAOI222_2 U240 ( .A(n271), .B(n219), .C(n218), .ZN(n223) );
  VHSR_CLKNAND2_2 U241 ( .A1(b[2]), .A2(a[4]), .ZN(n267) );
  VHSR_OAI211_2 U242 ( .A1(n281), .A2(n394), .B(a[5]), .C(b[1]), .ZN(n266) );
  VHSR_CLKNAND2_2 U243 ( .A1(a[6]), .A2(b[0]), .ZN(n265) );
  VHSR_MAOI222_2 U244 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NOR2_1 U245 ( .A1(n271), .A2(n219), .ZN(n222) );
  VHSR_IN_2 U246 ( .I(n223), .ZN(n220) );
  VHSR_AOI21_2 U247 ( .A1(n222), .A2(n221), .B(n220), .ZN(n258) );
  VHSR_CLKNAND2_2 U248 ( .A1(n264), .A2(n258), .ZN(n257) );
  VHSR_CLKNAND2_2 U249 ( .A1(n223), .A2(n257), .ZN(n247) );
  VHSR_CLKNAND2_2 U250 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_CLKNAND2_2 U251 ( .A1(n224), .A2(n246), .ZN(n242) );
  VHSR_IN_2 U252 ( .I(b[7]), .ZN(n279) );
  VHSR_IN_2 U253 ( .I(a[3]), .ZN(n320) );
  VHSR_IN_2 U254 ( .I(b[6]), .ZN(n280) );
  VHSR_IN_2 U255 ( .I(a[2]), .ZN(n318) );
  VHSR_OAI22_2 U256 ( .A1(n280), .A2(n320), .B1(n279), .B2(n318), .ZN(n240) );
  VHSR_NOR2_1 U257 ( .A1(n279), .A2(n318), .ZN(n226) );
  VHSR_IN_2 U258 ( .I(a[1]), .ZN(n393) );
  VHSR_NOR2_1 U259 ( .A1(n280), .A2(n393), .ZN(n225) );
  VHSR_IN_2 U260 ( .I(b[5]), .ZN(n276) );
  VHSR_AOI211_2 U261 ( .A1(b[4]), .A2(a[2]), .B(n276), .C(n320), .ZN(n231) );
  VHSR_OAI22_2 U262 ( .A1(n280), .A2(n318), .B1(n279), .B2(n393), .ZN(n230) );
  VHSR_AOI22_2 U263 ( .A1(n226), .A2(n225), .B1(n231), .B2(n230), .ZN(n232) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_IN_2 U265 ( .I(b[4]), .ZN(n333) );
  VHSR_IN_2 U266 ( .I(a[0]), .ZN(n398) );
  VHSR_OAI211_2 U267 ( .A1(n333), .A2(n398), .B(b[5]), .C(a[1]), .ZN(n262) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[6]), .A2(a[0]), .ZN(n261) );
  VHSR_MAOI222_2 U269 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NOR4_2 U270 ( .A1(n333), .A2(n276), .A3(n393), .A4(n398), .ZN(n269) );
  VHSR_NAND4_2 U271 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_OAI22_2 U272 ( .A1(n333), .A2(n320), .B1(n276), .B2(n318), .ZN(n227) );
  VHSR_AND2_2 U273 ( .A1(n237), .A2(n227), .Z(n229) );
  VHSR_OAI22_2 U274 ( .A1(n280), .A2(n393), .B1(n279), .B2(n398), .ZN(n228) );
  VHSR_AND2_2 U275 ( .A1(n260), .A2(n256), .Z(n255) );
  VHSR_AD1_1 U276 ( .A(n269), .B(n229), .CI(n228), .CO(n250), .S(n256) );
  VHSR_NOR2_1 U277 ( .A1(n255), .A2(n250), .ZN(n253) );
  VHSR_OAI21_2 U278 ( .A1(n231), .A2(n230), .B(n232), .ZN(n254) );
  VHSR_NOR2_1 U279 ( .A1(n253), .A2(n254), .ZN(n251) );
  VHSR_CLKNAND2_2 U280 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U281 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U282 ( .A1(n279), .A2(n320), .A3(n235), .ZN(n294) );
  VHSR_NOR2_1 U283 ( .A1(n317), .A2(n277), .ZN(n233) );
  VHSR_IAO21_2 U284 ( .A1(n234), .A2(n233), .B(n295), .ZN(n298) );
  VHSR_OAI32_2 U285 ( .A1(n294), .A2(n320), .A3(n279), .B1(n235), .B2(n294), 
        .ZN(n297) );
  VHSR_OAI21_2 U286 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U287 ( .A1(n240), .A2(n239), .ZN(n305) );
  VHSR_AOI21_2 U288 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U289 ( .A1(n245), .A2(n244), .ZN(n304) );
  VHSR_OAI21_2 U290 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_IN_2 U291 ( .I(n249), .ZN(n310) );
  VHSR_CLKNAND2_2 U292 ( .A1(n255), .A2(n250), .ZN(n252) );
  VHSR_AOI22_2 U293 ( .A1(n254), .A2(n253), .B1(n252), .B2(n251), .ZN(n309) );
  VHSR_IAO21_2 U294 ( .A1(n260), .A2(n256), .B(n255), .ZN(n313) );
  VHSR_OAI21_2 U295 ( .A1(n264), .A2(n258), .B(n257), .ZN(n259) );
  VHSR_IN_2 U296 ( .I(n259), .ZN(n312) );
  VHSR_AOI31_2 U297 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n326) );
  VHSR_AOI31_2 U298 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n325) );
  VHSR_CLKNAND2_2 U299 ( .A1(b[5]), .A2(a[0]), .ZN(n268) );
  VHSR_OAI32_2 U300 ( .A1(n269), .A2(n393), .A3(n333), .B1(n268), .B2(n269), 
        .ZN(n341) );
  VHSR_CLKNAND2_2 U301 ( .A1(a[4]), .A2(b[1]), .ZN(n270) );
  VHSR_OAI32_2 U302 ( .A1(n271), .A2(n394), .A3(n282), .B1(n270), .B2(n271), 
        .ZN(n340) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[4]), .ZN(n284) );
  VHSR_IN_2 U304 ( .I(n284), .ZN(n367) );
  VHSR_NOR2_1 U305 ( .A1(n394), .A2(n398), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U306 ( .A1(n367), .A2(product[0]), .ZN(n335) );
  VHSR_IN_2 U307 ( .I(n335), .ZN(n339) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[6]), .A2(b[6]), .ZN(n358) );
  VHSR_IN_2 U309 ( .I(n358), .ZN(n385) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[4]), .ZN(n302) );
  VHSR_NAND3_2 U311 ( .A1(a[7]), .A2(b[5]), .A3(n302), .ZN(n273) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[4]), .A2(b[6]), .ZN(n301) );
  VHSR_NAND3_2 U313 ( .A1(b[7]), .A2(a[5]), .A3(n301), .ZN(n272) );
  VHSR_CLKNAND2_2 U314 ( .A1(n273), .A2(n272), .ZN(n275) );
  VHSR_MAOI222_2 U315 ( .A(n358), .B(n273), .C(n272), .ZN(n342) );
  VHSR_IN_2 U316 ( .I(n342), .ZN(n274) );
  VHSR_OAI21_2 U317 ( .A1(n385), .A2(n275), .B(n274), .ZN(n290) );
  VHSR_NOR3_2 U318 ( .A1(n282), .A2(n276), .A3(n284), .ZN(n306) );
  VHSR_NOR3_2 U319 ( .A1(n277), .A2(n302), .A3(n276), .ZN(n350) );
  VHSR_AOI22_2 U320 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n278) );
  VHSR_NOR2_1 U321 ( .A1(n350), .A2(n278), .ZN(n286) );
  VHSR_NOR4_2 U322 ( .A1(n282), .A2(n281), .A3(n280), .A4(n279), .ZN(n348) );
  VHSR_AOI22_2 U323 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n283) );
  VHSR_NOR2_1 U324 ( .A1(n348), .A2(n283), .ZN(n285) );
  VHSR_NAND3_2 U325 ( .A1(b[5]), .A2(a[5]), .A3(n284), .ZN(n300) );
  VHSR_MAOI222_2 U326 ( .A(n302), .B(n301), .C(n300), .ZN(n299) );
  VHSR_AND2_2 U327 ( .A1(n292), .A2(n299), .Z(n291) );
  VHSR_AD1_1 U328 ( .A(n306), .B(n286), .CI(n285), .CO(n287), .S(n292) );
  VHSR_NOR2_1 U329 ( .A1(n291), .A2(n287), .ZN(n289) );
  VHSR_CLKNAND2_2 U330 ( .A1(n291), .A2(n287), .ZN(n288) );
  VHSR_NOR2_1 U331 ( .A1(n289), .A2(n290), .ZN(n343) );
  VHSR_AOI22_2 U332 ( .A1(n290), .A2(n289), .B1(n288), .B2(n343), .ZN(n383) );
  VHSR_IAO21_2 U333 ( .A1(n292), .A2(n299), .B(n291), .ZN(n381) );
  VHSR_AD1_1 U334 ( .A(n295), .B(n294), .CI(n293), .CO(n384), .S(n380) );
  VHSR_AD1_1 U335 ( .A(n298), .B(n297), .CI(n296), .CO(n293), .S(n362) );
  VHSR_AOI31_2 U336 ( .A1(n302), .A2(n301), .A3(n300), .B(n299), .ZN(n361) );
  VHSR_AD1_1 U337 ( .A(n305), .B(n304), .CI(n303), .CO(n296), .S(n365) );
  VHSR_AOI22_2 U338 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n307) );
  VHSR_NOR2_1 U339 ( .A1(n307), .A2(n306), .ZN(n364) );
  VHSR_AD1_1 U340 ( .A(n310), .B(n309), .CI(n308), .CO(n303), .S(n368) );
  VHSR_AD1_1 U341 ( .A(n313), .B(n312), .CI(n311), .CO(n308), .S(n371) );
  VHSR_AOI22_2 U342 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n314) );
  VHSR_AOI31_2 U343 ( .A1(a[3]), .A2(b[3]), .A3(n329), .B(n314), .ZN(n338) );
  VHSR_NOR2_1 U344 ( .A1(n317), .A2(n393), .ZN(n316) );
  VHSR_NOR2_1 U345 ( .A1(n395), .A2(n320), .ZN(n315) );
  VHSR_MAOI222_2 U346 ( .A(n329), .B(n316), .C(n315), .ZN(n322) );
  VHSR_OAI22_2 U347 ( .A1(n317), .A2(n398), .B1(n397), .B2(n393), .ZN(n375) );
  VHSR_AOI22_2 U348 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n396) );
  VHSR_NOR3_2 U349 ( .A1(n396), .A2(n398), .A3(n397), .ZN(n399) );
  VHSR_OAI22_2 U350 ( .A1(n395), .A2(n318), .B1(n394), .B2(n320), .ZN(n374) );
  VHSR_IN_2 U351 ( .I(n322), .ZN(n321) );
  VHSR_AOI21_2 U352 ( .A1(a[1]), .A2(b[3]), .B(n329), .ZN(n319) );
  VHSR_OAI32_2 U353 ( .A1(n321), .A2(n320), .A3(n395), .B1(n319), .B2(n321), 
        .ZN(n331) );
  VHSR_CLKNAND2_2 U354 ( .A1(n332), .A2(n331), .ZN(n330) );
  VHSR_CLKNAND2_2 U355 ( .A1(n322), .A2(n330), .ZN(n337) );
  VHSR_AND2_2 U356 ( .A1(n338), .A2(n337), .Z(n336) );
  VHSR_OAI211_2 U357 ( .A1(n329), .A2(n336), .B(a[3]), .C(b[3]), .ZN(n323) );
  VHSR_IN_2 U358 ( .I(n323), .ZN(n370) );
  VHSR_AD1_1 U359 ( .A(n326), .B(n325), .CI(n324), .CO(n311), .S(n378) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[3]), .A2(a[3]), .ZN(n328) );
  VHSR_CLKNAND2_2 U361 ( .A1(n336), .A2(n328), .ZN(n327) );
  VHSR_OAI31_2 U362 ( .A1(n329), .A2(n336), .A3(n328), .B(n327), .ZN(n377) );
  VHSR_OAI21_2 U363 ( .A1(n332), .A2(n331), .B(n330), .ZN(n392) );
  VHSR_NOR2_1 U364 ( .A1(n333), .A2(n398), .ZN(n334) );
  VHSR_AOI32_2 U365 ( .A1(b[0]), .A2(n335), .A3(a[4]), .B1(n334), .B2(n335), 
        .ZN(n391) );
  VHSR_IAO21_2 U366 ( .A1(n338), .A2(n337), .B(n336), .ZN(n373) );
  VHSR_AD1_1 U367 ( .A(n341), .B(n340), .CI(n339), .CO(n324), .S(n372) );
  VHSR_CLKNAND2_2 U368 ( .A1(a[7]), .A2(b[6]), .ZN(n345) );
  VHSR_AOI21_2 U369 ( .A1(a[6]), .A2(b[7]), .B(n345), .ZN(n344) );
  VHSR_AOI31_2 U370 ( .A1(a[6]), .A2(n345), .A3(b[7]), .B(n344), .ZN(n346) );
  VHSR_IN_2 U371 ( .I(n346), .ZN(n347) );
  VHSR_MAOI222_2 U372 ( .A(n350), .B(n348), .C(n347), .ZN(n357) );
  VHSR_OAI21_2 U373 ( .A1(n350), .A2(n349), .B(n357), .ZN(n354) );
  VHSR_CLKXOR2_2 U374 ( .A1(n355), .A2(n354), .Z(n351) );
  VHSR_CLKNAND2_2 U375 ( .A1(n352), .A2(n351), .ZN(n387) );
  VHSR_OAI21_2 U376 ( .A1(n352), .A2(n351), .B(n387), .ZN(n353) );
  VHSR_CLKNAND2_2 U377 ( .A1(a[7]), .A2(b[7]), .ZN(n386) );
  VHSR_NOR2_1 U378 ( .A1(n355), .A2(n354), .ZN(n356) );
  VHSR_AND3_2 U379 ( .A1(n388), .A2(n358), .A3(n387), .Z(n359) );
  VHSR_NOR2_1 U380 ( .A1(n386), .A2(n359), .ZN(product[15]) );
  VHSR_AD1_1 U381 ( .A(n378), .B(n377), .CI(n376), .CO(n369), .S(product[6])
         );
  VHSR_AD1_1 U382 ( .A(n381), .B(n380), .CI(n379), .CO(n382), .S(product[11])
         );
  VHSR_AD1_1 U383 ( .A(n384), .B(n383), .CI(n382), .CO(n352), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U384 ( .A1(n386), .A2(n385), .ZN(n389) );
  VHSR_XOR3_2 U385 ( .A1(n389), .A2(n388), .A3(n387), .Z(product[14]) );
  VHSR_AOI21_2 U386 ( .A1(n392), .A2(n391), .B(n390), .ZN(product[4]) );
  VHSR_OAI22_2 U387 ( .A1(n395), .A2(n398), .B1(n394), .B2(n393), .ZN(
        product[1]) );
  VHSR_OAI32_2 U388 ( .A1(n399), .A2(n398), .A3(n397), .B1(n396), .B2(n399), 
        .ZN(product[2]) );
endmodule


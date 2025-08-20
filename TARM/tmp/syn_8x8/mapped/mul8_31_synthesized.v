
module mul8_31 ( a, b, product );
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
         n389, n390, n391, n392, n393, n394, n395, n396, n397, n398, n399,
         n400, n401, n402, n403;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U203 ( .A1(n233), .B1(n252), .ZN(n239) );
  VHSR_NOR2_1 U204 ( .A1(n346), .A2(n345), .ZN(n358) );
  VHSR_INAND2_2 U205 ( .A1(n326), .B1(n337), .ZN(n341) );
  VHSR_NOR2_1 U206 ( .A1(n395), .A2(n394), .ZN(n393) );
  VHSR_INAND3_2 U207 ( .A1(n373), .B1(b[5]), .B2(a[5]), .ZN(n304) );
  VHSR_NOR2_1 U208 ( .A1(n273), .A2(n283), .ZN(n388) );
  VHSR_NOR2_1 U209 ( .A1(n284), .A2(n278), .ZN(n373) );
  VHSR_IN_2 U210 ( .I(n356), .ZN(product[13]) );
  VHSR_INOR2_1 U211 ( .A1(n360), .B1(n359), .ZN(n391) );
  VHSR_NOR2_2 U212 ( .A1(n296), .A2(n295), .ZN(n294) );
  VHSR_AD1_1 U213 ( .A(n380), .B(n379), .CI(n378), .CO(n375), .S(product[6])
         );
  VHSR_AD1_1 U214 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[10])
         );
  VHSR_AD1_1 U216 ( .A(n384), .B(n403), .CI(n383), .CO(n339), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n397), .B(n382), .CI(n381), .CO(n378), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U219 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(product[9])
         );
  VHSR_AD1_1 U220 ( .A(n365), .B(n364), .CI(n363), .CO(n385), .S(product[11])
         );
  VHSR_IN_2 U221 ( .I(b[1]), .ZN(n323) );
  VHSR_IN_2 U222 ( .I(a[0]), .ZN(n402) );
  VHSR_NOR2_1 U223 ( .A1(n323), .A2(n402), .ZN(product[0]) );
  VHSR_IN_2 U224 ( .I(b[0]), .ZN(n321) );
  VHSR_IN_2 U225 ( .I(a[1]), .ZN(n320) );
  VHSR_NOR2_1 U226 ( .A1(n321), .A2(n320), .ZN(product[1]) );
  VHSR_AOI22_2 U227 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n246) );
  VHSR_IN_2 U228 ( .I(b[3]), .ZN(n327) );
  VHSR_IN_2 U229 ( .I(b[2]), .ZN(n401) );
  VHSR_IN_2 U230 ( .I(a[5]), .ZN(n285) );
  VHSR_IN_2 U231 ( .I(a[4]), .ZN(n284) );
  VHSR_NOR4_2 U232 ( .A1(n327), .A2(n401), .A3(n285), .A4(n284), .ZN(n244) );
  VHSR_IN_2 U233 ( .I(a[7]), .ZN(n280) );
  VHSR_CLKNAND2_2 U234 ( .A1(a[6]), .A2(b[1]), .ZN(n222) );
  VHSR_NOR2_1 U235 ( .A1(n280), .A2(n222), .ZN(n212) );
  VHSR_OAI211_2 U236 ( .A1(n401), .A2(n284), .B(b[3]), .C(a[5]), .ZN(n213) );
  VHSR_IN_2 U237 ( .I(n213), .ZN(n211) );
  VHSR_IN_2 U238 ( .I(a[6]), .ZN(n273) );
  VHSR_NOR2_1 U239 ( .A1(n273), .A2(n401), .ZN(n214) );
  VHSR_MAOI222_2 U240 ( .A(n212), .B(n211), .C(n214), .ZN(n225) );
  VHSR_OAI31_2 U241 ( .A1(n280), .A2(n273), .A3(n323), .B(n213), .ZN(n215) );
  VHSR_OAI21_2 U242 ( .A1(n215), .A2(n214), .B(n225), .ZN(n216) );
  VHSR_IN_2 U243 ( .I(n216), .ZN(n249) );
  VHSR_IN_2 U244 ( .I(n222), .ZN(n219) );
  VHSR_NOR4_2 U245 ( .A1(n285), .A2(n284), .A3(n323), .A4(n321), .ZN(n272) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[2]), .A2(a[5]), .ZN(n218) );
  VHSR_CLKNAND2_2 U247 ( .A1(b[3]), .A2(a[4]), .ZN(n217) );
  VHSR_AOI21_2 U248 ( .A1(n218), .A2(n217), .B(n244), .ZN(n220) );
  VHSR_MAOI222_2 U249 ( .A(n219), .B(n272), .C(n220), .ZN(n224) );
  VHSR_CLKNAND2_2 U250 ( .A1(b[2]), .A2(a[4]), .ZN(n268) );
  VHSR_CLKNAND2_2 U251 ( .A1(a[4]), .A2(b[0]), .ZN(n395) );
  VHSR_NAND3_2 U252 ( .A1(b[1]), .A2(a[5]), .A3(n395), .ZN(n267) );
  VHSR_CLKNAND2_2 U253 ( .A1(a[6]), .A2(b[0]), .ZN(n266) );
  VHSR_MAOI222_2 U254 ( .A(n268), .B(n267), .C(n266), .ZN(n265) );
  VHSR_NOR2_1 U255 ( .A1(n272), .A2(n220), .ZN(n223) );
  VHSR_IN_2 U256 ( .I(n224), .ZN(n221) );
  VHSR_AOI21_2 U257 ( .A1(n223), .A2(n222), .B(n221), .ZN(n259) );
  VHSR_CLKNAND2_2 U258 ( .A1(n265), .A2(n259), .ZN(n258) );
  VHSR_CLKNAND2_2 U259 ( .A1(n224), .A2(n258), .ZN(n248) );
  VHSR_CLKNAND2_2 U260 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U261 ( .A1(n225), .A2(n247), .ZN(n243) );
  VHSR_NOR2_1 U262 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_NOR2_1 U263 ( .A1(n246), .A2(n242), .ZN(n235) );
  VHSR_AND3_2 U264 ( .A1(n235), .A2(b[3]), .A3(a[7]), .Z(n299) );
  VHSR_IN_2 U265 ( .I(b[7]), .ZN(n282) );
  VHSR_IN_2 U266 ( .I(a[3]), .ZN(n328) );
  VHSR_IN_2 U267 ( .I(b[6]), .ZN(n283) );
  VHSR_IN_2 U268 ( .I(a[2]), .ZN(n322) );
  VHSR_OAI22_2 U269 ( .A1(n283), .A2(n328), .B1(n282), .B2(n322), .ZN(n241) );
  VHSR_NOR2_1 U270 ( .A1(n282), .A2(n322), .ZN(n227) );
  VHSR_NOR2_1 U271 ( .A1(n283), .A2(n320), .ZN(n226) );
  VHSR_IN_2 U272 ( .I(b[5]), .ZN(n279) );
  VHSR_AOI211_2 U273 ( .A1(b[4]), .A2(a[2]), .B(n279), .C(n328), .ZN(n232) );
  VHSR_OAI22_2 U274 ( .A1(n283), .A2(n322), .B1(n282), .B2(n320), .ZN(n231) );
  VHSR_AOI22_2 U275 ( .A1(n227), .A2(n226), .B1(n232), .B2(n231), .ZN(n233) );
  VHSR_CLKNAND2_2 U276 ( .A1(b[4]), .A2(a[2]), .ZN(n264) );
  VHSR_CLKNAND2_2 U277 ( .A1(b[4]), .A2(a[0]), .ZN(n394) );
  VHSR_NAND3_2 U278 ( .A1(a[1]), .A2(b[5]), .A3(n394), .ZN(n263) );
  VHSR_CLKNAND2_2 U279 ( .A1(b[6]), .A2(a[0]), .ZN(n262) );
  VHSR_MAOI222_2 U280 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_IN_2 U281 ( .I(b[4]), .ZN(n278) );
  VHSR_NOR4_2 U282 ( .A1(n278), .A2(n279), .A3(n320), .A4(n402), .ZN(n270) );
  VHSR_NAND4_2 U283 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_OAI22_2 U284 ( .A1(n278), .A2(n328), .B1(n279), .B2(n322), .ZN(n228) );
  VHSR_AND2_2 U285 ( .A1(n238), .A2(n228), .Z(n230) );
  VHSR_OAI22_2 U286 ( .A1(n283), .A2(n320), .B1(n282), .B2(n402), .ZN(n229) );
  VHSR_AND2_2 U287 ( .A1(n261), .A2(n257), .Z(n256) );
  VHSR_AD1_1 U288 ( .A(n270), .B(n230), .CI(n229), .CO(n251), .S(n257) );
  VHSR_NOR2_1 U289 ( .A1(n256), .A2(n251), .ZN(n254) );
  VHSR_OAI21_2 U290 ( .A1(n232), .A2(n231), .B(n233), .ZN(n255) );
  VHSR_NOR2_1 U291 ( .A1(n254), .A2(n255), .ZN(n252) );
  VHSR_CLKNAND2_2 U292 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U293 ( .A1(n241), .A2(n237), .ZN(n236) );
  VHSR_NOR3_2 U294 ( .A1(n282), .A2(n328), .A3(n236), .ZN(n298) );
  VHSR_NOR2_1 U295 ( .A1(n327), .A2(n280), .ZN(n234) );
  VHSR_IAO21_2 U296 ( .A1(n235), .A2(n234), .B(n299), .ZN(n302) );
  VHSR_OAI32_2 U297 ( .A1(n298), .A2(n328), .A3(n282), .B1(n236), .B2(n298), 
        .ZN(n301) );
  VHSR_OAI21_2 U298 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U299 ( .A1(n241), .A2(n240), .ZN(n309) );
  VHSR_AOI21_2 U300 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U301 ( .A1(n246), .A2(n245), .ZN(n308) );
  VHSR_OAI21_2 U302 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_IN_2 U303 ( .I(n250), .ZN(n314) );
  VHSR_CLKNAND2_2 U304 ( .A1(n256), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U305 ( .A1(n255), .A2(n254), .B1(n253), .B2(n252), .ZN(n313) );
  VHSR_IAO21_2 U306 ( .A1(n261), .A2(n257), .B(n256), .ZN(n317) );
  VHSR_OAI21_2 U307 ( .A1(n265), .A2(n259), .B(n258), .ZN(n260) );
  VHSR_IN_2 U308 ( .I(n260), .ZN(n316) );
  VHSR_AOI31_2 U309 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n332) );
  VHSR_AOI31_2 U310 ( .A1(n268), .A2(n267), .A3(n266), .B(n265), .ZN(n331) );
  VHSR_CLKNAND2_2 U311 ( .A1(b[5]), .A2(a[0]), .ZN(n269) );
  VHSR_OAI32_2 U312 ( .A1(n270), .A2(n320), .A3(n278), .B1(n269), .B2(n270), 
        .ZN(n344) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[5]), .A2(b[0]), .ZN(n271) );
  VHSR_OAI32_2 U314 ( .A1(n272), .A2(n323), .A3(n284), .B1(n271), .B2(n272), 
        .ZN(n343) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[6]), .A2(b[4]), .ZN(n306) );
  VHSR_NAND3_2 U316 ( .A1(a[7]), .A2(b[5]), .A3(n306), .ZN(n275) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[4]), .A2(b[6]), .ZN(n305) );
  VHSR_NAND3_2 U318 ( .A1(b[7]), .A2(a[5]), .A3(n305), .ZN(n274) );
  VHSR_CLKNAND2_2 U319 ( .A1(n275), .A2(n274), .ZN(n277) );
  VHSR_IN_2 U320 ( .I(n388), .ZN(n361) );
  VHSR_MAOI222_2 U321 ( .A(n361), .B(n275), .C(n274), .ZN(n345) );
  VHSR_IN_2 U322 ( .I(n345), .ZN(n276) );
  VHSR_OAI21_2 U323 ( .A1(n388), .A2(n277), .B(n276), .ZN(n293) );
  VHSR_AND3_2 U324 ( .A1(n373), .A2(a[5]), .A3(b[5]), .Z(n310) );
  VHSR_NOR3_2 U325 ( .A1(n280), .A2(n306), .A3(n279), .ZN(n353) );
  VHSR_AOI22_2 U326 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n281) );
  VHSR_NOR2_1 U327 ( .A1(n353), .A2(n281), .ZN(n289) );
  VHSR_NOR4_2 U328 ( .A1(n285), .A2(n284), .A3(n283), .A4(n282), .ZN(n351) );
  VHSR_AOI22_2 U329 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n286) );
  VHSR_NOR2_1 U330 ( .A1(n351), .A2(n286), .ZN(n288) );
  VHSR_IN_2 U331 ( .I(n287), .ZN(n296) );
  VHSR_MAOI222_2 U332 ( .A(n306), .B(n305), .C(n304), .ZN(n303) );
  VHSR_IN_2 U333 ( .I(n303), .ZN(n295) );
  VHSR_AD1_1 U334 ( .A(n310), .B(n289), .CI(n288), .CO(n290), .S(n287) );
  VHSR_NOR2_1 U335 ( .A1(n294), .A2(n290), .ZN(n292) );
  VHSR_CLKNAND2_2 U336 ( .A1(n294), .A2(n290), .ZN(n291) );
  VHSR_NOR2_1 U337 ( .A1(n292), .A2(n293), .ZN(n346) );
  VHSR_AOI22_2 U338 ( .A1(n293), .A2(n292), .B1(n291), .B2(n346), .ZN(n386) );
  VHSR_AOI21_2 U339 ( .A1(n296), .A2(n295), .B(n294), .ZN(n365) );
  VHSR_AD1_1 U340 ( .A(n299), .B(n298), .CI(n297), .CO(n387), .S(n364) );
  VHSR_AD1_1 U341 ( .A(n302), .B(n301), .CI(n300), .CO(n297), .S(n368) );
  VHSR_AOI31_2 U342 ( .A1(n306), .A2(n305), .A3(n304), .B(n303), .ZN(n367) );
  VHSR_AD1_1 U343 ( .A(n309), .B(n308), .CI(n307), .CO(n300), .S(n371) );
  VHSR_AOI22_2 U344 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n311) );
  VHSR_NOR2_1 U345 ( .A1(n311), .A2(n310), .ZN(n370) );
  VHSR_AD1_1 U346 ( .A(n314), .B(n313), .CI(n312), .CO(n307), .S(n374) );
  VHSR_AD1_1 U347 ( .A(n317), .B(n316), .CI(n315), .CO(n312), .S(n377) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[2]), .A2(a[2]), .ZN(n329) );
  VHSR_IN_2 U349 ( .I(n329), .ZN(n336) );
  VHSR_AOI22_2 U350 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n318) );
  VHSR_AOI31_2 U351 ( .A1(a[3]), .A2(b[3]), .A3(n336), .B(n318), .ZN(n342) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[3]), .A2(a[1]), .ZN(n319) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[1]), .A2(a[3]), .ZN(n324) );
  VHSR_MAOI222_2 U354 ( .A(n329), .B(n319), .C(n324), .ZN(n326) );
  VHSR_OAI22_2 U355 ( .A1(n327), .A2(n402), .B1(n401), .B2(n320), .ZN(n384) );
  VHSR_AOI22_2 U356 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n400) );
  VHSR_NOR3_2 U357 ( .A1(n400), .A2(n402), .A3(n401), .ZN(n403) );
  VHSR_OAI22_2 U358 ( .A1(n323), .A2(n322), .B1(n321), .B2(n328), .ZN(n383) );
  VHSR_AOI21_2 U359 ( .A1(a[1]), .A2(b[3]), .B(n336), .ZN(n325) );
  VHSR_AOI21_2 U360 ( .A1(n325), .A2(n324), .B(n326), .ZN(n338) );
  VHSR_CLKNAND2_2 U361 ( .A1(n339), .A2(n338), .ZN(n337) );
  VHSR_CLKNAND2_2 U362 ( .A1(n342), .A2(n341), .ZN(n333) );
  VHSR_AOI211_2 U363 ( .A1(n329), .A2(n333), .B(n328), .C(n327), .ZN(n376) );
  VHSR_AD1_1 U364 ( .A(n332), .B(n331), .CI(n330), .CO(n315), .S(n380) );
  VHSR_IN_2 U365 ( .I(n333), .ZN(n340) );
  VHSR_CLKNAND2_2 U366 ( .A1(b[3]), .A2(a[3]), .ZN(n335) );
  VHSR_CLKNAND2_2 U367 ( .A1(n340), .A2(n335), .ZN(n334) );
  VHSR_OAI31_2 U368 ( .A1(n336), .A2(n340), .A3(n335), .B(n334), .ZN(n379) );
  VHSR_OAI21_2 U369 ( .A1(n339), .A2(n338), .B(n337), .ZN(n399) );
  VHSR_AOI211_2 U370 ( .A1(n395), .A2(n394), .B(n393), .C(n399), .ZN(n397) );
  VHSR_IAO21_2 U371 ( .A1(n342), .A2(n341), .B(n340), .ZN(n382) );
  VHSR_AD1_1 U372 ( .A(n344), .B(n343), .CI(n393), .CO(n330), .S(n381) );
  VHSR_CLKNAND2_2 U373 ( .A1(a[7]), .A2(b[6]), .ZN(n348) );
  VHSR_AOI21_2 U374 ( .A1(a[6]), .A2(b[7]), .B(n348), .ZN(n347) );
  VHSR_AOI31_2 U375 ( .A1(a[6]), .A2(n348), .A3(b[7]), .B(n347), .ZN(n349) );
  VHSR_IN_2 U376 ( .I(n349), .ZN(n350) );
  VHSR_OR2_2 U377 ( .A1(n351), .A2(n350), .Z(n352) );
  VHSR_MAOI222_2 U378 ( .A(n353), .B(n351), .C(n350), .ZN(n360) );
  VHSR_OAI21_2 U379 ( .A1(n353), .A2(n352), .B(n360), .ZN(n357) );
  VHSR_CLKXOR2_2 U380 ( .A1(n358), .A2(n357), .Z(n354) );
  VHSR_CLKNAND2_2 U381 ( .A1(n355), .A2(n354), .ZN(n390) );
  VHSR_OAI21_2 U382 ( .A1(n355), .A2(n354), .B(n390), .ZN(n356) );
  VHSR_CLKNAND2_2 U383 ( .A1(a[7]), .A2(b[7]), .ZN(n389) );
  VHSR_NOR2_1 U384 ( .A1(n358), .A2(n357), .ZN(n359) );
  VHSR_AND3_2 U385 ( .A1(n391), .A2(n361), .A3(n390), .Z(n362) );
  VHSR_NOR2_1 U386 ( .A1(n389), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U387 ( .A(n387), .B(n386), .CI(n385), .CO(n355), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U388 ( .A1(n389), .A2(n388), .ZN(n392) );
  VHSR_XOR3_2 U389 ( .A1(n392), .A2(n391), .A3(n390), .Z(product[14]) );
  VHSR_AOI21_2 U390 ( .A1(n395), .A2(n394), .B(n393), .ZN(n396) );
  VHSR_IN_2 U391 ( .I(n396), .ZN(n398) );
  VHSR_AOI21_2 U392 ( .A1(n399), .A2(n398), .B(n397), .ZN(product[4]) );
  VHSR_OAI32_2 U393 ( .A1(n403), .A2(n402), .A3(n401), .B1(n400), .B2(n403), 
        .ZN(product[2]) );
endmodule


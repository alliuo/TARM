
module mul8_148 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n217, n218,
         n219, n220, n221, n222, n223, n224, n225, n226, n227, n228, n229,
         n230, n231, n232, n233, n234, n235, n236, n237, n238, n239, n240,
         n241, n242, n243, n244, n245, n246, n247, n248, n249, n250, n251,
         n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
         n263, n264, n265, n266, n267, n268, n269, n270, n271, n272, n273,
         n274, n275, n276, n277, n278, n279, n280, n281, n282, n283, n284,
         n285, n286, n287, n288, n289, n290, n291, n292, n293, n294, n295,
         n296, n297, n298, n299, n300, n301, n302, n303, n304, n305, n306,
         n307, n308, n309, n310, n311, n312, n313, n314, n315, n316, n317,
         n318, n319, n320, n321, n322, n323, n324, n325, n326, n327, n328,
         n329, n330, n331, n332, n333, n334, n335, n336, n337, n338, n339,
         n340, n341, n342, n343, n344, n345, n346, n347, n348, n349, n350,
         n351, n352, n353, n354, n355, n356, n357, n358, n359, n360, n361,
         n362, n363, n364, n365, n366, n367, n368, n369, n370, n371, n372,
         n373, n374, n375, n376, n377, n378, n379, n380, n381, n382, n383,
         n384, n385, n386, n387, n388, n389, n390, n391, n392, n393, n394,
         n395, n396, n397, n398, n399, n400, n401, n402, n403, n404, n405;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U207 ( .A1(n229), .B1(n220), .ZN(n257) );
  VHSR_INAND2_2 U208 ( .A1(n323), .B1(n322), .ZN(n324) );
  VHSR_NOR2_1 U209 ( .A1(n346), .A2(n345), .ZN(n358) );
  VHSR_NOR2_1 U210 ( .A1(n317), .A2(n318), .ZN(n333) );
  VHSR_IOA21_2 U211 ( .A1(n395), .A2(n394), .B(n393), .ZN(n397) );
  VHSR_INOR2_2 U212 ( .A1(n360), .B1(n359), .ZN(n391) );
  VHSR_IN_2 U213 ( .I(n356), .ZN(product[13]) );
  VHSR_MOAI22_1 U214 ( .A1(n280), .A2(n318), .B1(b[4]), .B2(a[3]), .ZN(n231)
         );
  VHSR_AD1_1 U215 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[9])
         );
  VHSR_AD1_1 U216 ( .A(n375), .B(n403), .CI(n374), .CO(n338), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n396), .B(n373), .CI(n372), .CO(n376), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U219 ( .A(n365), .B(n364), .CI(n363), .CO(n382), .S(product[10])
         );
  VHSR_AOI22_2 U220 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n250) );
  VHSR_IN_2 U221 ( .I(b[3]), .ZN(n320) );
  VHSR_IN_2 U222 ( .I(b[2]), .ZN(n317) );
  VHSR_IN_2 U223 ( .I(a[5]), .ZN(n286) );
  VHSR_IN_2 U224 ( .I(a[4]), .ZN(n285) );
  VHSR_NOR4_2 U225 ( .A1(n320), .A2(n317), .A3(n286), .A4(n285), .ZN(n248) );
  VHSR_CLKNAND2_2 U226 ( .A1(a[6]), .A2(b[1]), .ZN(n226) );
  VHSR_IN_2 U227 ( .I(n226), .ZN(n223) );
  VHSR_IN_2 U228 ( .I(a[7]), .ZN(n281) );
  VHSR_IN_2 U229 ( .I(b[1]), .ZN(n402) );
  VHSR_OAI21_2 U230 ( .A1(n281), .A2(n402), .B(n317), .ZN(n219) );
  VHSR_CLKNAND2_2 U231 ( .A1(b[2]), .A2(a[4]), .ZN(n271) );
  VHSR_CLKNAND2_2 U232 ( .A1(a[5]), .A2(n271), .ZN(n218) );
  VHSR_I2NOR4_2 U233 ( .A1(n219), .A2(a[6]), .B1(n320), .B2(n218), .ZN(n217)
         );
  VHSR_AOI31_2 U234 ( .A1(a[7]), .A2(b[2]), .A3(n223), .B(n217), .ZN(n229) );
  VHSR_IAO22_2 U235 ( .B1(a[6]), .B2(n219), .A1(n320), .A2(n218), .ZN(n220) );
  VHSR_IN_2 U236 ( .I(b[0]), .ZN(n400) );
  VHSR_NOR4_2 U237 ( .A1(n286), .A2(n285), .A3(n402), .A4(n400), .ZN(n275) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[2]), .A2(a[5]), .ZN(n222) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[3]), .A2(a[4]), .ZN(n221) );
  VHSR_AOI21_2 U240 ( .A1(n222), .A2(n221), .B(n248), .ZN(n224) );
  VHSR_MAOI222_2 U241 ( .A(n223), .B(n275), .C(n224), .ZN(n228) );
  VHSR_CLKNAND2_2 U242 ( .A1(a[6]), .A2(b[0]), .ZN(n270) );
  VHSR_OAI211_2 U243 ( .A1(n285), .A2(n400), .B(a[5]), .C(b[1]), .ZN(n269) );
  VHSR_MAOI222_2 U244 ( .A(n271), .B(n270), .C(n269), .ZN(n268) );
  VHSR_NOR2_1 U245 ( .A1(n275), .A2(n224), .ZN(n227) );
  VHSR_IN_2 U246 ( .I(n228), .ZN(n225) );
  VHSR_AOI21_2 U247 ( .A1(n227), .A2(n226), .B(n225), .ZN(n260) );
  VHSR_CLKNAND2_2 U248 ( .A1(n268), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U249 ( .A1(n228), .A2(n259), .ZN(n256) );
  VHSR_CLKNAND2_2 U250 ( .A1(n257), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U251 ( .A1(n229), .A2(n255), .ZN(n247) );
  VHSR_NOR2_1 U252 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_NOR2_1 U253 ( .A1(n250), .A2(n246), .ZN(n239) );
  VHSR_AND3_2 U254 ( .A1(n239), .A2(b[3]), .A3(a[7]), .Z(n299) );
  VHSR_IN_2 U255 ( .I(b[7]), .ZN(n283) );
  VHSR_IN_2 U256 ( .I(a[3]), .ZN(n319) );
  VHSR_IN_2 U257 ( .I(b[6]), .ZN(n284) );
  VHSR_IN_2 U258 ( .I(a[2]), .ZN(n318) );
  VHSR_OAI22_2 U259 ( .A1(n284), .A2(n319), .B1(n283), .B2(n318), .ZN(n245) );
  VHSR_AOI22_2 U260 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n236) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[4]), .A2(a[2]), .ZN(n267) );
  VHSR_NAND3_2 U262 ( .A1(a[3]), .A2(b[5]), .A3(n267), .ZN(n235) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[7]), .A2(a[2]), .ZN(n230) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[6]), .A2(a[1]), .ZN(n232) );
  VHSR_OAI22_2 U265 ( .A1(n236), .A2(n235), .B1(n230), .B2(n232), .ZN(n237) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[4]), .A2(a[0]), .ZN(n394) );
  VHSR_NAND3_2 U267 ( .A1(a[1]), .A2(b[5]), .A3(n394), .ZN(n266) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[6]), .A2(a[0]), .ZN(n265) );
  VHSR_MAOI222_2 U269 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NAND4_2 U270 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n242) );
  VHSR_IN_2 U271 ( .I(b[5]), .ZN(n280) );
  VHSR_AND2_2 U272 ( .A1(n242), .A2(n231), .Z(n234) );
  VHSR_IN_2 U273 ( .I(a[0]), .ZN(n401) );
  VHSR_OAI21_2 U274 ( .A1(n283), .A2(n401), .B(n232), .ZN(n233) );
  VHSR_IN_2 U275 ( .I(a[1]), .ZN(n399) );
  VHSR_NOR3_2 U276 ( .A1(n280), .A2(n399), .A3(n394), .ZN(n273) );
  VHSR_AND2_2 U277 ( .A1(n264), .A2(n263), .Z(n262) );
  VHSR_AD1_1 U278 ( .A(n234), .B(n233), .CI(n273), .CO(n251), .S(n263) );
  VHSR_AOI21_2 U279 ( .A1(n236), .A2(n235), .B(n237), .ZN(n254) );
  VHSR_OAI32_2 U280 ( .A1(n237), .A2(n262), .A3(n251), .B1(n254), .B2(n237), 
        .ZN(n243) );
  VHSR_CLKNAND2_2 U281 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U282 ( .A1(n245), .A2(n241), .ZN(n240) );
  VHSR_NOR3_2 U283 ( .A1(n283), .A2(n319), .A3(n240), .ZN(n298) );
  VHSR_NOR2_1 U284 ( .A1(n320), .A2(n281), .ZN(n238) );
  VHSR_IAO21_2 U285 ( .A1(n239), .A2(n238), .B(n299), .ZN(n302) );
  VHSR_OAI32_2 U286 ( .A1(n298), .A2(n319), .A3(n283), .B1(n240), .B2(n298), 
        .ZN(n301) );
  VHSR_OAI21_2 U287 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_XNOR2_2 U288 ( .A1(n245), .A2(n244), .ZN(n309) );
  VHSR_AOI21_2 U289 ( .A1(n248), .A2(n247), .B(n246), .ZN(n249) );
  VHSR_XNOR2_2 U290 ( .A1(n250), .A2(n249), .ZN(n308) );
  VHSR_NOR2_1 U291 ( .A1(n262), .A2(n251), .ZN(n253) );
  VHSR_AOI22_2 U292 ( .A1(n262), .A2(n251), .B1(n254), .B2(n253), .ZN(n252) );
  VHSR_OAI21_2 U293 ( .A1(n254), .A2(n253), .B(n252), .ZN(n314) );
  VHSR_OAI21_2 U294 ( .A1(n257), .A2(n256), .B(n255), .ZN(n258) );
  VHSR_IN_2 U295 ( .I(n258), .ZN(n313) );
  VHSR_OAI21_2 U296 ( .A1(n268), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U297 ( .I(n261), .ZN(n329) );
  VHSR_IAO21_2 U298 ( .A1(n264), .A2(n263), .B(n262), .ZN(n328) );
  VHSR_AOI31_2 U299 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n336) );
  VHSR_AOI31_2 U300 ( .A1(n271), .A2(n270), .A3(n269), .B(n268), .ZN(n335) );
  VHSR_AOI22_2 U301 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n272) );
  VHSR_NOR2_1 U302 ( .A1(n273), .A2(n272), .ZN(n341) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[4]), .ZN(n288) );
  VHSR_IN_2 U304 ( .I(n288), .ZN(n370) );
  VHSR_NOR2_1 U305 ( .A1(n400), .A2(n401), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U306 ( .A1(n370), .A2(product[0]), .ZN(n393) );
  VHSR_IN_2 U307 ( .I(n393), .ZN(n340) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[1]), .ZN(n274) );
  VHSR_OAI32_2 U309 ( .A1(n275), .A2(n400), .A3(n286), .B1(n274), .B2(n275), 
        .ZN(n339) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[6]), .ZN(n361) );
  VHSR_IN_2 U311 ( .I(n361), .ZN(n388) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n306) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n306), .ZN(n277) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[6]), .ZN(n305) );
  VHSR_NAND3_2 U315 ( .A1(b[7]), .A2(a[5]), .A3(n305), .ZN(n276) );
  VHSR_CLKNAND2_2 U316 ( .A1(n277), .A2(n276), .ZN(n279) );
  VHSR_MAOI222_2 U317 ( .A(n361), .B(n277), .C(n276), .ZN(n345) );
  VHSR_IN_2 U318 ( .I(n345), .ZN(n278) );
  VHSR_OAI21_2 U319 ( .A1(n388), .A2(n279), .B(n278), .ZN(n294) );
  VHSR_NOR3_2 U320 ( .A1(n286), .A2(n280), .A3(n288), .ZN(n310) );
  VHSR_NOR3_2 U321 ( .A1(n281), .A2(n306), .A3(n280), .ZN(n353) );
  VHSR_AOI22_2 U322 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n282) );
  VHSR_NOR2_1 U323 ( .A1(n353), .A2(n282), .ZN(n290) );
  VHSR_NOR4_2 U324 ( .A1(n286), .A2(n285), .A3(n284), .A4(n283), .ZN(n351) );
  VHSR_AOI22_2 U325 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n287) );
  VHSR_NOR2_1 U326 ( .A1(n351), .A2(n287), .ZN(n289) );
  VHSR_NAND3_2 U327 ( .A1(b[5]), .A2(a[5]), .A3(n288), .ZN(n304) );
  VHSR_MAOI222_2 U328 ( .A(n306), .B(n305), .C(n304), .ZN(n303) );
  VHSR_AND2_2 U329 ( .A1(n296), .A2(n303), .Z(n295) );
  VHSR_AD1_1 U330 ( .A(n310), .B(n290), .CI(n289), .CO(n291), .S(n296) );
  VHSR_NOR2_1 U331 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U332 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_NOR2_1 U333 ( .A1(n293), .A2(n294), .ZN(n346) );
  VHSR_AOI22_2 U334 ( .A1(n294), .A2(n293), .B1(n292), .B2(n346), .ZN(n386) );
  VHSR_IAO21_2 U335 ( .A1(n296), .A2(n303), .B(n295), .ZN(n384) );
  VHSR_AD1_1 U336 ( .A(n299), .B(n298), .CI(n297), .CO(n387), .S(n383) );
  VHSR_AD1_1 U337 ( .A(n302), .B(n301), .CI(n300), .CO(n297), .S(n365) );
  VHSR_AOI31_2 U338 ( .A1(n306), .A2(n305), .A3(n304), .B(n303), .ZN(n364) );
  VHSR_AD1_1 U339 ( .A(n309), .B(n308), .CI(n307), .CO(n300), .S(n368) );
  VHSR_AOI22_2 U340 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n311) );
  VHSR_NOR2_1 U341 ( .A1(n311), .A2(n310), .ZN(n367) );
  VHSR_AD1_1 U342 ( .A(n314), .B(n313), .CI(n312), .CO(n307), .S(n371) );
  VHSR_NOR4_2 U343 ( .A1(n320), .A2(n317), .A3(n399), .A4(n401), .ZN(n344) );
  VHSR_CLKNAND2_2 U344 ( .A1(b[3]), .A2(a[3]), .ZN(n331) );
  VHSR_IN_2 U345 ( .I(n333), .ZN(n322) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n315) );
  VHSR_IAO21_2 U347 ( .A1(n331), .A2(n322), .B(n315), .ZN(n343) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[2]), .A2(a[1]), .ZN(n316) );
  VHSR_OAI32_2 U349 ( .A1(n344), .A2(n401), .A3(n320), .B1(n316), .B2(n344), 
        .ZN(n375) );
  VHSR_AOI22_2 U350 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n405) );
  VHSR_NOR3_2 U351 ( .A1(n405), .A2(n401), .A3(n317), .ZN(n403) );
  VHSR_OAI22_2 U352 ( .A1(n402), .A2(n318), .B1(n400), .B2(n319), .ZN(n374) );
  VHSR_IN_2 U353 ( .I(n338), .ZN(n326) );
  VHSR_NOR2_1 U354 ( .A1(n402), .A2(n319), .ZN(n321) );
  VHSR_AOI211_2 U355 ( .A1(b[2]), .A2(a[0]), .B(n320), .C(n399), .ZN(n323) );
  VHSR_MAOI222_2 U356 ( .A(n321), .B(n333), .C(n323), .ZN(n325) );
  VHSR_AOI32_2 U357 ( .A1(a[3]), .A2(n325), .A3(b[1]), .B1(n324), .B2(n325), 
        .ZN(n337) );
  VHSR_OAI21_2 U358 ( .A1(n326), .A2(n337), .B(n325), .ZN(n342) );
  VHSR_IAO21_2 U359 ( .A1(n333), .A2(n332), .B(n331), .ZN(n381) );
  VHSR_AD1_1 U360 ( .A(n329), .B(n328), .CI(n327), .CO(n312), .S(n380) );
  VHSR_OAI21_2 U361 ( .A1(n333), .A2(n331), .B(n332), .ZN(n330) );
  VHSR_OAI31_2 U362 ( .A1(n333), .A2(n332), .A3(n331), .B(n330), .ZN(n378) );
  VHSR_AD1_1 U363 ( .A(n336), .B(n335), .CI(n334), .CO(n327), .S(n377) );
  VHSR_CLKNAND2_2 U364 ( .A1(a[4]), .A2(b[0]), .ZN(n395) );
  VHSR_CLKXOR2_2 U365 ( .A1(n338), .A2(n337), .Z(n398) );
  VHSR_AOI211_2 U366 ( .A1(n395), .A2(n394), .B(n340), .C(n398), .ZN(n396) );
  VHSR_AD1_1 U367 ( .A(n341), .B(n340), .CI(n339), .CO(n334), .S(n373) );
  VHSR_AD1_1 U368 ( .A(n344), .B(n343), .CI(n342), .CO(n332), .S(n372) );
  VHSR_CLKNAND2_2 U369 ( .A1(a[7]), .A2(b[6]), .ZN(n348) );
  VHSR_AOI21_2 U370 ( .A1(a[6]), .A2(b[7]), .B(n348), .ZN(n347) );
  VHSR_AOI31_2 U371 ( .A1(a[6]), .A2(n348), .A3(b[7]), .B(n347), .ZN(n349) );
  VHSR_IN_2 U372 ( .I(n349), .ZN(n350) );
  VHSR_OR2_2 U373 ( .A1(n351), .A2(n350), .Z(n352) );
  VHSR_MAOI222_2 U374 ( .A(n353), .B(n351), .C(n350), .ZN(n360) );
  VHSR_OAI21_2 U375 ( .A1(n353), .A2(n352), .B(n360), .ZN(n357) );
  VHSR_CLKXOR2_2 U376 ( .A1(n358), .A2(n357), .Z(n354) );
  VHSR_CLKNAND2_2 U377 ( .A1(n355), .A2(n354), .ZN(n390) );
  VHSR_OAI21_2 U378 ( .A1(n355), .A2(n354), .B(n390), .ZN(n356) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[7]), .A2(b[7]), .ZN(n389) );
  VHSR_NOR2_1 U380 ( .A1(n358), .A2(n357), .ZN(n359) );
  VHSR_AND3_2 U381 ( .A1(n391), .A2(n361), .A3(n390), .Z(n362) );
  VHSR_NOR2_1 U382 ( .A1(n389), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U383 ( .A(n378), .B(n377), .CI(n376), .CO(n379), .S(product[6])
         );
  VHSR_AD1_1 U384 ( .A(n381), .B(n380), .CI(n379), .CO(n369), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U385 ( .A(n384), .B(n383), .CI(n382), .CO(n385), .S(product[11])
         );
  VHSR_AD1_1 U386 ( .A(n387), .B(n386), .CI(n385), .CO(n355), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U387 ( .A1(n389), .A2(n388), .ZN(n392) );
  VHSR_XOR3_2 U388 ( .A1(n392), .A2(n391), .A3(n390), .Z(product[14]) );
  VHSR_AOI21_2 U389 ( .A1(n398), .A2(n397), .B(n396), .ZN(product[4]) );
  VHSR_OAI22_2 U390 ( .A1(n402), .A2(n401), .B1(n400), .B2(n399), .ZN(
        product[1]) );
  VHSR_CLKNAND2_2 U391 ( .A1(b[2]), .A2(a[0]), .ZN(n404) );
  VHSR_AOI21_2 U392 ( .A1(n405), .A2(n404), .B(n403), .ZN(product[2]) );
endmodule


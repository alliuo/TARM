
module mul8_19 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \mul_ll_ll/out[0] , \intadd_0/SUM[7] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n213, n214, n215, n216, n217, n218, n219, n220,
         n221, n222, n223, n224, n225, n226, n227, n228, n229, n230, n231,
         n232, n233, n234, n235, n236, n237, n238, n239, n240, n241, n242,
         n243, n244, n245, n246, n247, n248, n249, n250, n251, n252, n253,
         n254, n255, n256, n257, n258, n259, n260, n261, n262, n263, n264,
         n265, n266, n267, n268, n269, n270, n271, n272, n273, n274, n275,
         n276, n277, n278, n279, n280, n281, n282, n283, n284, n285, n286,
         n287, n288, n289, n290, n291, n292, n293, n294, n295, n296, n297,
         n298, n299, n300, n301, n302, n303, n304, n305, n306, n307, n308,
         n309, n310, n311, n312, n313, n314, n315, n316, n317, n318, n319,
         n320, n321, n322, n323, n324, n325, n326, n327, n328, n329, n330,
         n331, n332, n333, n334, n335, n336, n337, n338, n339, n340, n341,
         n342, n343, n344, n345, n346, n347, n348, n349, n350, n351, n352,
         n353, n354, n355, n356, n357, n358, n359, n360, n361, n362, n363,
         n364, n365, n366, n367, n368, n369, n370, n371, n372, n373, n374,
         n375, n376, n377, n378, n379, n380, n381, n382, n383, n384, n385,
         n386, n387, n388, n389, n390, n391, n392, n393, n394, n395, n396,
         n397, n398, n399, n400;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND3_2 U204 ( .A1(n265), .B1(a[5]), .B2(b[3]), .ZN(n221) );
  VHSR_INAND2_2 U205 ( .A1(n270), .B1(n216), .ZN(n217) );
  VHSR_INOR2_2 U206 ( .A1(n233), .B1(n250), .ZN(n238) );
  VHSR_INOR2_2 U207 ( .A1(n219), .B1(n256), .ZN(n248) );
  VHSR_INOR3_2 U208 ( .A1(n364), .B1(n283), .B2(n284), .ZN(n307) );
  VHSR_NOR2_1 U209 ( .A1(n276), .A2(n234), .ZN(n296) );
  VHSR_INOR2_2 U210 ( .A1(n357), .B1(n356), .ZN(n388) );
  VHSR_IN_2 U211 ( .I(n353), .ZN(product[13]) );
  VHSR_INAND2_1 U212 ( .A1(n348), .B1(n346), .ZN(n349) );
  VHSR_AD1_1 U213 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U214 ( .A(n372), .B(n400), .CI(n371), .CO(n336), .S(product[3])
         );
  VHSR_AD1_1 U215 ( .A(n394), .B(n370), .CI(n369), .CO(n373), .S(product[5])
         );
  VHSR_AD1_1 U216 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U217 ( .A(n362), .B(n361), .CI(n360), .CO(n376), .S(product[9])
         );
  VHSR_PULL0_0 U218 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U219 ( .I(b[1]), .ZN(n323) );
  VHSR_IN_2 U220 ( .I(a[0]), .ZN(n399) );
  VHSR_NOR2_1 U221 ( .A1(n323), .A2(n399), .ZN(product[1]) );
  VHSR_IN_2 U222 ( .I(a[7]), .ZN(n276) );
  VHSR_IN_2 U223 ( .I(b[2]), .ZN(n398) );
  VHSR_IN_2 U224 ( .I(a[4]), .ZN(n280) );
  VHSR_NOR2_1 U225 ( .A1(n398), .A2(n280), .ZN(n265) );
  VHSR_NAND3_2 U226 ( .A1(a[6]), .A2(a[7]), .A3(b[1]), .ZN(n220) );
  VHSR_OAI21_2 U227 ( .A1(a[6]), .A2(a[7]), .B(b[2]), .ZN(n223) );
  VHSR_MAOI222_2 U228 ( .A(n221), .B(n220), .C(n223), .ZN(n225) );
  VHSR_IN_2 U229 ( .I(a[6]), .ZN(n215) );
  VHSR_NOR2_1 U230 ( .A1(n215), .A2(n323), .ZN(n218) );
  VHSR_IN_2 U231 ( .I(a[5]), .ZN(n283) );
  VHSR_IN_2 U232 ( .I(b[0]), .ZN(n320) );
  VHSR_NOR4_2 U233 ( .A1(n283), .A2(n280), .A3(n323), .A4(n320), .ZN(n270) );
  VHSR_NAND4_2 U234 ( .A1(b[3]), .A2(b[2]), .A3(a[5]), .A4(a[4]), .ZN(n243) );
  VHSR_IN_2 U235 ( .I(b[3]), .ZN(n319) );
  VHSR_NOR2_1 U236 ( .A1(n319), .A2(n280), .ZN(n213) );
  VHSR_AOI32_2 U237 ( .A1(a[5]), .A2(n243), .A3(b[2]), .B1(n213), .B2(n243), 
        .ZN(n216) );
  VHSR_IN_2 U238 ( .I(n216), .ZN(n214) );
  VHSR_MAOI222_2 U239 ( .A(n218), .B(n270), .C(n214), .ZN(n219) );
  VHSR_AOI211_2 U240 ( .A1(a[4]), .A2(b[0]), .B(n283), .C(n323), .ZN(n264) );
  VHSR_NOR2_1 U241 ( .A1(n215), .A2(n320), .ZN(n263) );
  VHSR_MAOI222_2 U242 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_OAI21_2 U243 ( .A1(n218), .A2(n217), .B(n219), .ZN(n257) );
  VHSR_NOR2_1 U244 ( .A1(n262), .A2(n257), .ZN(n256) );
  VHSR_AND2_2 U245 ( .A1(n221), .A2(n220), .Z(n222) );
  VHSR_AOI21_2 U246 ( .A1(n223), .A2(n222), .B(n225), .ZN(n224) );
  VHSR_IN_2 U247 ( .I(n224), .ZN(n247) );
  VHSR_NOR2_1 U248 ( .A1(n248), .A2(n247), .ZN(n246) );
  VHSR_NOR2_1 U249 ( .A1(n225), .A2(n246), .ZN(n242) );
  VHSR_CLKNAND2_2 U250 ( .A1(n242), .A2(n243), .ZN(n241) );
  VHSR_NAND3_2 U251 ( .A1(a[6]), .A2(b[3]), .A3(n241), .ZN(n234) );
  VHSR_IN_2 U252 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U253 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U254 ( .I(b[6]), .ZN(n279) );
  VHSR_IN_2 U255 ( .I(a[2]), .ZN(n321) );
  VHSR_OAI22_2 U256 ( .A1(n279), .A2(n324), .B1(n278), .B2(n321), .ZN(n240) );
  VHSR_NOR2_1 U257 ( .A1(n278), .A2(n321), .ZN(n227) );
  VHSR_IN_2 U258 ( .I(a[1]), .ZN(n318) );
  VHSR_NOR2_1 U259 ( .A1(n279), .A2(n318), .ZN(n226) );
  VHSR_IN_2 U260 ( .I(b[5]), .ZN(n284) );
  VHSR_AOI211_2 U261 ( .A1(b[4]), .A2(a[2]), .B(n284), .C(n324), .ZN(n232) );
  VHSR_OAI22_2 U262 ( .A1(n279), .A2(n321), .B1(n278), .B2(n318), .ZN(n231) );
  VHSR_AOI22_2 U263 ( .A1(n227), .A2(n226), .B1(n232), .B2(n231), .ZN(n233) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[4]), .A2(a[2]), .ZN(n261) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[4]), .A2(a[0]), .ZN(n391) );
  VHSR_NAND3_2 U266 ( .A1(a[1]), .A2(b[5]), .A3(n391), .ZN(n260) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[0]), .ZN(n259) );
  VHSR_MAOI222_2 U268 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_IN_2 U269 ( .I(b[4]), .ZN(n275) );
  VHSR_NOR4_2 U270 ( .A1(n275), .A2(n284), .A3(n318), .A4(n399), .ZN(n268) );
  VHSR_NAND4_2 U271 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_OAI22_2 U272 ( .A1(n275), .A2(n324), .B1(n284), .B2(n321), .ZN(n228) );
  VHSR_AND2_2 U273 ( .A1(n237), .A2(n228), .Z(n230) );
  VHSR_OAI22_2 U274 ( .A1(n279), .A2(n318), .B1(n278), .B2(n399), .ZN(n229) );
  VHSR_AND2_2 U275 ( .A1(n258), .A2(n255), .Z(n254) );
  VHSR_AD1_1 U276 ( .A(n268), .B(n230), .CI(n229), .CO(n249), .S(n255) );
  VHSR_NOR2_1 U277 ( .A1(n254), .A2(n249), .ZN(n252) );
  VHSR_OAI21_2 U278 ( .A1(n232), .A2(n231), .B(n233), .ZN(n253) );
  VHSR_NOR2_1 U279 ( .A1(n252), .A2(n253), .ZN(n250) );
  VHSR_CLKNAND2_2 U280 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U281 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U282 ( .A1(n278), .A2(n324), .A3(n235), .ZN(n295) );
  VHSR_OAI32_2 U283 ( .A1(n296), .A2(n276), .A3(n319), .B1(n234), .B2(n296), 
        .ZN(n299) );
  VHSR_OAI32_2 U284 ( .A1(n295), .A2(n324), .A3(n278), .B1(n235), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U285 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U286 ( .A1(n240), .A2(n239), .ZN(n306) );
  VHSR_OAI21_2 U287 ( .A1(n243), .A2(n242), .B(n241), .ZN(n245) );
  VHSR_CLKNAND2_2 U288 ( .A1(b[3]), .A2(a[6]), .ZN(n244) );
  VHSR_CLKXOR2_2 U289 ( .A1(n245), .A2(n244), .Z(n305) );
  VHSR_AOI21_2 U290 ( .A1(n248), .A2(n247), .B(n246), .ZN(n311) );
  VHSR_CLKNAND2_2 U291 ( .A1(n254), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U292 ( .A1(n253), .A2(n252), .B1(n251), .B2(n250), .ZN(n310) );
  VHSR_IAO21_2 U293 ( .A1(n258), .A2(n255), .B(n254), .ZN(n314) );
  VHSR_AOI21_2 U294 ( .A1(n262), .A2(n257), .B(n256), .ZN(n313) );
  VHSR_AOI31_2 U295 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n330) );
  VHSR_OAI31_2 U296 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n266) );
  VHSR_IN_2 U297 ( .I(n266), .ZN(n329) );
  VHSR_CLKNAND2_2 U298 ( .A1(b[5]), .A2(a[0]), .ZN(n267) );
  VHSR_OAI32_2 U299 ( .A1(n268), .A2(n318), .A3(n275), .B1(n267), .B2(n268), 
        .ZN(n341) );
  VHSR_CLKNAND2_2 U300 ( .A1(a[4]), .A2(b[1]), .ZN(n269) );
  VHSR_OAI32_2 U301 ( .A1(n270), .A2(n320), .A3(n283), .B1(n269), .B2(n270), 
        .ZN(n340) );
  VHSR_NOR3_2 U302 ( .A1(n280), .A2(n320), .A3(n391), .ZN(n390) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[6]), .A2(b[6]), .ZN(n358) );
  VHSR_IN_2 U304 ( .I(n358), .ZN(n385) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_NAND3_2 U306 ( .A1(a[7]), .A2(b[5]), .A3(n303), .ZN(n272) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[4]), .A2(b[6]), .ZN(n302) );
  VHSR_NAND3_2 U308 ( .A1(b[7]), .A2(a[5]), .A3(n302), .ZN(n271) );
  VHSR_CLKNAND2_2 U309 ( .A1(n272), .A2(n271), .ZN(n274) );
  VHSR_MAOI222_2 U310 ( .A(n358), .B(n272), .C(n271), .ZN(n342) );
  VHSR_IN_2 U311 ( .I(n342), .ZN(n273) );
  VHSR_OAI21_2 U312 ( .A1(n385), .A2(n274), .B(n273), .ZN(n290) );
  VHSR_NOR2_1 U313 ( .A1(n280), .A2(n275), .ZN(n364) );
  VHSR_NOR3_2 U314 ( .A1(n276), .A2(n303), .A3(n284), .ZN(n350) );
  VHSR_AOI22_2 U315 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n277) );
  VHSR_NOR2_1 U316 ( .A1(n350), .A2(n277), .ZN(n286) );
  VHSR_NOR4_2 U317 ( .A1(n283), .A2(n280), .A3(n279), .A4(n278), .ZN(n348) );
  VHSR_AOI22_2 U318 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n281) );
  VHSR_NOR2_1 U319 ( .A1(n348), .A2(n281), .ZN(n285) );
  VHSR_IN_2 U320 ( .I(n282), .ZN(n293) );
  VHSR_OR3_2 U321 ( .A1(n364), .A2(n284), .A3(n283), .Z(n301) );
  VHSR_MAOI222_2 U322 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_IN_2 U323 ( .I(n300), .ZN(n292) );
  VHSR_NOR2_1 U324 ( .A1(n293), .A2(n292), .ZN(n291) );
  VHSR_AD1_1 U325 ( .A(n307), .B(n286), .CI(n285), .CO(n287), .S(n282) );
  VHSR_NOR2_1 U326 ( .A1(n291), .A2(n287), .ZN(n289) );
  VHSR_CLKNAND2_2 U327 ( .A1(n291), .A2(n287), .ZN(n288) );
  VHSR_NOR2_1 U328 ( .A1(n289), .A2(n290), .ZN(n343) );
  VHSR_AOI22_2 U329 ( .A1(n290), .A2(n289), .B1(n288), .B2(n343), .ZN(n383) );
  VHSR_AOI21_2 U330 ( .A1(n293), .A2(n292), .B(n291), .ZN(n381) );
  VHSR_AD1_1 U331 ( .A(n296), .B(n295), .CI(n294), .CO(n384), .S(n380) );
  VHSR_AD1_1 U332 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n378) );
  VHSR_AOI31_2 U333 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n377) );
  VHSR_AD1_1 U334 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n362) );
  VHSR_AOI22_2 U335 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n308) );
  VHSR_NOR2_1 U336 ( .A1(n308), .A2(n307), .ZN(n361) );
  VHSR_AD1_1 U337 ( .A(n311), .B(n310), .CI(n309), .CO(n304), .S(n365) );
  VHSR_AD1_1 U338 ( .A(n314), .B(n313), .CI(n312), .CO(n309), .S(n368) );
  VHSR_NOR2_1 U339 ( .A1(n398), .A2(n321), .ZN(n333) );
  VHSR_AOI22_2 U340 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n315) );
  VHSR_AOI31_2 U341 ( .A1(a[3]), .A2(b[3]), .A3(n333), .B(n315), .ZN(n339) );
  VHSR_NOR2_1 U342 ( .A1(n323), .A2(n324), .ZN(n317) );
  VHSR_NOR2_1 U343 ( .A1(n319), .A2(n318), .ZN(n316) );
  VHSR_MAOI222_2 U344 ( .A(n333), .B(n317), .C(n316), .ZN(n326) );
  VHSR_OAI22_2 U345 ( .A1(n319), .A2(n399), .B1(n398), .B2(n318), .ZN(n372) );
  VHSR_AOI22_2 U346 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n397) );
  VHSR_NOR3_2 U347 ( .A1(n397), .A2(n399), .A3(n398), .ZN(n400) );
  VHSR_OAI22_2 U348 ( .A1(n323), .A2(n321), .B1(n320), .B2(n324), .ZN(n371) );
  VHSR_IN_2 U349 ( .I(n326), .ZN(n325) );
  VHSR_AOI21_2 U350 ( .A1(a[1]), .A2(b[3]), .B(n333), .ZN(n322) );
  VHSR_OAI32_2 U351 ( .A1(n325), .A2(n324), .A3(n323), .B1(n322), .B2(n325), 
        .ZN(n335) );
  VHSR_CLKNAND2_2 U352 ( .A1(n336), .A2(n335), .ZN(n334) );
  VHSR_CLKNAND2_2 U353 ( .A1(n326), .A2(n334), .ZN(n338) );
  VHSR_AND2_2 U354 ( .A1(n339), .A2(n338), .Z(n337) );
  VHSR_OAI211_2 U355 ( .A1(n333), .A2(n337), .B(a[3]), .C(b[3]), .ZN(n327) );
  VHSR_IN_2 U356 ( .I(n327), .ZN(n367) );
  VHSR_AD1_1 U357 ( .A(n330), .B(n329), .CI(n328), .CO(n312), .S(n375) );
  VHSR_CLKNAND2_2 U358 ( .A1(b[3]), .A2(a[3]), .ZN(n332) );
  VHSR_CLKNAND2_2 U359 ( .A1(n337), .A2(n332), .ZN(n331) );
  VHSR_OAI31_2 U360 ( .A1(n333), .A2(n337), .A3(n332), .B(n331), .ZN(n374) );
  VHSR_CLKNAND2_2 U361 ( .A1(a[4]), .A2(b[0]), .ZN(n392) );
  VHSR_OAI21_2 U362 ( .A1(n336), .A2(n335), .B(n334), .ZN(n396) );
  VHSR_AOI211_2 U363 ( .A1(n392), .A2(n391), .B(n390), .C(n396), .ZN(n394) );
  VHSR_IAO21_2 U364 ( .A1(n339), .A2(n338), .B(n337), .ZN(n370) );
  VHSR_AD1_1 U365 ( .A(n341), .B(n340), .CI(n390), .CO(n328), .S(n369) );
  VHSR_NOR2_1 U366 ( .A1(n343), .A2(n342), .ZN(n355) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[7]), .A2(b[6]), .ZN(n345) );
  VHSR_AOI21_2 U368 ( .A1(a[6]), .A2(b[7]), .B(n345), .ZN(n344) );
  VHSR_AOI31_2 U369 ( .A1(a[6]), .A2(n345), .A3(b[7]), .B(n344), .ZN(n346) );
  VHSR_IN_2 U370 ( .I(n346), .ZN(n347) );
  VHSR_MAOI222_2 U371 ( .A(n350), .B(n348), .C(n347), .ZN(n357) );
  VHSR_OAI21_2 U372 ( .A1(n350), .A2(n349), .B(n357), .ZN(n354) );
  VHSR_CLKXOR2_2 U373 ( .A1(n355), .A2(n354), .Z(n351) );
  VHSR_CLKNAND2_2 U374 ( .A1(n352), .A2(n351), .ZN(n387) );
  VHSR_OAI21_2 U375 ( .A1(n352), .A2(n351), .B(n387), .ZN(n353) );
  VHSR_CLKNAND2_2 U376 ( .A1(a[7]), .A2(b[7]), .ZN(n386) );
  VHSR_NOR2_1 U377 ( .A1(n355), .A2(n354), .ZN(n356) );
  VHSR_AND3_2 U378 ( .A1(n388), .A2(n358), .A3(n387), .Z(n359) );
  VHSR_NOR2_1 U379 ( .A1(n386), .A2(n359), .ZN(product[15]) );
  VHSR_AD1_1 U380 ( .A(n375), .B(n374), .CI(n373), .CO(n366), .S(product[6])
         );
  VHSR_AD1_1 U381 ( .A(n378), .B(n377), .CI(n376), .CO(n379), .S(product[10])
         );
  VHSR_AD1_1 U382 ( .A(n381), .B(n380), .CI(n379), .CO(n382), .S(product[11])
         );
  VHSR_AD1_1 U383 ( .A(n384), .B(n383), .CI(n382), .CO(n352), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U384 ( .A1(n386), .A2(n385), .ZN(n389) );
  VHSR_XOR3_2 U385 ( .A1(n389), .A2(n388), .A3(n387), .Z(product[14]) );
  VHSR_AOI21_2 U386 ( .A1(n392), .A2(n391), .B(n390), .ZN(n393) );
  VHSR_IN_2 U387 ( .I(n393), .ZN(n395) );
  VHSR_AOI21_2 U388 ( .A1(n396), .A2(n395), .B(n394), .ZN(product[4]) );
  VHSR_OAI32_2 U389 ( .A1(n400), .A2(n399), .A3(n398), .B1(n397), .B2(n400), 
        .ZN(product[2]) );
endmodule


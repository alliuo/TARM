
module mul8_118 ( a, b, product );
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
         n391, n392, n393, n394, n395, n396, n397, n398, n399, n400, n401,
         n402, n403, n404, n405;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U204 ( .A1(n234), .B1(n253), .ZN(n240) );
  VHSR_NOR2_1 U205 ( .A1(n348), .A2(n347), .ZN(n360) );
  VHSR_INAND2_2 U206 ( .A1(n328), .B1(n339), .ZN(n343) );
  VHSR_NOR2_1 U207 ( .A1(n397), .A2(n396), .ZN(n395) );
  VHSR_INAND3_2 U208 ( .A1(n375), .B1(b[5]), .B2(a[5]), .ZN(n305) );
  VHSR_NOR2_1 U209 ( .A1(n274), .A2(n284), .ZN(n390) );
  VHSR_NOR2_1 U210 ( .A1(n286), .A2(n279), .ZN(n375) );
  VHSR_IN_2 U211 ( .I(n358), .ZN(product[13]) );
  VHSR_INOR2_1 U212 ( .A1(n362), .B1(n361), .ZN(n393) );
  VHSR_NOR2_2 U213 ( .A1(n297), .A2(n296), .ZN(n295) );
  VHSR_AD1_1 U214 ( .A(n382), .B(n381), .CI(n380), .CO(n377), .S(product[6])
         );
  VHSR_AD1_1 U215 ( .A(n376), .B(n375), .CI(n374), .CO(n371), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U216 ( .A(n370), .B(n369), .CI(n368), .CO(n365), .S(product[10])
         );
  VHSR_AD1_1 U217 ( .A(n386), .B(n405), .CI(n385), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U218 ( .A(n399), .B(n384), .CI(n383), .CO(n380), .S(product[5])
         );
  VHSR_AD1_1 U219 ( .A(n379), .B(n378), .CI(n377), .CO(n374), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U220 ( .A(n373), .B(n372), .CI(n371), .CO(n368), .S(product[9])
         );
  VHSR_AD1_1 U221 ( .A(n367), .B(n366), .CI(n365), .CO(n387), .S(product[11])
         );
  VHSR_IN_2 U222 ( .I(b[0]), .ZN(n323) );
  VHSR_IN_2 U223 ( .I(a[1]), .ZN(n321) );
  VHSR_NOR2_1 U224 ( .A1(n323), .A2(n321), .ZN(product[1]) );
  VHSR_AOI22_2 U225 ( .A1(b[2]), .A2(a[7]), .B1(b[3]), .B2(a[6]), .ZN(n247) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n329) );
  VHSR_CLKNAND2_2 U227 ( .A1(b[2]), .A2(a[4]), .ZN(n269) );
  VHSR_IN_2 U228 ( .I(a[5]), .ZN(n285) );
  VHSR_NOR3_2 U229 ( .A1(n329), .A2(n269), .A3(n285), .ZN(n245) );
  VHSR_IN_2 U230 ( .I(a[7]), .ZN(n281) );
  VHSR_CLKNAND2_2 U231 ( .A1(a[6]), .A2(b[1]), .ZN(n223) );
  VHSR_NOR2_1 U232 ( .A1(n281), .A2(n223), .ZN(n214) );
  VHSR_IN_2 U233 ( .I(b[2]), .ZN(n403) );
  VHSR_IN_2 U234 ( .I(a[4]), .ZN(n286) );
  VHSR_OAI211_2 U235 ( .A1(n403), .A2(n286), .B(b[3]), .C(a[5]), .ZN(n215) );
  VHSR_IN_2 U236 ( .I(n215), .ZN(n213) );
  VHSR_IN_2 U237 ( .I(a[6]), .ZN(n274) );
  VHSR_NOR2_1 U238 ( .A1(n403), .A2(n274), .ZN(n216) );
  VHSR_MAOI222_2 U239 ( .A(n214), .B(n213), .C(n216), .ZN(n226) );
  VHSR_IN_2 U240 ( .I(b[1]), .ZN(n325) );
  VHSR_OAI31_2 U241 ( .A1(n274), .A2(n281), .A3(n325), .B(n215), .ZN(n217) );
  VHSR_OAI21_2 U242 ( .A1(n217), .A2(n216), .B(n226), .ZN(n218) );
  VHSR_IN_2 U243 ( .I(n218), .ZN(n250) );
  VHSR_IN_2 U244 ( .I(n223), .ZN(n220) );
  VHSR_NOR4_2 U245 ( .A1(n286), .A2(n285), .A3(n325), .A4(n323), .ZN(n273) );
  VHSR_AOI22_2 U246 ( .A1(b[2]), .A2(a[5]), .B1(b[3]), .B2(a[4]), .ZN(n219) );
  VHSR_NOR2_1 U247 ( .A1(n245), .A2(n219), .ZN(n221) );
  VHSR_MAOI222_2 U248 ( .A(n220), .B(n273), .C(n221), .ZN(n225) );
  VHSR_CLKNAND2_2 U249 ( .A1(a[4]), .A2(b[0]), .ZN(n397) );
  VHSR_NAND3_2 U250 ( .A1(b[1]), .A2(a[5]), .A3(n397), .ZN(n268) );
  VHSR_CLKNAND2_2 U251 ( .A1(a[6]), .A2(b[0]), .ZN(n267) );
  VHSR_MAOI222_2 U252 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_NOR2_1 U253 ( .A1(n273), .A2(n221), .ZN(n224) );
  VHSR_IN_2 U254 ( .I(n225), .ZN(n222) );
  VHSR_AOI21_2 U255 ( .A1(n224), .A2(n223), .B(n222), .ZN(n260) );
  VHSR_CLKNAND2_2 U256 ( .A1(n266), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U257 ( .A1(n225), .A2(n259), .ZN(n249) );
  VHSR_CLKNAND2_2 U258 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_CLKNAND2_2 U259 ( .A1(n226), .A2(n248), .ZN(n244) );
  VHSR_NOR2_1 U260 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_NOR2_1 U261 ( .A1(n247), .A2(n243), .ZN(n236) );
  VHSR_AND3_2 U262 ( .A1(n236), .A2(a[7]), .A3(b[3]), .Z(n300) );
  VHSR_IN_2 U263 ( .I(b[7]), .ZN(n283) );
  VHSR_IN_2 U264 ( .I(a[3]), .ZN(n330) );
  VHSR_IN_2 U265 ( .I(b[6]), .ZN(n284) );
  VHSR_IN_2 U266 ( .I(a[2]), .ZN(n324) );
  VHSR_OAI22_2 U267 ( .A1(n284), .A2(n330), .B1(n283), .B2(n324), .ZN(n242) );
  VHSR_NOR2_1 U268 ( .A1(n283), .A2(n324), .ZN(n228) );
  VHSR_NOR2_1 U269 ( .A1(n284), .A2(n321), .ZN(n227) );
  VHSR_IN_2 U270 ( .I(b[5]), .ZN(n280) );
  VHSR_AOI211_2 U271 ( .A1(b[4]), .A2(a[2]), .B(n280), .C(n330), .ZN(n233) );
  VHSR_OAI22_2 U272 ( .A1(n284), .A2(n324), .B1(n283), .B2(n321), .ZN(n232) );
  VHSR_AOI22_2 U273 ( .A1(n228), .A2(n227), .B1(n233), .B2(n232), .ZN(n234) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[4]), .A2(a[2]), .ZN(n265) );
  VHSR_CLKNAND2_2 U275 ( .A1(b[4]), .A2(a[0]), .ZN(n396) );
  VHSR_NAND3_2 U276 ( .A1(a[1]), .A2(b[5]), .A3(n396), .ZN(n264) );
  VHSR_CLKNAND2_2 U277 ( .A1(b[6]), .A2(a[0]), .ZN(n263) );
  VHSR_MAOI222_2 U278 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_IN_2 U279 ( .I(b[4]), .ZN(n279) );
  VHSR_IN_2 U280 ( .I(a[0]), .ZN(n404) );
  VHSR_NOR4_2 U281 ( .A1(n279), .A2(n280), .A3(n321), .A4(n404), .ZN(n271) );
  VHSR_NAND4_2 U282 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_OAI22_2 U283 ( .A1(n279), .A2(n330), .B1(n280), .B2(n324), .ZN(n229) );
  VHSR_AND2_2 U284 ( .A1(n239), .A2(n229), .Z(n231) );
  VHSR_OAI22_2 U285 ( .A1(n284), .A2(n321), .B1(n283), .B2(n404), .ZN(n230) );
  VHSR_AND2_2 U286 ( .A1(n262), .A2(n258), .Z(n257) );
  VHSR_AD1_1 U287 ( .A(n271), .B(n231), .CI(n230), .CO(n252), .S(n258) );
  VHSR_NOR2_1 U288 ( .A1(n257), .A2(n252), .ZN(n255) );
  VHSR_OAI21_2 U289 ( .A1(n233), .A2(n232), .B(n234), .ZN(n256) );
  VHSR_NOR2_1 U290 ( .A1(n255), .A2(n256), .ZN(n253) );
  VHSR_CLKNAND2_2 U291 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U292 ( .A1(n242), .A2(n238), .ZN(n237) );
  VHSR_NOR3_2 U293 ( .A1(n283), .A2(n330), .A3(n237), .ZN(n299) );
  VHSR_NOR2_1 U294 ( .A1(n281), .A2(n329), .ZN(n235) );
  VHSR_IAO21_2 U295 ( .A1(n236), .A2(n235), .B(n300), .ZN(n303) );
  VHSR_OAI32_2 U296 ( .A1(n299), .A2(n330), .A3(n283), .B1(n237), .B2(n299), 
        .ZN(n302) );
  VHSR_OAI21_2 U297 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U298 ( .A1(n242), .A2(n241), .ZN(n310) );
  VHSR_AOI21_2 U299 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U300 ( .A1(n247), .A2(n246), .ZN(n309) );
  VHSR_OAI21_2 U301 ( .A1(n250), .A2(n249), .B(n248), .ZN(n251) );
  VHSR_IN_2 U302 ( .I(n251), .ZN(n315) );
  VHSR_CLKNAND2_2 U303 ( .A1(n257), .A2(n252), .ZN(n254) );
  VHSR_AOI22_2 U304 ( .A1(n256), .A2(n255), .B1(n254), .B2(n253), .ZN(n314) );
  VHSR_IAO21_2 U305 ( .A1(n262), .A2(n258), .B(n257), .ZN(n318) );
  VHSR_OAI21_2 U306 ( .A1(n266), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U307 ( .I(n261), .ZN(n317) );
  VHSR_AOI31_2 U308 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n334) );
  VHSR_AOI31_2 U309 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n333) );
  VHSR_CLKNAND2_2 U310 ( .A1(b[5]), .A2(a[0]), .ZN(n270) );
  VHSR_OAI32_2 U311 ( .A1(n271), .A2(n321), .A3(n279), .B1(n270), .B2(n271), 
        .ZN(n346) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[5]), .A2(b[0]), .ZN(n272) );
  VHSR_OAI32_2 U313 ( .A1(n273), .A2(n325), .A3(n286), .B1(n272), .B2(n273), 
        .ZN(n345) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[6]), .A2(b[4]), .ZN(n307) );
  VHSR_NAND3_2 U315 ( .A1(a[7]), .A2(b[5]), .A3(n307), .ZN(n276) );
  VHSR_CLKNAND2_2 U316 ( .A1(a[4]), .A2(b[6]), .ZN(n306) );
  VHSR_NAND3_2 U317 ( .A1(b[7]), .A2(a[5]), .A3(n306), .ZN(n275) );
  VHSR_CLKNAND2_2 U318 ( .A1(n276), .A2(n275), .ZN(n278) );
  VHSR_IN_2 U319 ( .I(n390), .ZN(n363) );
  VHSR_MAOI222_2 U320 ( .A(n363), .B(n276), .C(n275), .ZN(n347) );
  VHSR_IN_2 U321 ( .I(n347), .ZN(n277) );
  VHSR_OAI21_2 U322 ( .A1(n390), .A2(n278), .B(n277), .ZN(n294) );
  VHSR_AND3_2 U323 ( .A1(n375), .A2(a[5]), .A3(b[5]), .Z(n311) );
  VHSR_NOR3_2 U324 ( .A1(n281), .A2(n307), .A3(n280), .ZN(n355) );
  VHSR_AOI22_2 U325 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n282) );
  VHSR_NOR2_1 U326 ( .A1(n355), .A2(n282), .ZN(n290) );
  VHSR_NOR4_2 U327 ( .A1(n286), .A2(n285), .A3(n284), .A4(n283), .ZN(n353) );
  VHSR_AOI22_2 U328 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n287) );
  VHSR_NOR2_1 U329 ( .A1(n353), .A2(n287), .ZN(n289) );
  VHSR_IN_2 U330 ( .I(n288), .ZN(n297) );
  VHSR_MAOI222_2 U331 ( .A(n307), .B(n306), .C(n305), .ZN(n304) );
  VHSR_IN_2 U332 ( .I(n304), .ZN(n296) );
  VHSR_AD1_1 U333 ( .A(n311), .B(n290), .CI(n289), .CO(n291), .S(n288) );
  VHSR_NOR2_1 U334 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U335 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_NOR2_1 U336 ( .A1(n293), .A2(n294), .ZN(n348) );
  VHSR_AOI22_2 U337 ( .A1(n294), .A2(n293), .B1(n292), .B2(n348), .ZN(n388) );
  VHSR_AOI21_2 U338 ( .A1(n297), .A2(n296), .B(n295), .ZN(n367) );
  VHSR_AD1_1 U339 ( .A(n300), .B(n299), .CI(n298), .CO(n389), .S(n366) );
  VHSR_AD1_1 U340 ( .A(n303), .B(n302), .CI(n301), .CO(n298), .S(n370) );
  VHSR_AOI31_2 U341 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n369) );
  VHSR_AD1_1 U342 ( .A(n310), .B(n309), .CI(n308), .CO(n301), .S(n373) );
  VHSR_AOI22_2 U343 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n312) );
  VHSR_NOR2_1 U344 ( .A1(n312), .A2(n311), .ZN(n372) );
  VHSR_AD1_1 U345 ( .A(n315), .B(n314), .CI(n313), .CO(n308), .S(n376) );
  VHSR_AD1_1 U346 ( .A(n318), .B(n317), .CI(n316), .CO(n313), .S(n379) );
  VHSR_CLKNAND2_2 U347 ( .A1(b[2]), .A2(a[2]), .ZN(n331) );
  VHSR_IN_2 U348 ( .I(n331), .ZN(n338) );
  VHSR_AOI22_2 U349 ( .A1(b[2]), .A2(a[3]), .B1(b[3]), .B2(a[2]), .ZN(n319) );
  VHSR_AOI31_2 U350 ( .A1(a[3]), .A2(b[3]), .A3(n338), .B(n319), .ZN(n344) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[3]), .A2(a[1]), .ZN(n320) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[1]), .A2(a[3]), .ZN(n326) );
  VHSR_MAOI222_2 U353 ( .A(n331), .B(n320), .C(n326), .ZN(n328) );
  VHSR_OAI22_2 U354 ( .A1(n403), .A2(n321), .B1(n329), .B2(n404), .ZN(n386) );
  VHSR_AOI21_2 U355 ( .A1(n325), .A2(n323), .B(n404), .ZN(product[0]) );
  VHSR_AOI32_2 U356 ( .A1(b[0]), .A2(product[0]), .A3(a[2]), .B1(a[1]), .B2(
        product[0]), .ZN(n322) );
  VHSR_AOI211_2 U357 ( .A1(n325), .A2(n324), .B(n403), .C(n322), .ZN(n405) );
  VHSR_OAI22_2 U358 ( .A1(n325), .A2(n324), .B1(n323), .B2(n330), .ZN(n385) );
  VHSR_AOI21_2 U359 ( .A1(a[1]), .A2(b[3]), .B(n338), .ZN(n327) );
  VHSR_AOI21_2 U360 ( .A1(n327), .A2(n326), .B(n328), .ZN(n340) );
  VHSR_CLKNAND2_2 U361 ( .A1(n341), .A2(n340), .ZN(n339) );
  VHSR_CLKNAND2_2 U362 ( .A1(n344), .A2(n343), .ZN(n335) );
  VHSR_AOI211_2 U363 ( .A1(n331), .A2(n335), .B(n330), .C(n329), .ZN(n378) );
  VHSR_AD1_1 U364 ( .A(n334), .B(n333), .CI(n332), .CO(n316), .S(n382) );
  VHSR_IN_2 U365 ( .I(n335), .ZN(n342) );
  VHSR_CLKNAND2_2 U366 ( .A1(b[3]), .A2(a[3]), .ZN(n337) );
  VHSR_CLKNAND2_2 U367 ( .A1(n342), .A2(n337), .ZN(n336) );
  VHSR_OAI31_2 U368 ( .A1(n338), .A2(n342), .A3(n337), .B(n336), .ZN(n381) );
  VHSR_OAI21_2 U369 ( .A1(n341), .A2(n340), .B(n339), .ZN(n401) );
  VHSR_AOI211_2 U370 ( .A1(n397), .A2(n396), .B(n395), .C(n401), .ZN(n399) );
  VHSR_IAO21_2 U371 ( .A1(n344), .A2(n343), .B(n342), .ZN(n384) );
  VHSR_AD1_1 U372 ( .A(n346), .B(n345), .CI(n395), .CO(n332), .S(n383) );
  VHSR_CLKNAND2_2 U373 ( .A1(a[6]), .A2(b[7]), .ZN(n350) );
  VHSR_AOI21_2 U374 ( .A1(a[7]), .A2(b[6]), .B(n350), .ZN(n349) );
  VHSR_AOI31_2 U375 ( .A1(a[7]), .A2(n350), .A3(b[6]), .B(n349), .ZN(n351) );
  VHSR_IN_2 U376 ( .I(n351), .ZN(n352) );
  VHSR_OR2_2 U377 ( .A1(n353), .A2(n352), .Z(n354) );
  VHSR_MAOI222_2 U378 ( .A(n355), .B(n353), .C(n352), .ZN(n362) );
  VHSR_OAI21_2 U379 ( .A1(n355), .A2(n354), .B(n362), .ZN(n359) );
  VHSR_CLKXOR2_2 U380 ( .A1(n360), .A2(n359), .Z(n356) );
  VHSR_CLKNAND2_2 U381 ( .A1(n357), .A2(n356), .ZN(n392) );
  VHSR_OAI21_2 U382 ( .A1(n357), .A2(n356), .B(n392), .ZN(n358) );
  VHSR_CLKNAND2_2 U383 ( .A1(a[7]), .A2(b[7]), .ZN(n391) );
  VHSR_NOR2_1 U384 ( .A1(n360), .A2(n359), .ZN(n361) );
  VHSR_AND3_2 U385 ( .A1(n393), .A2(n363), .A3(n392), .Z(n364) );
  VHSR_NOR2_1 U386 ( .A1(n391), .A2(n364), .ZN(product[15]) );
  VHSR_AD1_1 U387 ( .A(n389), .B(n388), .CI(n387), .CO(n357), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U388 ( .A1(n391), .A2(n390), .ZN(n394) );
  VHSR_XOR3_2 U389 ( .A1(n394), .A2(n393), .A3(n392), .Z(product[14]) );
  VHSR_AOI21_2 U390 ( .A1(n397), .A2(n396), .B(n395), .ZN(n398) );
  VHSR_IN_2 U391 ( .I(n398), .ZN(n400) );
  VHSR_AOI21_2 U392 ( .A1(n401), .A2(n400), .B(n399), .ZN(product[4]) );
  VHSR_AOI22_2 U393 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n402) );
  VHSR_OAI32_2 U394 ( .A1(n405), .A2(n404), .A3(n403), .B1(n402), .B2(n405), 
        .ZN(product[2]) );
endmodule


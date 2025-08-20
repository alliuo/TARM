
module mul8_85 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n211, n212, n213, n214, n215,
         n216, n217, n218, n219, n220, n221, n222, n223, n224, n225, n226,
         n227, n228, n229, n230, n231, n232, n233, n234, n235, n236, n237,
         n238, n239, n240, n241, n242, n243, n244, n245, n246, n247, n248,
         n249, n250, n251, n252, n253, n254, n255, n256, n257, n258, n259,
         n260, n261, n262, n263, n264, n265, n266, n267, n268, n269, n270,
         n271, n272, n273, n274, n275, n276, n277, n278, n279, n280, n281,
         n282, n283, n284, n285, n286, n287, n288, n289, n290, n291, n292,
         n293, n294, n295, n296, n297, n298, n299, n300, n301, n302, n303,
         n304, n305, n306, n307, n308, n309, n310, n311, n312, n313, n314,
         n315, n316, n317, n318, n319, n320, n321, n322, n323, n324, n325,
         n326, n327, n328, n329, n330, n331, n332, n333, n334, n335, n336,
         n337, n338, n339, n340, n341, n342, n343, n344, n345, n346, n347,
         n348, n349, n350, n351, n352, n353, n354, n355, n356, n357, n358,
         n359, n360, n361, n362, n363, n364, n365, n366, n367, n368, n369,
         n370, n371, n372, n373, n374, n375, n376, n377, n378, n379, n380,
         n381, n382, n383, n384, n385, n386, n387, n388, n389, n390, n391,
         n392, n393, n394, n395, n396, n397, n398, n399, n400, n401;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U203 ( .A1(n256), .B1(n277), .ZN(n264) );
  VHSR_IOA21_2 U204 ( .A1(n255), .A2(n254), .B(n256), .ZN(n280) );
  VHSR_NOR2_1 U205 ( .A1(n330), .A2(n332), .ZN(n343) );
  VHSR_INOR3_2 U206 ( .A1(n259), .B1(n331), .B2(n257), .ZN(n308) );
  VHSR_IOA21_2 U207 ( .A1(n390), .A2(n389), .B(n388), .ZN(n392) );
  VHSR_IN_2 U208 ( .I(n354), .ZN(product[15]) );
  VHSR_MOAI22_1 U209 ( .A1(n332), .A2(n249), .B1(a[3]), .B2(b[4]), .ZN(n250)
         );
  VHSR_NOR2_2 U210 ( .A1(n240), .A2(n251), .ZN(n385) );
  VHSR_AD1_1 U211 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U212 ( .A(n360), .B(n359), .CI(n358), .CO(n355), .S(product[11])
         );
  VHSR_AD1_1 U213 ( .A(n373), .B(n398), .CI(n372), .CO(n346), .S(product[3])
         );
  VHSR_AD1_1 U214 ( .A(n391), .B(n371), .CI(n370), .CO(n374), .S(product[5])
         );
  VHSR_AD1_1 U215 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U216 ( .A(n363), .B(n362), .CI(n361), .CO(n377), .S(product[9])
         );
  VHSR_AD1_1 U217 ( .A(n357), .B(n356), .CI(n355), .CO(n380), .S(product[12])
         );
  VHSR_CLKNAND2_2 U218 ( .A1(a[7]), .A2(b[7]), .ZN(n383) );
  VHSR_IN_2 U219 ( .I(n383), .ZN(n353) );
  VHSR_IN_2 U220 ( .I(a[6]), .ZN(n240) );
  VHSR_IN_2 U221 ( .I(b[6]), .ZN(n251) );
  VHSR_AOI22_2 U222 ( .A1(a[6]), .A2(b[7]), .B1(a[7]), .B2(b[6]), .ZN(n211) );
  VHSR_AOI21_2 U223 ( .A1(n353), .A2(n385), .B(n211), .ZN(n230) );
  VHSR_IN_2 U224 ( .I(b[5]), .ZN(n249) );
  VHSR_IN_2 U225 ( .I(a[7]), .ZN(n257) );
  VHSR_CLKNAND2_2 U226 ( .A1(a[6]), .A2(b[4]), .ZN(n314) );
  VHSR_NAND4_2 U227 ( .A1(a[5]), .A2(a[4]), .A3(b[7]), .A4(b[6]), .ZN(n218) );
  VHSR_OAI31_2 U228 ( .A1(n249), .A2(n257), .A3(n314), .B(n218), .ZN(n229) );
  VHSR_IN_2 U229 ( .I(a[5]), .ZN(n237) );
  VHSR_IN_2 U230 ( .I(b[7]), .ZN(n261) );
  VHSR_AOI211_2 U231 ( .A1(a[4]), .A2(b[6]), .B(n237), .C(n261), .ZN(n213) );
  VHSR_AOI211_2 U232 ( .A1(a[6]), .A2(b[4]), .B(n257), .C(n249), .ZN(n212) );
  VHSR_MAOI222_2 U233 ( .A(n213), .B(n385), .C(n212), .ZN(n227) );
  VHSR_AOI31_2 U234 ( .A1(b[5]), .A2(a[7]), .A3(n314), .B(n385), .ZN(n216) );
  VHSR_IN_2 U235 ( .I(n213), .ZN(n215) );
  VHSR_IN_2 U236 ( .I(n227), .ZN(n214) );
  VHSR_AOI21_2 U237 ( .A1(n216), .A2(n215), .B(n214), .ZN(n301) );
  VHSR_IN_2 U238 ( .I(a[4]), .ZN(n236) );
  VHSR_NOR2_1 U239 ( .A1(n236), .A2(n261), .ZN(n217) );
  VHSR_AOI32_2 U240 ( .A1(n218), .A2(b[6]), .A3(a[5]), .B1(n217), .B2(n218), 
        .ZN(n224) );
  VHSR_IN_2 U241 ( .I(n224), .ZN(n221) );
  VHSR_CLKNAND2_2 U242 ( .A1(a[4]), .A2(b[4]), .ZN(n298) );
  VHSR_NOR3_2 U243 ( .A1(n237), .A2(n249), .A3(n298), .ZN(n320) );
  VHSR_IN_2 U244 ( .I(n314), .ZN(n220) );
  VHSR_AOI22_2 U245 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n219) );
  VHSR_AOI31_2 U246 ( .A1(b[5]), .A2(a[7]), .A3(n220), .B(n219), .ZN(n222) );
  VHSR_MAOI222_2 U247 ( .A(n221), .B(n320), .C(n222), .ZN(n226) );
  VHSR_CLKNAND2_2 U248 ( .A1(a[4]), .A2(b[6]), .ZN(n315) );
  VHSR_NAND3_2 U249 ( .A1(b[5]), .A2(a[5]), .A3(n298), .ZN(n313) );
  VHSR_MAOI222_2 U250 ( .A(n315), .B(n314), .C(n313), .ZN(n312) );
  VHSR_NOR2_1 U251 ( .A1(n320), .A2(n222), .ZN(n225) );
  VHSR_IN_2 U252 ( .I(n226), .ZN(n223) );
  VHSR_AOI21_2 U253 ( .A1(n225), .A2(n224), .B(n223), .ZN(n304) );
  VHSR_CLKNAND2_2 U254 ( .A1(n312), .A2(n304), .ZN(n303) );
  VHSR_CLKNAND2_2 U255 ( .A1(n226), .A2(n303), .ZN(n300) );
  VHSR_CLKNAND2_2 U256 ( .A1(n301), .A2(n300), .ZN(n299) );
  VHSR_CLKNAND2_2 U257 ( .A1(n227), .A2(n299), .ZN(n228) );
  VHSR_AD1_1 U258 ( .A(n230), .B(n229), .CI(n228), .CO(n384), .S(n381) );
  VHSR_AOI22_2 U259 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n271) );
  VHSR_IN_2 U260 ( .I(b[3]), .ZN(n331) );
  VHSR_IN_2 U261 ( .I(b[2]), .ZN(n330) );
  VHSR_NOR4_2 U262 ( .A1(n331), .A2(n330), .A3(n237), .A4(n236), .ZN(n269) );
  VHSR_IN_2 U263 ( .I(b[1]), .ZN(n397) );
  VHSR_NOR2_1 U264 ( .A1(n257), .A2(n397), .ZN(n232) );
  VHSR_NOR2_1 U265 ( .A1(n240), .A2(n330), .ZN(n231) );
  VHSR_AOI211_2 U266 ( .A1(b[2]), .A2(a[4]), .B(n331), .C(n237), .ZN(n233) );
  VHSR_MAOI222_2 U267 ( .A(n232), .B(n231), .C(n233), .ZN(n246) );
  VHSR_OAI22_2 U268 ( .A1(n240), .A2(n330), .B1(n257), .B2(n397), .ZN(n234) );
  VHSR_OAI21_2 U269 ( .A1(n234), .A2(n233), .B(n246), .ZN(n235) );
  VHSR_IN_2 U270 ( .I(n235), .ZN(n274) );
  VHSR_IN_2 U271 ( .I(b[0]), .ZN(n395) );
  VHSR_NOR4_2 U272 ( .A1(n237), .A2(n236), .A3(n397), .A4(n395), .ZN(n297) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[2]), .A2(a[5]), .ZN(n239) );
  VHSR_CLKNAND2_2 U274 ( .A1(b[3]), .A2(a[4]), .ZN(n238) );
  VHSR_AOI21_2 U275 ( .A1(n239), .A2(n238), .B(n269), .ZN(n241) );
  VHSR_OAI22_2 U276 ( .A1(n240), .A2(n397), .B1(n257), .B2(n395), .ZN(n242) );
  VHSR_MAOI222_2 U277 ( .A(n297), .B(n241), .C(n242), .ZN(n245) );
  VHSR_CLKNAND2_2 U278 ( .A1(b[2]), .A2(a[4]), .ZN(n293) );
  VHSR_CLKNAND2_2 U279 ( .A1(a[4]), .A2(b[0]), .ZN(n390) );
  VHSR_NAND3_2 U280 ( .A1(b[1]), .A2(a[5]), .A3(n390), .ZN(n292) );
  VHSR_CLKNAND2_2 U281 ( .A1(a[6]), .A2(b[0]), .ZN(n291) );
  VHSR_MAOI222_2 U282 ( .A(n293), .B(n292), .C(n291), .ZN(n290) );
  VHSR_OR2_2 U283 ( .A1(n297), .A2(n241), .Z(n243) );
  VHSR_OAI21_2 U284 ( .A1(n243), .A2(n242), .B(n245), .ZN(n244) );
  VHSR_IN_2 U285 ( .I(n244), .ZN(n284) );
  VHSR_CLKNAND2_2 U286 ( .A1(n290), .A2(n284), .ZN(n283) );
  VHSR_CLKNAND2_2 U287 ( .A1(n245), .A2(n283), .ZN(n273) );
  VHSR_CLKNAND2_2 U288 ( .A1(n274), .A2(n273), .ZN(n272) );
  VHSR_CLKNAND2_2 U289 ( .A1(n246), .A2(n272), .ZN(n268) );
  VHSR_NOR2_1 U290 ( .A1(n269), .A2(n268), .ZN(n267) );
  VHSR_NOR2_1 U291 ( .A1(n271), .A2(n267), .ZN(n259) );
  VHSR_IN_2 U292 ( .I(a[3]), .ZN(n334) );
  VHSR_IN_2 U293 ( .I(a[2]), .ZN(n332) );
  VHSR_OAI22_2 U294 ( .A1(n332), .A2(n261), .B1(n334), .B2(n251), .ZN(n266) );
  VHSR_NOR2_1 U295 ( .A1(n332), .A2(n261), .ZN(n248) );
  VHSR_IN_2 U296 ( .I(a[1]), .ZN(n394) );
  VHSR_NOR2_1 U297 ( .A1(n251), .A2(n394), .ZN(n247) );
  VHSR_CLKNAND2_2 U298 ( .A1(a[2]), .A2(b[4]), .ZN(n289) );
  VHSR_NAND3_2 U299 ( .A1(a[3]), .A2(b[5]), .A3(n289), .ZN(n254) );
  VHSR_AOI22_2 U300 ( .A1(a[2]), .A2(b[6]), .B1(b[7]), .B2(a[1]), .ZN(n255) );
  VHSR_IAO22_2 U301 ( .B1(n248), .B2(n247), .A1(n254), .A2(n255), .ZN(n256) );
  VHSR_CLKNAND2_2 U302 ( .A1(b[4]), .A2(a[0]), .ZN(n389) );
  VHSR_NAND3_2 U303 ( .A1(a[1]), .A2(b[5]), .A3(n389), .ZN(n288) );
  VHSR_CLKNAND2_2 U304 ( .A1(b[6]), .A2(a[0]), .ZN(n287) );
  VHSR_MAOI222_2 U305 ( .A(n289), .B(n288), .C(n287), .ZN(n286) );
  VHSR_NOR3_2 U306 ( .A1(n249), .A2(n394), .A3(n389), .ZN(n294) );
  VHSR_NAND4_2 U307 ( .A1(a[2]), .A2(a[3]), .A3(b[4]), .A4(b[5]), .ZN(n263) );
  VHSR_AND2_2 U308 ( .A1(n263), .A2(n250), .Z(n253) );
  VHSR_IN_2 U309 ( .I(a[0]), .ZN(n396) );
  VHSR_OAI22_2 U310 ( .A1(n261), .A2(n396), .B1(n251), .B2(n394), .ZN(n252) );
  VHSR_AND2_2 U311 ( .A1(n286), .A2(n282), .Z(n281) );
  VHSR_AD1_1 U312 ( .A(n294), .B(n253), .CI(n252), .CO(n276), .S(n282) );
  VHSR_NOR2_1 U313 ( .A1(n281), .A2(n276), .ZN(n279) );
  VHSR_NOR2_1 U314 ( .A1(n279), .A2(n280), .ZN(n277) );
  VHSR_CLKNAND2_2 U315 ( .A1(n264), .A2(n263), .ZN(n262) );
  VHSR_CLKNAND2_2 U316 ( .A1(n266), .A2(n262), .ZN(n260) );
  VHSR_NOR3_2 U317 ( .A1(n261), .A2(n334), .A3(n260), .ZN(n307) );
  VHSR_NOR2_1 U318 ( .A1(n331), .A2(n257), .ZN(n258) );
  VHSR_IAO21_2 U319 ( .A1(n259), .A2(n258), .B(n308), .ZN(n311) );
  VHSR_OAI32_2 U320 ( .A1(n307), .A2(n334), .A3(n261), .B1(n260), .B2(n307), 
        .ZN(n310) );
  VHSR_OAI21_2 U321 ( .A1(n264), .A2(n263), .B(n262), .ZN(n265) );
  VHSR_XNOR2_2 U322 ( .A1(n266), .A2(n265), .ZN(n318) );
  VHSR_AOI21_2 U323 ( .A1(n269), .A2(n268), .B(n267), .ZN(n270) );
  VHSR_XNOR2_2 U324 ( .A1(n271), .A2(n270), .ZN(n317) );
  VHSR_OAI21_2 U325 ( .A1(n274), .A2(n273), .B(n272), .ZN(n275) );
  VHSR_IN_2 U326 ( .I(n275), .ZN(n323) );
  VHSR_CLKNAND2_2 U327 ( .A1(n281), .A2(n276), .ZN(n278) );
  VHSR_AOI22_2 U328 ( .A1(n280), .A2(n279), .B1(n278), .B2(n277), .ZN(n322) );
  VHSR_IAO21_2 U329 ( .A1(n286), .A2(n282), .B(n281), .ZN(n326) );
  VHSR_OAI21_2 U330 ( .A1(n290), .A2(n284), .B(n283), .ZN(n285) );
  VHSR_IN_2 U331 ( .I(n285), .ZN(n325) );
  VHSR_AOI31_2 U332 ( .A1(n289), .A2(n288), .A3(n287), .B(n286), .ZN(n340) );
  VHSR_AOI31_2 U333 ( .A1(n293), .A2(n292), .A3(n291), .B(n290), .ZN(n339) );
  VHSR_AOI22_2 U334 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n295) );
  VHSR_NOR2_1 U335 ( .A1(n295), .A2(n294), .ZN(n352) );
  VHSR_AOI22_2 U336 ( .A1(a[5]), .A2(b[0]), .B1(a[4]), .B2(b[1]), .ZN(n296) );
  VHSR_NOR2_1 U337 ( .A1(n297), .A2(n296), .ZN(n351) );
  VHSR_IN_2 U338 ( .I(n298), .ZN(n365) );
  VHSR_NOR2_1 U339 ( .A1(n395), .A2(n396), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U340 ( .A1(n365), .A2(product[0]), .ZN(n388) );
  VHSR_IN_2 U341 ( .I(n388), .ZN(n350) );
  VHSR_OAI21_2 U342 ( .A1(n301), .A2(n300), .B(n299), .ZN(n302) );
  VHSR_IN_2 U343 ( .I(n302), .ZN(n356) );
  VHSR_OAI21_2 U344 ( .A1(n312), .A2(n304), .B(n303), .ZN(n305) );
  VHSR_IN_2 U345 ( .I(n305), .ZN(n360) );
  VHSR_AD1_1 U346 ( .A(n308), .B(n307), .CI(n306), .CO(n357), .S(n359) );
  VHSR_AD1_1 U347 ( .A(n311), .B(n310), .CI(n309), .CO(n306), .S(n379) );
  VHSR_AOI31_2 U348 ( .A1(n315), .A2(n314), .A3(n313), .B(n312), .ZN(n378) );
  VHSR_AD1_1 U349 ( .A(n318), .B(n317), .CI(n316), .CO(n309), .S(n363) );
  VHSR_AOI22_2 U350 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n319) );
  VHSR_NOR2_1 U351 ( .A1(n320), .A2(n319), .ZN(n362) );
  VHSR_AD1_1 U352 ( .A(n323), .B(n322), .CI(n321), .CO(n316), .S(n366) );
  VHSR_AD1_1 U353 ( .A(n326), .B(n325), .CI(n324), .CO(n321), .S(n369) );
  VHSR_AOI22_2 U354 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n327) );
  VHSR_AOI31_2 U355 ( .A1(a[3]), .A2(b[3]), .A3(n343), .B(n327), .ZN(n349) );
  VHSR_NOR2_1 U356 ( .A1(n331), .A2(n394), .ZN(n329) );
  VHSR_NOR2_1 U357 ( .A1(n397), .A2(n334), .ZN(n328) );
  VHSR_MAOI222_2 U358 ( .A(n343), .B(n329), .C(n328), .ZN(n336) );
  VHSR_OAI22_2 U359 ( .A1(n331), .A2(n396), .B1(n330), .B2(n394), .ZN(n373) );
  VHSR_CLKNAND2_2 U360 ( .A1(b[0]), .A2(a[2]), .ZN(n401) );
  VHSR_CLKNAND2_2 U361 ( .A1(b[2]), .A2(a[0]), .ZN(n400) );
  VHSR_CLKNAND2_2 U362 ( .A1(b[1]), .A2(a[1]), .ZN(n399) );
  VHSR_MAOI222_2 U363 ( .A(n401), .B(n400), .C(n399), .ZN(n398) );
  VHSR_OAI22_2 U364 ( .A1(n397), .A2(n332), .B1(n395), .B2(n334), .ZN(n372) );
  VHSR_IN_2 U365 ( .I(n336), .ZN(n335) );
  VHSR_AOI21_2 U366 ( .A1(a[1]), .A2(b[3]), .B(n343), .ZN(n333) );
  VHSR_OAI32_2 U367 ( .A1(n335), .A2(n334), .A3(n397), .B1(n333), .B2(n335), 
        .ZN(n345) );
  VHSR_CLKNAND2_2 U368 ( .A1(n346), .A2(n345), .ZN(n344) );
  VHSR_CLKNAND2_2 U369 ( .A1(n336), .A2(n344), .ZN(n348) );
  VHSR_AND2_2 U370 ( .A1(n349), .A2(n348), .Z(n347) );
  VHSR_OAI211_2 U371 ( .A1(n343), .A2(n347), .B(a[3]), .C(b[3]), .ZN(n337) );
  VHSR_IN_2 U372 ( .I(n337), .ZN(n368) );
  VHSR_AD1_1 U373 ( .A(n340), .B(n339), .CI(n338), .CO(n324), .S(n376) );
  VHSR_CLKNAND2_2 U374 ( .A1(b[3]), .A2(a[3]), .ZN(n342) );
  VHSR_CLKNAND2_2 U375 ( .A1(n347), .A2(n342), .ZN(n341) );
  VHSR_OAI31_2 U376 ( .A1(n343), .A2(n347), .A3(n342), .B(n341), .ZN(n375) );
  VHSR_OAI21_2 U377 ( .A1(n346), .A2(n345), .B(n344), .ZN(n393) );
  VHSR_AOI211_2 U378 ( .A1(n390), .A2(n389), .B(n350), .C(n393), .ZN(n391) );
  VHSR_IAO21_2 U379 ( .A1(n349), .A2(n348), .B(n347), .ZN(n371) );
  VHSR_AD1_1 U380 ( .A(n352), .B(n351), .CI(n350), .CO(n338), .S(n370) );
  VHSR_AND2_2 U381 ( .A1(n381), .A2(n380), .Z(n387) );
  VHSR_OAI31_2 U382 ( .A1(n384), .A2(n387), .A3(n385), .B(n353), .ZN(n354) );
  VHSR_AD1_1 U383 ( .A(n376), .B(n375), .CI(n374), .CO(n367), .S(product[6])
         );
  VHSR_AD1_1 U384 ( .A(n379), .B(n378), .CI(n377), .CO(n358), .S(product[10])
         );
  VHSR_IAO21_2 U385 ( .A1(n381), .A2(n380), .B(n387), .ZN(product[13]) );
  VHSR_OAI21_2 U386 ( .A1(n385), .A2(n383), .B(n384), .ZN(n382) );
  VHSR_OAI31_2 U387 ( .A1(n385), .A2(n384), .A3(n383), .B(n382), .ZN(n386) );
  VHSR_CLKXOR2_2 U388 ( .A1(n387), .A2(n386), .Z(product[14]) );
  VHSR_AOI21_2 U389 ( .A1(n393), .A2(n392), .B(n391), .ZN(product[4]) );
  VHSR_OAI22_2 U390 ( .A1(n397), .A2(n396), .B1(n395), .B2(n394), .ZN(
        product[1]) );
  VHSR_AOI31_2 U391 ( .A1(n401), .A2(n400), .A3(n399), .B(n398), .ZN(
        product[2]) );
endmodule


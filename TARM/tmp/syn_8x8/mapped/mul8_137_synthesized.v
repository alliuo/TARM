
module mul8_137 ( a, b, product );
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
         n391, n392, n393, n394, n395, n396;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U204 ( .A1(n239), .B1(n215), .ZN(n217) );
  VHSR_NOR2_1 U205 ( .A1(n394), .A2(n278), .ZN(n261) );
  VHSR_INOR2_2 U206 ( .A1(n221), .B1(n247), .ZN(n240) );
  VHSR_INOR2_2 U207 ( .A1(n219), .B1(n250), .ZN(n249) );
  VHSR_INAND2_2 U208 ( .A1(n313), .B1(n332), .ZN(n328) );
  VHSR_NOR2_1 U209 ( .A1(n231), .A2(n230), .ZN(n291) );
  VHSR_IOA21_2 U210 ( .A1(n386), .A2(n385), .B(n384), .ZN(n389) );
  VHSR_INOR2_2 U211 ( .A1(n351), .B1(n350), .ZN(n382) );
  VHSR_IN_2 U212 ( .I(n347), .ZN(product[13]) );
  VHSR_MOAI22_1 U213 ( .A1(n272), .A2(n311), .B1(b[4]), .B2(a[3]), .ZN(n223)
         );
  VHSR_AD1_1 U214 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[6])
         );
  VHSR_AD1_1 U215 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(product[9])
         );
  VHSR_AD1_1 U216 ( .A(n372), .B(n396), .CI(n371), .CO(n334), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n370), .B(n369), .CI(n387), .CO(n366), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U219 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U220 ( .A(n356), .B(n355), .CI(n354), .CO(n373), .S(product[10])
         );
  VHSR_CLKNAND2_2 U221 ( .A1(b[3]), .A2(a[7]), .ZN(n231) );
  VHSR_IN_2 U222 ( .I(b[3]), .ZN(n314) );
  VHSR_IN_2 U223 ( .I(a[6]), .ZN(n267) );
  VHSR_IN_2 U224 ( .I(a[7]), .ZN(n273) );
  VHSR_IN_2 U225 ( .I(b[2]), .ZN(n394) );
  VHSR_OAI22_2 U226 ( .A1(n314), .A2(n267), .B1(n273), .B2(n394), .ZN(n242) );
  VHSR_IN_2 U227 ( .I(a[4]), .ZN(n278) );
  VHSR_CLKNAND2_2 U228 ( .A1(b[3]), .A2(a[5]), .ZN(n213) );
  VHSR_IN_2 U229 ( .I(b[1]), .ZN(n392) );
  VHSR_OAI22_2 U230 ( .A1(n261), .A2(n213), .B1(n273), .B2(n392), .ZN(n220) );
  VHSR_IN_2 U231 ( .I(a[5]), .ZN(n277) );
  VHSR_NOR4_2 U232 ( .A1(n261), .A2(n231), .A3(n277), .A4(n392), .ZN(n214) );
  VHSR_AOI31_2 U233 ( .A1(a[6]), .A2(b[2]), .A3(n220), .B(n214), .ZN(n221) );
  VHSR_NOR2_1 U234 ( .A1(n267), .A2(n392), .ZN(n216) );
  VHSR_IN_2 U235 ( .I(b[0]), .ZN(n391) );
  VHSR_NOR4_2 U236 ( .A1(n278), .A2(n277), .A3(n392), .A4(n391), .ZN(n266) );
  VHSR_NAND3_2 U237 ( .A1(b[3]), .A2(n261), .A3(a[5]), .ZN(n239) );
  VHSR_AOI22_2 U238 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n215) );
  VHSR_MAOI222_2 U239 ( .A(n216), .B(n266), .C(n217), .ZN(n219) );
  VHSR_AOI211_2 U240 ( .A1(a[4]), .A2(b[0]), .B(n277), .C(n392), .ZN(n260) );
  VHSR_AOI21_2 U241 ( .A1(n273), .A2(n267), .B(n391), .ZN(n259) );
  VHSR_MAOI222_2 U242 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_OR2_2 U243 ( .A1(n266), .A2(n217), .Z(n218) );
  VHSR_AOI32_2 U244 ( .A1(b[1]), .A2(n219), .A3(a[6]), .B1(n218), .B2(n219), 
        .ZN(n251) );
  VHSR_NOR2_1 U245 ( .A1(n258), .A2(n251), .ZN(n250) );
  VHSR_AOI32_2 U246 ( .A1(a[6]), .A2(n221), .A3(b[2]), .B1(n220), .B2(n221), 
        .ZN(n248) );
  VHSR_NOR2_1 U247 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U248 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U249 ( .A1(n242), .A2(n238), .ZN(n230) );
  VHSR_IN_2 U250 ( .I(b[7]), .ZN(n275) );
  VHSR_IN_2 U251 ( .I(a[3]), .ZN(n315) );
  VHSR_IN_2 U252 ( .I(b[6]), .ZN(n276) );
  VHSR_IN_2 U253 ( .I(a[2]), .ZN(n311) );
  VHSR_OAI22_2 U254 ( .A1(n276), .A2(n315), .B1(n275), .B2(n311), .ZN(n237) );
  VHSR_AOI22_2 U255 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n228) );
  VHSR_CLKNAND2_2 U256 ( .A1(b[4]), .A2(a[2]), .ZN(n257) );
  VHSR_NAND3_2 U257 ( .A1(a[3]), .A2(b[5]), .A3(n257), .ZN(n227) );
  VHSR_CLKNAND2_2 U258 ( .A1(b[7]), .A2(a[2]), .ZN(n222) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[6]), .A2(a[1]), .ZN(n224) );
  VHSR_OAI22_2 U260 ( .A1(n228), .A2(n227), .B1(n222), .B2(n224), .ZN(n229) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[4]), .A2(a[0]), .ZN(n385) );
  VHSR_NAND3_2 U262 ( .A1(a[1]), .A2(b[5]), .A3(n385), .ZN(n256) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[6]), .A2(a[0]), .ZN(n255) );
  VHSR_MAOI222_2 U264 ( .A(n257), .B(n256), .C(n255), .ZN(n254) );
  VHSR_NAND4_2 U265 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n234) );
  VHSR_IN_2 U266 ( .I(b[5]), .ZN(n272) );
  VHSR_AND2_2 U267 ( .A1(n234), .A2(n223), .Z(n226) );
  VHSR_IN_2 U268 ( .I(a[0]), .ZN(n395) );
  VHSR_OAI21_2 U269 ( .A1(n275), .A2(n395), .B(n224), .ZN(n225) );
  VHSR_IN_2 U270 ( .I(a[1]), .ZN(n390) );
  VHSR_NOR3_2 U271 ( .A1(n272), .A2(n390), .A3(n385), .ZN(n264) );
  VHSR_AND2_2 U272 ( .A1(n254), .A2(n253), .Z(n252) );
  VHSR_AD1_1 U273 ( .A(n226), .B(n225), .CI(n264), .CO(n243), .S(n253) );
  VHSR_AOI21_2 U274 ( .A1(n228), .A2(n227), .B(n229), .ZN(n246) );
  VHSR_OAI32_2 U275 ( .A1(n229), .A2(n252), .A3(n243), .B1(n246), .B2(n229), 
        .ZN(n235) );
  VHSR_CLKNAND2_2 U276 ( .A1(n235), .A2(n234), .ZN(n233) );
  VHSR_CLKNAND2_2 U277 ( .A1(n237), .A2(n233), .ZN(n232) );
  VHSR_NOR3_2 U278 ( .A1(n275), .A2(n315), .A3(n232), .ZN(n290) );
  VHSR_AOI21_2 U279 ( .A1(n231), .A2(n230), .B(n291), .ZN(n294) );
  VHSR_OAI32_2 U280 ( .A1(n290), .A2(n315), .A3(n275), .B1(n232), .B2(n290), 
        .ZN(n293) );
  VHSR_OAI21_2 U281 ( .A1(n235), .A2(n234), .B(n233), .ZN(n236) );
  VHSR_XNOR2_2 U282 ( .A1(n237), .A2(n236), .ZN(n301) );
  VHSR_OAI21_2 U283 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U284 ( .A1(n242), .A2(n241), .ZN(n300) );
  VHSR_NOR2_1 U285 ( .A1(n252), .A2(n243), .ZN(n245) );
  VHSR_AOI22_2 U286 ( .A1(n252), .A2(n243), .B1(n246), .B2(n245), .ZN(n244) );
  VHSR_OAI21_2 U287 ( .A1(n246), .A2(n245), .B(n244), .ZN(n306) );
  VHSR_AOI21_2 U288 ( .A1(n249), .A2(n248), .B(n247), .ZN(n305) );
  VHSR_AOI21_2 U289 ( .A1(n258), .A2(n251), .B(n250), .ZN(n318) );
  VHSR_IAO21_2 U290 ( .A1(n254), .A2(n253), .B(n252), .ZN(n317) );
  VHSR_AOI31_2 U291 ( .A1(n257), .A2(n256), .A3(n255), .B(n254), .ZN(n326) );
  VHSR_OAI31_2 U292 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n262) );
  VHSR_IN_2 U293 ( .I(n262), .ZN(n325) );
  VHSR_AOI22_2 U294 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n263) );
  VHSR_NOR2_1 U295 ( .A1(n264), .A2(n263), .ZN(n331) );
  VHSR_CLKNAND2_2 U296 ( .A1(a[4]), .A2(b[4]), .ZN(n280) );
  VHSR_IN_2 U297 ( .I(n280), .ZN(n361) );
  VHSR_NOR2_1 U298 ( .A1(n391), .A2(n395), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U299 ( .A1(n361), .A2(product[0]), .ZN(n384) );
  VHSR_IN_2 U300 ( .I(n384), .ZN(n335) );
  VHSR_CLKNAND2_2 U301 ( .A1(a[5]), .A2(b[0]), .ZN(n265) );
  VHSR_OAI32_2 U302 ( .A1(n266), .A2(n392), .A3(n278), .B1(n265), .B2(n266), 
        .ZN(n330) );
  VHSR_NOR2_1 U303 ( .A1(n267), .A2(n276), .ZN(n379) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[6]), .ZN(n297) );
  VHSR_NAND3_2 U305 ( .A1(b[7]), .A2(a[5]), .A3(n297), .ZN(n269) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[6]), .A2(b[4]), .ZN(n298) );
  VHSR_NAND3_2 U307 ( .A1(a[7]), .A2(b[5]), .A3(n298), .ZN(n268) );
  VHSR_CLKNAND2_2 U308 ( .A1(n269), .A2(n268), .ZN(n271) );
  VHSR_IN_2 U309 ( .I(n379), .ZN(n352) );
  VHSR_MAOI222_2 U310 ( .A(n352), .B(n269), .C(n268), .ZN(n336) );
  VHSR_IN_2 U311 ( .I(n336), .ZN(n270) );
  VHSR_OAI21_2 U312 ( .A1(n379), .A2(n271), .B(n270), .ZN(n286) );
  VHSR_NOR3_2 U313 ( .A1(n277), .A2(n272), .A3(n280), .ZN(n302) );
  VHSR_NOR3_2 U314 ( .A1(n273), .A2(n298), .A3(n272), .ZN(n344) );
  VHSR_AOI22_2 U315 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n274) );
  VHSR_NOR2_1 U316 ( .A1(n344), .A2(n274), .ZN(n282) );
  VHSR_NOR4_2 U317 ( .A1(n278), .A2(n277), .A3(n276), .A4(n275), .ZN(n342) );
  VHSR_AOI22_2 U318 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n279) );
  VHSR_NOR2_1 U319 ( .A1(n342), .A2(n279), .ZN(n281) );
  VHSR_NAND3_2 U320 ( .A1(b[5]), .A2(a[5]), .A3(n280), .ZN(n296) );
  VHSR_MAOI222_2 U321 ( .A(n298), .B(n297), .C(n296), .ZN(n295) );
  VHSR_AND2_2 U322 ( .A1(n288), .A2(n295), .Z(n287) );
  VHSR_AD1_1 U323 ( .A(n302), .B(n282), .CI(n281), .CO(n283), .S(n288) );
  VHSR_NOR2_1 U324 ( .A1(n287), .A2(n283), .ZN(n285) );
  VHSR_CLKNAND2_2 U325 ( .A1(n287), .A2(n283), .ZN(n284) );
  VHSR_NOR2_1 U326 ( .A1(n285), .A2(n286), .ZN(n337) );
  VHSR_AOI22_2 U327 ( .A1(n286), .A2(n285), .B1(n284), .B2(n337), .ZN(n377) );
  VHSR_IAO21_2 U328 ( .A1(n288), .A2(n295), .B(n287), .ZN(n375) );
  VHSR_AD1_1 U329 ( .A(n291), .B(n290), .CI(n289), .CO(n378), .S(n374) );
  VHSR_AD1_1 U330 ( .A(n294), .B(n293), .CI(n292), .CO(n289), .S(n356) );
  VHSR_AOI31_2 U331 ( .A1(n298), .A2(n297), .A3(n296), .B(n295), .ZN(n355) );
  VHSR_AD1_1 U332 ( .A(n301), .B(n300), .CI(n299), .CO(n292), .S(n359) );
  VHSR_AOI22_2 U333 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n303) );
  VHSR_NOR2_1 U334 ( .A1(n303), .A2(n302), .ZN(n358) );
  VHSR_AD1_1 U335 ( .A(n306), .B(n305), .CI(n304), .CO(n299), .S(n362) );
  VHSR_CLKNAND2_2 U336 ( .A1(b[2]), .A2(a[2]), .ZN(n319) );
  VHSR_NOR2_1 U337 ( .A1(n394), .A2(n315), .ZN(n308) );
  VHSR_OAI21_2 U338 ( .A1(n314), .A2(n311), .B(n308), .ZN(n307) );
  VHSR_OAI31_2 U339 ( .A1(n314), .A2(n308), .A3(n311), .B(n307), .ZN(n329) );
  VHSR_AOI22_2 U340 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n312) );
  VHSR_CLKNAND2_2 U341 ( .A1(b[3]), .A2(a[3]), .ZN(n322) );
  VHSR_NOR2_1 U342 ( .A1(n392), .A2(n390), .ZN(n310) );
  VHSR_IN_2 U343 ( .I(n310), .ZN(n309) );
  VHSR_OAI22_2 U344 ( .A1(n319), .A2(n312), .B1(n322), .B2(n309), .ZN(n313) );
  VHSR_OAI22_2 U345 ( .A1(n314), .A2(n395), .B1(n394), .B2(n390), .ZN(n372) );
  VHSR_AOI21_2 U346 ( .A1(a[2]), .A2(b[0]), .B(n310), .ZN(n393) );
  VHSR_NOR3_2 U347 ( .A1(n393), .A2(n395), .A3(n394), .ZN(n396) );
  VHSR_OAI22_2 U348 ( .A1(n392), .A2(n311), .B1(n391), .B2(n315), .ZN(n371) );
  VHSR_AOI21_2 U349 ( .A1(n312), .A2(n319), .B(n313), .ZN(n333) );
  VHSR_CLKNAND2_2 U350 ( .A1(n334), .A2(n333), .ZN(n332) );
  VHSR_CLKNAND2_2 U351 ( .A1(n329), .A2(n328), .ZN(n320) );
  VHSR_AOI211_2 U352 ( .A1(n319), .A2(n320), .B(n315), .C(n314), .ZN(n365) );
  VHSR_AD1_1 U353 ( .A(n318), .B(n317), .CI(n316), .CO(n304), .S(n364) );
  VHSR_IN_2 U354 ( .I(n319), .ZN(n323) );
  VHSR_IN_2 U355 ( .I(n320), .ZN(n327) );
  VHSR_CLKNAND2_2 U356 ( .A1(n327), .A2(n322), .ZN(n321) );
  VHSR_OAI31_2 U357 ( .A1(n323), .A2(n327), .A3(n322), .B(n321), .ZN(n368) );
  VHSR_AD1_1 U358 ( .A(n326), .B(n325), .CI(n324), .CO(n316), .S(n367) );
  VHSR_IAO21_2 U359 ( .A1(n329), .A2(n328), .B(n327), .ZN(n370) );
  VHSR_AD1_1 U360 ( .A(n331), .B(n335), .CI(n330), .CO(n324), .S(n369) );
  VHSR_CLKNAND2_2 U361 ( .A1(a[4]), .A2(b[0]), .ZN(n386) );
  VHSR_OAI21_2 U362 ( .A1(n334), .A2(n333), .B(n332), .ZN(n388) );
  VHSR_AOI211_2 U363 ( .A1(n386), .A2(n385), .B(n335), .C(n388), .ZN(n387) );
  VHSR_NOR2_1 U364 ( .A1(n337), .A2(n336), .ZN(n349) );
  VHSR_CLKNAND2_2 U365 ( .A1(a[6]), .A2(b[7]), .ZN(n339) );
  VHSR_AOI21_2 U366 ( .A1(a[7]), .A2(b[6]), .B(n339), .ZN(n338) );
  VHSR_AOI31_2 U367 ( .A1(a[7]), .A2(n339), .A3(b[6]), .B(n338), .ZN(n340) );
  VHSR_IN_2 U368 ( .I(n340), .ZN(n341) );
  VHSR_OR2_2 U369 ( .A1(n342), .A2(n341), .Z(n343) );
  VHSR_MAOI222_2 U370 ( .A(n344), .B(n342), .C(n341), .ZN(n351) );
  VHSR_OAI21_2 U371 ( .A1(n344), .A2(n343), .B(n351), .ZN(n348) );
  VHSR_CLKXOR2_2 U372 ( .A1(n349), .A2(n348), .Z(n345) );
  VHSR_CLKNAND2_2 U373 ( .A1(n346), .A2(n345), .ZN(n381) );
  VHSR_OAI21_2 U374 ( .A1(n346), .A2(n345), .B(n381), .ZN(n347) );
  VHSR_CLKNAND2_2 U375 ( .A1(a[7]), .A2(b[7]), .ZN(n380) );
  VHSR_NOR2_1 U376 ( .A1(n349), .A2(n348), .ZN(n350) );
  VHSR_AND3_2 U377 ( .A1(n382), .A2(n352), .A3(n381), .Z(n353) );
  VHSR_NOR2_1 U378 ( .A1(n380), .A2(n353), .ZN(product[15]) );
  VHSR_AD1_1 U379 ( .A(n375), .B(n374), .CI(n373), .CO(n376), .S(product[11])
         );
  VHSR_AD1_1 U380 ( .A(n378), .B(n377), .CI(n376), .CO(n346), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U381 ( .A1(n380), .A2(n379), .ZN(n383) );
  VHSR_XOR3_2 U382 ( .A1(n383), .A2(n382), .A3(n381), .Z(product[14]) );
  VHSR_AOI21_2 U383 ( .A1(n389), .A2(n388), .B(n387), .ZN(product[4]) );
  VHSR_OAI22_2 U384 ( .A1(n392), .A2(n395), .B1(n391), .B2(n390), .ZN(
        product[1]) );
  VHSR_OAI32_2 U385 ( .A1(n396), .A2(n395), .A3(n394), .B1(n393), .B2(n396), 
        .ZN(product[2]) );
endmodule


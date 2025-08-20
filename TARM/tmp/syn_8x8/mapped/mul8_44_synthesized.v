
module mul8_44 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \mul_ll_ll/out[0] , \intadd_0/SUM[7] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n210, n211, n212, n213, n214, n215, n216, n217,
         n218, n219, n220, n221, n222, n223, n224, n225, n226, n227, n228,
         n229, n230, n231, n232, n233, n234, n235, n236, n237, n238, n239,
         n240, n241, n242, n243, n244, n245, n246, n247, n248, n249, n250,
         n251, n252, n253, n254, n255, n256, n257, n258, n259, n260, n261,
         n262, n263, n264, n265, n266, n267, n268, n269, n270, n271, n272,
         n273, n274, n275, n276, n277, n278, n279, n280, n281, n282, n283,
         n284, n285, n286, n287, n288, n289, n290, n291, n292, n293, n294,
         n295, n296, n297, n298, n299, n300, n301, n302, n303, n304, n305,
         n306, n307, n308, n309, n310, n311, n312, n313, n314, n315, n316,
         n317, n318, n319, n320, n321, n322, n323, n324, n325, n326, n327,
         n328, n329, n330, n331, n332, n333, n334, n335, n336, n337, n338,
         n339, n340, n341, n342, n343, n344, n345, n346, n347, n348, n349,
         n350, n351, n352, n353, n354, n355, n356, n357, n358, n359, n360,
         n361, n362, n363, n364, n365, n366, n367, n368, n369, n370, n371,
         n372, n373, n374, n375, n376, n377, n378, n379, n380, n381, n382,
         n383, n384, n385, n386, n387, n388, n389, n390, n391, n392, n393,
         n394;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_NOR2_1 U201 ( .A1(n274), .A2(n393), .ZN(n261) );
  VHSR_INAND2_2 U202 ( .A1(n342), .B1(n340), .ZN(n343) );
  VHSR_INOR2_2 U203 ( .A1(n218), .B1(n252), .ZN(n249) );
  VHSR_INAND2_2 U204 ( .A1(n316), .B1(n333), .ZN(n329) );
  VHSR_INOR3_2 U205 ( .A1(n358), .B1(n278), .B2(n279), .ZN(n301) );
  VHSR_NOR2_1 U206 ( .A1(n232), .A2(n272), .ZN(n290) );
  VHSR_INOR2_2 U207 ( .A1(n351), .B1(n350), .ZN(n382) );
  VHSR_IN_2 U208 ( .I(n347), .ZN(product[13]) );
  VHSR_AD1_1 U209 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(product[6])
         );
  VHSR_AD1_1 U210 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U211 ( .A(n369), .B(n394), .CI(n368), .CO(n335), .S(product[3])
         );
  VHSR_AD1_1 U212 ( .A(n367), .B(n366), .CI(n388), .CO(n363), .S(product[5])
         );
  VHSR_AD1_1 U213 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U214 ( .A(n356), .B(n355), .CI(n354), .CO(n370), .S(product[9])
         );
  VHSR_PULL0_0 U215 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U216 ( .I(a[0]), .ZN(n392) );
  VHSR_IN_2 U217 ( .I(b[1]), .ZN(n313) );
  VHSR_NOR2_1 U218 ( .A1(n392), .A2(n313), .ZN(product[1]) );
  VHSR_IN_2 U219 ( .I(a[5]), .ZN(n279) );
  VHSR_IN_2 U220 ( .I(b[3]), .ZN(n307) );
  VHSR_IN_2 U221 ( .I(a[4]), .ZN(n274) );
  VHSR_IN_2 U222 ( .I(b[2]), .ZN(n393) );
  VHSR_NOR3_2 U223 ( .A1(n279), .A2(n307), .A3(n261), .ZN(n222) );
  VHSR_IN_2 U224 ( .I(n222), .ZN(n212) );
  VHSR_CLKNAND2_2 U225 ( .A1(a[6]), .A2(b[2]), .ZN(n211) );
  VHSR_CLKNAND2_2 U226 ( .A1(a[7]), .A2(b[1]), .ZN(n210) );
  VHSR_MAOI222_2 U227 ( .A(n212), .B(n211), .C(n210), .ZN(n223) );
  VHSR_IN_2 U228 ( .I(a[6]), .ZN(n219) );
  VHSR_NOR2_1 U229 ( .A1(n219), .A2(n313), .ZN(n215) );
  VHSR_CLKNAND2_2 U230 ( .A1(a[4]), .A2(b[0]), .ZN(n385) );
  VHSR_NOR3_2 U231 ( .A1(n279), .A2(n313), .A3(n385), .ZN(n266) );
  VHSR_NOR2_1 U232 ( .A1(n279), .A2(n393), .ZN(n214) );
  VHSR_OAI21_2 U233 ( .A1(n274), .A2(n307), .B(n214), .ZN(n213) );
  VHSR_OAI31_2 U234 ( .A1(n274), .A2(n214), .A3(n307), .B(n213), .ZN(n216) );
  VHSR_MAOI222_2 U235 ( .A(n215), .B(n266), .C(n216), .ZN(n218) );
  VHSR_AOI211_2 U236 ( .A1(a[4]), .A2(b[0]), .B(n279), .C(n313), .ZN(n260) );
  VHSR_IN_2 U237 ( .I(a[7]), .ZN(n272) );
  VHSR_IN_2 U238 ( .I(b[0]), .ZN(n311) );
  VHSR_AOI21_2 U239 ( .A1(n219), .A2(n272), .B(n311), .ZN(n259) );
  VHSR_MAOI222_2 U240 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_OR2_2 U241 ( .A1(n266), .A2(n216), .Z(n217) );
  VHSR_AOI32_2 U242 ( .A1(b[1]), .A2(n218), .A3(a[6]), .B1(n217), .B2(n218), 
        .ZN(n253) );
  VHSR_NOR2_1 U243 ( .A1(n258), .A2(n253), .ZN(n252) );
  VHSR_OAI22_2 U244 ( .A1(n219), .A2(n393), .B1(n272), .B2(n313), .ZN(n221) );
  VHSR_IN_2 U245 ( .I(n223), .ZN(n220) );
  VHSR_OAI21_2 U246 ( .A1(n222), .A2(n221), .B(n220), .ZN(n248) );
  VHSR_NOR2_1 U247 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_NOR2_1 U248 ( .A1(n223), .A2(n247), .ZN(n241) );
  VHSR_NAND3_2 U249 ( .A1(a[5]), .A2(n261), .A3(b[3]), .ZN(n240) );
  VHSR_CLKNAND2_2 U250 ( .A1(a[6]), .A2(b[3]), .ZN(n239) );
  VHSR_IN_2 U251 ( .I(b[7]), .ZN(n275) );
  VHSR_IN_2 U252 ( .I(a[3]), .ZN(n312) );
  VHSR_IN_2 U253 ( .I(b[6]), .ZN(n276) );
  VHSR_IN_2 U254 ( .I(a[2]), .ZN(n314) );
  VHSR_OAI22_2 U255 ( .A1(n276), .A2(n312), .B1(n275), .B2(n314), .ZN(n238) );
  VHSR_AOI22_2 U256 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n230) );
  VHSR_CLKNAND2_2 U257 ( .A1(b[4]), .A2(a[2]), .ZN(n257) );
  VHSR_NAND3_2 U258 ( .A1(a[3]), .A2(b[5]), .A3(n257), .ZN(n229) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[7]), .A2(a[2]), .ZN(n224) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[6]), .A2(a[1]), .ZN(n226) );
  VHSR_OAI22_2 U261 ( .A1(n230), .A2(n229), .B1(n224), .B2(n226), .ZN(n231) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[6]), .A2(a[0]), .ZN(n256) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[4]), .A2(a[0]), .ZN(n386) );
  VHSR_NAND3_2 U264 ( .A1(a[1]), .A2(b[5]), .A3(n386), .ZN(n255) );
  VHSR_MAOI222_2 U265 ( .A(n257), .B(n256), .C(n255), .ZN(n254) );
  VHSR_IN_2 U266 ( .I(b[5]), .ZN(n278) );
  VHSR_IN_2 U267 ( .I(a[1]), .ZN(n308) );
  VHSR_NOR3_2 U268 ( .A1(n278), .A2(n308), .A3(n386), .ZN(n263) );
  VHSR_NAND4_2 U269 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n235) );
  VHSR_IN_2 U270 ( .I(b[4]), .ZN(n271) );
  VHSR_OAI22_2 U271 ( .A1(n271), .A2(n312), .B1(n278), .B2(n314), .ZN(n225) );
  VHSR_AND2_2 U272 ( .A1(n235), .A2(n225), .Z(n228) );
  VHSR_OAI21_2 U273 ( .A1(n275), .A2(n392), .B(n226), .ZN(n227) );
  VHSR_AND2_2 U274 ( .A1(n254), .A2(n251), .Z(n250) );
  VHSR_AD1_1 U275 ( .A(n263), .B(n228), .CI(n227), .CO(n243), .S(n251) );
  VHSR_AOI21_2 U276 ( .A1(n230), .A2(n229), .B(n231), .ZN(n246) );
  VHSR_OAI32_2 U277 ( .A1(n231), .A2(n250), .A3(n243), .B1(n246), .B2(n231), 
        .ZN(n236) );
  VHSR_CLKNAND2_2 U278 ( .A1(n236), .A2(n235), .ZN(n234) );
  VHSR_CLKNAND2_2 U279 ( .A1(n238), .A2(n234), .ZN(n233) );
  VHSR_NOR3_2 U280 ( .A1(n275), .A2(n312), .A3(n233), .ZN(n289) );
  VHSR_OAI32_2 U281 ( .A1(n290), .A2(n307), .A3(n272), .B1(n232), .B2(n290), 
        .ZN(n293) );
  VHSR_OAI32_2 U282 ( .A1(n289), .A2(n312), .A3(n275), .B1(n233), .B2(n289), 
        .ZN(n292) );
  VHSR_OAI21_2 U283 ( .A1(n236), .A2(n235), .B(n234), .ZN(n237) );
  VHSR_XNOR2_2 U284 ( .A1(n238), .A2(n237), .ZN(n300) );
  VHSR_AD1_1 U285 ( .A(n241), .B(n240), .CI(n239), .CO(n232), .S(n242) );
  VHSR_IN_2 U286 ( .I(n242), .ZN(n299) );
  VHSR_NOR2_1 U287 ( .A1(n250), .A2(n243), .ZN(n245) );
  VHSR_AOI22_2 U288 ( .A1(n250), .A2(n243), .B1(n246), .B2(n245), .ZN(n244) );
  VHSR_OAI21_2 U289 ( .A1(n246), .A2(n245), .B(n244), .ZN(n305) );
  VHSR_AOI21_2 U290 ( .A1(n249), .A2(n248), .B(n247), .ZN(n304) );
  VHSR_IAO21_2 U291 ( .A1(n254), .A2(n251), .B(n250), .ZN(n320) );
  VHSR_AOI21_2 U292 ( .A1(n258), .A2(n253), .B(n252), .ZN(n319) );
  VHSR_AOI31_2 U293 ( .A1(n257), .A2(n256), .A3(n255), .B(n254), .ZN(n327) );
  VHSR_OAI31_2 U294 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n262) );
  VHSR_IN_2 U295 ( .I(n262), .ZN(n326) );
  VHSR_AOI22_2 U296 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n264) );
  VHSR_NOR2_1 U297 ( .A1(n264), .A2(n263), .ZN(n332) );
  VHSR_NOR2_1 U298 ( .A1(n386), .A2(n385), .ZN(n384) );
  VHSR_AOI22_2 U299 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n265) );
  VHSR_NOR2_1 U300 ( .A1(n266), .A2(n265), .ZN(n331) );
  VHSR_CLKNAND2_2 U301 ( .A1(a[6]), .A2(b[6]), .ZN(n352) );
  VHSR_IN_2 U302 ( .I(n352), .ZN(n379) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[6]), .A2(b[4]), .ZN(n297) );
  VHSR_NAND3_2 U304 ( .A1(a[7]), .A2(b[5]), .A3(n297), .ZN(n268) );
  VHSR_CLKNAND2_2 U305 ( .A1(b[6]), .A2(a[4]), .ZN(n296) );
  VHSR_NAND3_2 U306 ( .A1(b[7]), .A2(a[5]), .A3(n296), .ZN(n267) );
  VHSR_CLKNAND2_2 U307 ( .A1(n268), .A2(n267), .ZN(n270) );
  VHSR_MAOI222_2 U308 ( .A(n352), .B(n268), .C(n267), .ZN(n336) );
  VHSR_IN_2 U309 ( .I(n336), .ZN(n269) );
  VHSR_OAI21_2 U310 ( .A1(n379), .A2(n270), .B(n269), .ZN(n285) );
  VHSR_NOR2_1 U311 ( .A1(n271), .A2(n274), .ZN(n358) );
  VHSR_NOR3_2 U312 ( .A1(n272), .A2(n297), .A3(n278), .ZN(n344) );
  VHSR_AOI22_2 U313 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n273) );
  VHSR_NOR2_1 U314 ( .A1(n344), .A2(n273), .ZN(n281) );
  VHSR_NOR4_2 U315 ( .A1(n276), .A2(n275), .A3(n274), .A4(n279), .ZN(n342) );
  VHSR_AOI22_2 U316 ( .A1(b[6]), .A2(a[5]), .B1(b[7]), .B2(a[4]), .ZN(n277) );
  VHSR_NOR2_1 U317 ( .A1(n342), .A2(n277), .ZN(n280) );
  VHSR_OR3_2 U318 ( .A1(n358), .A2(n279), .A3(n278), .Z(n295) );
  VHSR_MAOI222_2 U319 ( .A(n297), .B(n296), .C(n295), .ZN(n294) );
  VHSR_AND2_2 U320 ( .A1(n287), .A2(n294), .Z(n286) );
  VHSR_AD1_1 U321 ( .A(n301), .B(n281), .CI(n280), .CO(n282), .S(n287) );
  VHSR_NOR2_1 U322 ( .A1(n286), .A2(n282), .ZN(n284) );
  VHSR_CLKNAND2_2 U323 ( .A1(n286), .A2(n282), .ZN(n283) );
  VHSR_NOR2_1 U324 ( .A1(n284), .A2(n285), .ZN(n337) );
  VHSR_AOI22_2 U325 ( .A1(n285), .A2(n284), .B1(n283), .B2(n337), .ZN(n377) );
  VHSR_IAO21_2 U326 ( .A1(n287), .A2(n294), .B(n286), .ZN(n375) );
  VHSR_AD1_1 U327 ( .A(n290), .B(n289), .CI(n288), .CO(n378), .S(n374) );
  VHSR_AD1_1 U328 ( .A(n293), .B(n292), .CI(n291), .CO(n288), .S(n372) );
  VHSR_AOI31_2 U329 ( .A1(n297), .A2(n296), .A3(n295), .B(n294), .ZN(n371) );
  VHSR_AD1_1 U330 ( .A(n300), .B(n299), .CI(n298), .CO(n291), .S(n356) );
  VHSR_AOI22_2 U331 ( .A1(b[4]), .A2(a[5]), .B1(b[5]), .B2(a[4]), .ZN(n302) );
  VHSR_NOR2_1 U332 ( .A1(n302), .A2(n301), .ZN(n355) );
  VHSR_AD1_1 U333 ( .A(n305), .B(n304), .CI(n303), .CO(n298), .S(n359) );
  VHSR_NOR2_1 U334 ( .A1(n314), .A2(n393), .ZN(n324) );
  VHSR_IN_2 U335 ( .I(n324), .ZN(n317) );
  VHSR_CLKNAND2_2 U336 ( .A1(a[3]), .A2(b[3]), .ZN(n323) );
  VHSR_AOI22_2 U337 ( .A1(a[2]), .A2(b[3]), .B1(a[3]), .B2(b[2]), .ZN(n306) );
  VHSR_IAO21_2 U338 ( .A1(n317), .A2(n323), .B(n306), .ZN(n330) );
  VHSR_AOI22_2 U339 ( .A1(a[3]), .A2(b[1]), .B1(a[1]), .B2(b[3]), .ZN(n315) );
  VHSR_CLKNAND2_2 U340 ( .A1(a[1]), .A2(b[1]), .ZN(n309) );
  VHSR_OAI22_2 U341 ( .A1(n317), .A2(n315), .B1(n323), .B2(n309), .ZN(n316) );
  VHSR_OAI22_2 U342 ( .A1(n308), .A2(n393), .B1(n392), .B2(n307), .ZN(n369) );
  VHSR_IN_2 U343 ( .I(n309), .ZN(n310) );
  VHSR_AOI21_2 U344 ( .A1(b[0]), .A2(a[2]), .B(n310), .ZN(n391) );
  VHSR_NOR3_2 U345 ( .A1(n391), .A2(n393), .A3(n392), .ZN(n394) );
  VHSR_OAI22_2 U346 ( .A1(n314), .A2(n313), .B1(n312), .B2(n311), .ZN(n368) );
  VHSR_AOI21_2 U347 ( .A1(n315), .A2(n317), .B(n316), .ZN(n334) );
  VHSR_CLKNAND2_2 U348 ( .A1(n335), .A2(n334), .ZN(n333) );
  VHSR_CLKNAND2_2 U349 ( .A1(n330), .A2(n329), .ZN(n321) );
  VHSR_AOI21_2 U350 ( .A1(n317), .A2(n321), .B(n323), .ZN(n362) );
  VHSR_AD1_1 U351 ( .A(n320), .B(n319), .CI(n318), .CO(n303), .S(n361) );
  VHSR_IN_2 U352 ( .I(n321), .ZN(n328) );
  VHSR_CLKNAND2_2 U353 ( .A1(n328), .A2(n323), .ZN(n322) );
  VHSR_OAI31_2 U354 ( .A1(n324), .A2(n328), .A3(n323), .B(n322), .ZN(n365) );
  VHSR_AD1_1 U355 ( .A(n327), .B(n326), .CI(n325), .CO(n318), .S(n364) );
  VHSR_IAO21_2 U356 ( .A1(n330), .A2(n329), .B(n328), .ZN(n367) );
  VHSR_AD1_1 U357 ( .A(n332), .B(n384), .CI(n331), .CO(n325), .S(n366) );
  VHSR_OAI21_2 U358 ( .A1(n335), .A2(n334), .B(n333), .ZN(n389) );
  VHSR_AOI211_2 U359 ( .A1(n386), .A2(n385), .B(n384), .C(n389), .ZN(n388) );
  VHSR_NOR2_1 U360 ( .A1(n337), .A2(n336), .ZN(n349) );
  VHSR_CLKNAND2_2 U361 ( .A1(b[6]), .A2(a[7]), .ZN(n339) );
  VHSR_AOI21_2 U362 ( .A1(a[6]), .A2(b[7]), .B(n339), .ZN(n338) );
  VHSR_AOI31_2 U363 ( .A1(a[6]), .A2(n339), .A3(b[7]), .B(n338), .ZN(n340) );
  VHSR_IN_2 U364 ( .I(n340), .ZN(n341) );
  VHSR_MAOI222_2 U365 ( .A(n344), .B(n342), .C(n341), .ZN(n351) );
  VHSR_OAI21_2 U366 ( .A1(n344), .A2(n343), .B(n351), .ZN(n348) );
  VHSR_CLKXOR2_2 U367 ( .A1(n349), .A2(n348), .Z(n345) );
  VHSR_CLKNAND2_2 U368 ( .A1(n346), .A2(n345), .ZN(n381) );
  VHSR_OAI21_2 U369 ( .A1(n346), .A2(n345), .B(n381), .ZN(n347) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[7]), .A2(b[7]), .ZN(n380) );
  VHSR_NOR2_1 U371 ( .A1(n349), .A2(n348), .ZN(n350) );
  VHSR_AND3_2 U372 ( .A1(n382), .A2(n352), .A3(n381), .Z(n353) );
  VHSR_NOR2_1 U373 ( .A1(n380), .A2(n353), .ZN(product[15]) );
  VHSR_AD1_1 U374 ( .A(n372), .B(n371), .CI(n370), .CO(n373), .S(product[10])
         );
  VHSR_AD1_1 U375 ( .A(n375), .B(n374), .CI(n373), .CO(n376), .S(product[11])
         );
  VHSR_AD1_1 U376 ( .A(n378), .B(n377), .CI(n376), .CO(n346), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U377 ( .A1(n380), .A2(n379), .ZN(n383) );
  VHSR_XOR3_2 U378 ( .A1(n383), .A2(n382), .A3(n381), .Z(product[14]) );
  VHSR_AOI21_2 U379 ( .A1(n386), .A2(n385), .B(n384), .ZN(n387) );
  VHSR_IN_2 U380 ( .I(n387), .ZN(n390) );
  VHSR_AOI21_2 U381 ( .A1(n390), .A2(n389), .B(n388), .ZN(product[4]) );
  VHSR_OAI32_2 U382 ( .A1(n394), .A2(n393), .A3(n392), .B1(n391), .B2(n394), 
        .ZN(product[2]) );
endmodule



module mul8_6 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , n206, n207, n208, n209, n210,
         n211, n212, n213, n214, n215, n216, n217, n218, n219, n220, n221,
         n222, n223, n224, n225, n226, n227, n228, n229, n230, n231, n232,
         n233, n234, n235, n236, n237, n238, n239, n240, n241, n242, n243,
         n244, n245, n246, n247, n248, n249, n250, n251, n252, n253, n254,
         n255, n256, n257, n258, n259, n260, n261, n262, n263, n264, n265,
         n266, n267, n268, n269, n270, n271, n272, n273, n274, n275, n276,
         n277, n278, n279, n280, n281, n282, n283, n284, n285, n286, n287,
         n288, n289, n290, n291, n292, n293, n294, n295, n296, n297, n298,
         n299, n300, n301, n302, n303, n304, n305, n306, n307, n308, n309,
         n310, n311, n312, n313, n314, n315, n316, n317, n318, n319, n320,
         n321, n322, n323, n324, n325, n326, n327, n328, n329, n330, n331,
         n332, n333, n334, n335, n336, n337, n338, n339, n340, n341, n342,
         n343, n344, n345, n346, n347, n348, n349, n350, n351, n352, n353,
         n354, n355, n356, n357, n358, n359, n360, n361, n362, n363, n364,
         n365, n366, n367, n368, n369, n370, n371, n372, n373, n374, n375,
         n376, n377, n378, n379, n380, n381, n382, n383, n384, n385, n386,
         n387, n388, n389, n390, n391;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;

  VHSR_INOR2_2 U198 ( .A1(a[6]), .B1(n389), .ZN(n206) );
  VHSR_INOR2_2 U199 ( .A1(n227), .B1(n246), .ZN(n233) );
  VHSR_NOR2_1 U200 ( .A1(n240), .A2(n236), .ZN(n229) );
  VHSR_NOR2_1 U201 ( .A1(n284), .A2(n285), .ZN(n335) );
  VHSR_NOR2_1 U202 ( .A1(n384), .A2(n383), .ZN(n382) );
  VHSR_NOR2_1 U203 ( .A1(n276), .A2(n324), .ZN(n359) );
  VHSR_IN_2 U204 ( .I(n345), .ZN(product[13]) );
  VHSR_INOR2_1 U205 ( .A1(n349), .B1(n348), .ZN(n380) );
  VHSR_NOR2_2 U206 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_NOR2_2 U207 ( .A1(n335), .A2(n334), .ZN(n347) );
  VHSR_NOR2_2 U208 ( .A1(n286), .A2(n282), .ZN(n284) );
  VHSR_MOAI22_1 U209 ( .A1(n272), .A2(n387), .B1(a[6]), .B2(b[2]), .ZN(n209)
         );
  VHSR_AD1_1 U210 ( .A(n357), .B(n356), .CI(n355), .CO(n352), .S(product[9])
         );
  VHSR_AD1_1 U211 ( .A(n367), .B(n391), .CI(n366), .CO(n323), .S(product[3])
         );
  VHSR_AD1_1 U212 ( .A(n382), .B(n365), .CI(n364), .CO(n368), .S(product[5])
         );
  VHSR_AD1_1 U213 ( .A(n363), .B(n362), .CI(n361), .CO(n358), .S(product[7])
         );
  VHSR_AD1_1 U214 ( .A(n360), .B(n359), .CI(n358), .CO(n355), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n354), .B(n353), .CI(n352), .CO(n371), .S(product[10])
         );
  VHSR_AOI22_2 U216 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n240) );
  VHSR_IN_2 U217 ( .I(b[3]), .ZN(n319) );
  VHSR_IN_2 U218 ( .I(b[2]), .ZN(n389) );
  VHSR_IN_2 U219 ( .I(a[5]), .ZN(n277) );
  VHSR_IN_2 U220 ( .I(a[4]), .ZN(n276) );
  VHSR_NOR4_2 U221 ( .A1(n319), .A2(n389), .A3(n277), .A4(n276), .ZN(n238) );
  VHSR_IN_2 U222 ( .I(a[7]), .ZN(n272) );
  VHSR_IN_2 U223 ( .I(b[1]), .ZN(n387) );
  VHSR_NOR2_1 U224 ( .A1(n272), .A2(n387), .ZN(n207) );
  VHSR_AOI211_2 U225 ( .A1(b[2]), .A2(a[4]), .B(n319), .C(n277), .ZN(n208) );
  VHSR_MAOI222_2 U226 ( .A(n207), .B(n206), .C(n208), .ZN(n219) );
  VHSR_OAI21_2 U227 ( .A1(n209), .A2(n208), .B(n219), .ZN(n210) );
  VHSR_IN_2 U228 ( .I(n210), .ZN(n243) );
  VHSR_IN_2 U229 ( .I(b[0]), .ZN(n386) );
  VHSR_NOR4_2 U230 ( .A1(n277), .A2(n276), .A3(n387), .A4(n386), .ZN(n266) );
  VHSR_CLKNAND2_2 U231 ( .A1(b[2]), .A2(a[5]), .ZN(n212) );
  VHSR_CLKNAND2_2 U232 ( .A1(b[3]), .A2(a[4]), .ZN(n211) );
  VHSR_AOI21_2 U233 ( .A1(n212), .A2(n211), .B(n238), .ZN(n214) );
  VHSR_AOI22_2 U234 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n216) );
  VHSR_IN_2 U235 ( .I(n216), .ZN(n213) );
  VHSR_MAOI222_2 U236 ( .A(n266), .B(n214), .C(n213), .ZN(n218) );
  VHSR_CLKNAND2_2 U237 ( .A1(b[2]), .A2(a[4]), .ZN(n262) );
  VHSR_OAI211_2 U238 ( .A1(n276), .A2(n386), .B(a[5]), .C(b[1]), .ZN(n261) );
  VHSR_CLKNAND2_2 U239 ( .A1(a[6]), .A2(b[0]), .ZN(n260) );
  VHSR_MAOI222_2 U240 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_NOR2_1 U241 ( .A1(n266), .A2(n214), .ZN(n217) );
  VHSR_IN_2 U242 ( .I(n218), .ZN(n215) );
  VHSR_AOI21_2 U243 ( .A1(n217), .A2(n216), .B(n215), .ZN(n253) );
  VHSR_CLKNAND2_2 U244 ( .A1(n259), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U245 ( .A1(n218), .A2(n252), .ZN(n242) );
  VHSR_CLKNAND2_2 U246 ( .A1(n243), .A2(n242), .ZN(n241) );
  VHSR_CLKNAND2_2 U247 ( .A1(n219), .A2(n241), .ZN(n237) );
  VHSR_AND3_2 U248 ( .A1(n229), .A2(b[3]), .A3(a[7]), .Z(n290) );
  VHSR_IN_2 U249 ( .I(b[7]), .ZN(n274) );
  VHSR_IN_2 U250 ( .I(a[3]), .ZN(n320) );
  VHSR_IN_2 U251 ( .I(b[6]), .ZN(n275) );
  VHSR_IN_2 U252 ( .I(a[2]), .ZN(n315) );
  VHSR_OAI22_2 U253 ( .A1(n275), .A2(n320), .B1(n274), .B2(n315), .ZN(n235) );
  VHSR_NOR2_1 U254 ( .A1(n274), .A2(n315), .ZN(n221) );
  VHSR_IN_2 U255 ( .I(a[1]), .ZN(n385) );
  VHSR_NOR2_1 U256 ( .A1(n275), .A2(n385), .ZN(n220) );
  VHSR_IN_2 U257 ( .I(b[5]), .ZN(n271) );
  VHSR_AOI211_2 U258 ( .A1(b[4]), .A2(a[2]), .B(n271), .C(n320), .ZN(n226) );
  VHSR_OAI22_2 U259 ( .A1(n275), .A2(n315), .B1(n274), .B2(n385), .ZN(n225) );
  VHSR_AOI22_2 U260 ( .A1(n221), .A2(n220), .B1(n226), .B2(n225), .ZN(n227) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[4]), .A2(a[2]), .ZN(n258) );
  VHSR_IN_2 U262 ( .I(b[4]), .ZN(n324) );
  VHSR_IN_2 U263 ( .I(a[0]), .ZN(n390) );
  VHSR_OAI211_2 U264 ( .A1(n324), .A2(n390), .B(b[5]), .C(a[1]), .ZN(n257) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[0]), .ZN(n256) );
  VHSR_MAOI222_2 U266 ( .A(n258), .B(n257), .C(n256), .ZN(n255) );
  VHSR_NOR4_2 U267 ( .A1(n324), .A2(n271), .A3(n385), .A4(n390), .ZN(n264) );
  VHSR_NAND4_2 U268 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n232) );
  VHSR_OAI22_2 U269 ( .A1(n324), .A2(n320), .B1(n271), .B2(n315), .ZN(n222) );
  VHSR_AND2_2 U270 ( .A1(n232), .A2(n222), .Z(n224) );
  VHSR_OAI22_2 U271 ( .A1(n275), .A2(n385), .B1(n274), .B2(n390), .ZN(n223) );
  VHSR_AND2_2 U272 ( .A1(n255), .A2(n251), .Z(n250) );
  VHSR_AD1_1 U273 ( .A(n264), .B(n224), .CI(n223), .CO(n245), .S(n251) );
  VHSR_NOR2_1 U274 ( .A1(n250), .A2(n245), .ZN(n248) );
  VHSR_OAI21_2 U275 ( .A1(n226), .A2(n225), .B(n227), .ZN(n249) );
  VHSR_NOR2_1 U276 ( .A1(n248), .A2(n249), .ZN(n246) );
  VHSR_CLKNAND2_2 U277 ( .A1(n233), .A2(n232), .ZN(n231) );
  VHSR_CLKNAND2_2 U278 ( .A1(n235), .A2(n231), .ZN(n230) );
  VHSR_NOR3_2 U279 ( .A1(n274), .A2(n320), .A3(n230), .ZN(n289) );
  VHSR_NOR2_1 U280 ( .A1(n319), .A2(n272), .ZN(n228) );
  VHSR_IAO21_2 U281 ( .A1(n229), .A2(n228), .B(n290), .ZN(n293) );
  VHSR_OAI32_2 U282 ( .A1(n289), .A2(n320), .A3(n274), .B1(n230), .B2(n289), 
        .ZN(n292) );
  VHSR_OAI21_2 U283 ( .A1(n233), .A2(n232), .B(n231), .ZN(n234) );
  VHSR_XNOR2_2 U284 ( .A1(n235), .A2(n234), .ZN(n300) );
  VHSR_AOI21_2 U285 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U286 ( .A1(n240), .A2(n239), .ZN(n299) );
  VHSR_OAI21_2 U287 ( .A1(n243), .A2(n242), .B(n241), .ZN(n244) );
  VHSR_IN_2 U288 ( .I(n244), .ZN(n305) );
  VHSR_CLKNAND2_2 U289 ( .A1(n250), .A2(n245), .ZN(n247) );
  VHSR_AOI22_2 U290 ( .A1(n249), .A2(n248), .B1(n247), .B2(n246), .ZN(n304) );
  VHSR_IAO21_2 U291 ( .A1(n255), .A2(n251), .B(n250), .ZN(n308) );
  VHSR_OAI21_2 U292 ( .A1(n259), .A2(n253), .B(n252), .ZN(n254) );
  VHSR_IN_2 U293 ( .I(n254), .ZN(n307) );
  VHSR_AOI31_2 U294 ( .A1(n258), .A2(n257), .A3(n256), .B(n255), .ZN(n318) );
  VHSR_AOI31_2 U295 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n317) );
  VHSR_CLKNAND2_2 U296 ( .A1(b[5]), .A2(a[0]), .ZN(n263) );
  VHSR_OAI32_2 U297 ( .A1(n264), .A2(n385), .A3(n324), .B1(n263), .B2(n264), 
        .ZN(n333) );
  VHSR_CLKNAND2_2 U298 ( .A1(a[4]), .A2(b[1]), .ZN(n265) );
  VHSR_OAI32_2 U299 ( .A1(n266), .A2(n386), .A3(n277), .B1(n265), .B2(n266), 
        .ZN(n332) );
  VHSR_NOR2_1 U300 ( .A1(n386), .A2(n390), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U301 ( .A1(n359), .A2(product[0]), .ZN(n326) );
  VHSR_IN_2 U302 ( .I(n326), .ZN(n331) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[6]), .A2(b[6]), .ZN(n350) );
  VHSR_IN_2 U304 ( .I(n350), .ZN(n377) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[6]), .A2(b[4]), .ZN(n297) );
  VHSR_NAND3_2 U306 ( .A1(a[7]), .A2(b[5]), .A3(n297), .ZN(n268) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[4]), .A2(b[6]), .ZN(n296) );
  VHSR_NAND3_2 U308 ( .A1(b[7]), .A2(a[5]), .A3(n296), .ZN(n267) );
  VHSR_CLKNAND2_2 U309 ( .A1(n268), .A2(n267), .ZN(n270) );
  VHSR_MAOI222_2 U310 ( .A(n350), .B(n268), .C(n267), .ZN(n334) );
  VHSR_IN_2 U311 ( .I(n334), .ZN(n269) );
  VHSR_OAI21_2 U312 ( .A1(n377), .A2(n270), .B(n269), .ZN(n285) );
  VHSR_IN_2 U313 ( .I(n359), .ZN(n279) );
  VHSR_NOR3_2 U314 ( .A1(n277), .A2(n271), .A3(n279), .ZN(n301) );
  VHSR_NOR3_2 U315 ( .A1(n272), .A2(n297), .A3(n271), .ZN(n342) );
  VHSR_AOI22_2 U316 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n273) );
  VHSR_NOR2_1 U317 ( .A1(n342), .A2(n273), .ZN(n281) );
  VHSR_NOR4_2 U318 ( .A1(n277), .A2(n276), .A3(n275), .A4(n274), .ZN(n340) );
  VHSR_AOI22_2 U319 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n278) );
  VHSR_NOR2_1 U320 ( .A1(n340), .A2(n278), .ZN(n280) );
  VHSR_NAND3_2 U321 ( .A1(b[5]), .A2(a[5]), .A3(n279), .ZN(n295) );
  VHSR_MAOI222_2 U322 ( .A(n297), .B(n296), .C(n295), .ZN(n294) );
  VHSR_AND2_2 U323 ( .A1(n287), .A2(n294), .Z(n286) );
  VHSR_AD1_1 U324 ( .A(n301), .B(n281), .CI(n280), .CO(n282), .S(n287) );
  VHSR_CLKNAND2_2 U325 ( .A1(n286), .A2(n282), .ZN(n283) );
  VHSR_AOI22_2 U326 ( .A1(n285), .A2(n284), .B1(n283), .B2(n335), .ZN(n375) );
  VHSR_IAO21_2 U327 ( .A1(n287), .A2(n294), .B(n286), .ZN(n373) );
  VHSR_AD1_1 U328 ( .A(n290), .B(n289), .CI(n288), .CO(n376), .S(n372) );
  VHSR_AD1_1 U329 ( .A(n293), .B(n292), .CI(n291), .CO(n288), .S(n354) );
  VHSR_AOI31_2 U330 ( .A1(n297), .A2(n296), .A3(n295), .B(n294), .ZN(n353) );
  VHSR_AD1_1 U331 ( .A(n300), .B(n299), .CI(n298), .CO(n291), .S(n357) );
  VHSR_AOI22_2 U332 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n302) );
  VHSR_NOR2_1 U333 ( .A1(n302), .A2(n301), .ZN(n356) );
  VHSR_AD1_1 U334 ( .A(n305), .B(n304), .CI(n303), .CO(n298), .S(n360) );
  VHSR_AD1_1 U335 ( .A(n308), .B(n307), .CI(n306), .CO(n303), .S(n363) );
  VHSR_NOR2_1 U336 ( .A1(n319), .A2(n385), .ZN(n311) );
  VHSR_NOR2_1 U337 ( .A1(n387), .A2(n320), .ZN(n310) );
  VHSR_NOR2_1 U338 ( .A1(n389), .A2(n315), .ZN(n309) );
  VHSR_MAOI222_2 U339 ( .A(n311), .B(n310), .C(n309), .ZN(n314) );
  VHSR_OAI22_2 U340 ( .A1(n319), .A2(n390), .B1(n389), .B2(n385), .ZN(n367) );
  VHSR_AOI22_2 U341 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n388) );
  VHSR_NOR3_2 U342 ( .A1(n388), .A2(n390), .A3(n389), .ZN(n391) );
  VHSR_OAI22_2 U343 ( .A1(n387), .A2(n315), .B1(n386), .B2(n320), .ZN(n366) );
  VHSR_IN_2 U344 ( .I(n314), .ZN(n313) );
  VHSR_AOI22_2 U345 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n312) );
  VHSR_OAI32_2 U346 ( .A1(n313), .A2(n315), .A3(n389), .B1(n312), .B2(n313), 
        .ZN(n322) );
  VHSR_CLKNAND2_2 U347 ( .A1(n323), .A2(n322), .ZN(n321) );
  VHSR_CLKNAND2_2 U348 ( .A1(n314), .A2(n321), .ZN(n328) );
  VHSR_OAI22_2 U349 ( .A1(n319), .A2(n315), .B1(n389), .B2(n320), .ZN(n329) );
  VHSR_CLKNAND2_2 U350 ( .A1(n328), .A2(n329), .ZN(n327) );
  VHSR_NOR3_2 U351 ( .A1(n319), .A2(n320), .A3(n327), .ZN(n362) );
  VHSR_AD1_1 U352 ( .A(n318), .B(n317), .CI(n316), .CO(n306), .S(n370) );
  VHSR_OAI32_2 U353 ( .A1(n362), .A2(n320), .A3(n319), .B1(n327), .B2(n362), 
        .ZN(n369) );
  VHSR_OAI21_2 U354 ( .A1(n323), .A2(n322), .B(n321), .ZN(n384) );
  VHSR_NOR2_1 U355 ( .A1(n324), .A2(n390), .ZN(n325) );
  VHSR_AOI32_2 U356 ( .A1(b[0]), .A2(n326), .A3(a[4]), .B1(n325), .B2(n326), 
        .ZN(n383) );
  VHSR_OAI21_2 U357 ( .A1(n329), .A2(n328), .B(n327), .ZN(n330) );
  VHSR_IN_2 U358 ( .I(n330), .ZN(n365) );
  VHSR_AD1_1 U359 ( .A(n333), .B(n332), .CI(n331), .CO(n316), .S(n364) );
  VHSR_CLKNAND2_2 U360 ( .A1(a[7]), .A2(b[6]), .ZN(n337) );
  VHSR_AOI21_2 U361 ( .A1(a[6]), .A2(b[7]), .B(n337), .ZN(n336) );
  VHSR_AOI31_2 U362 ( .A1(a[6]), .A2(n337), .A3(b[7]), .B(n336), .ZN(n338) );
  VHSR_IN_2 U363 ( .I(n338), .ZN(n339) );
  VHSR_OR2_2 U364 ( .A1(n340), .A2(n339), .Z(n341) );
  VHSR_MAOI222_2 U365 ( .A(n342), .B(n340), .C(n339), .ZN(n349) );
  VHSR_OAI21_2 U366 ( .A1(n342), .A2(n341), .B(n349), .ZN(n346) );
  VHSR_CLKXOR2_2 U367 ( .A1(n347), .A2(n346), .Z(n343) );
  VHSR_CLKNAND2_2 U368 ( .A1(n344), .A2(n343), .ZN(n379) );
  VHSR_OAI21_2 U369 ( .A1(n344), .A2(n343), .B(n379), .ZN(n345) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[7]), .A2(b[7]), .ZN(n378) );
  VHSR_NOR2_1 U371 ( .A1(n347), .A2(n346), .ZN(n348) );
  VHSR_AND3_2 U372 ( .A1(n380), .A2(n350), .A3(n379), .Z(n351) );
  VHSR_NOR2_1 U373 ( .A1(n378), .A2(n351), .ZN(product[15]) );
  VHSR_AD1_1 U374 ( .A(n370), .B(n369), .CI(n368), .CO(n361), .S(product[6])
         );
  VHSR_AD1_1 U375 ( .A(n373), .B(n372), .CI(n371), .CO(n374), .S(product[11])
         );
  VHSR_AD1_1 U376 ( .A(n376), .B(n375), .CI(n374), .CO(n344), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U377 ( .A1(n378), .A2(n377), .ZN(n381) );
  VHSR_XOR3_2 U378 ( .A1(n381), .A2(n380), .A3(n379), .Z(product[14]) );
  VHSR_AOI21_2 U379 ( .A1(n384), .A2(n383), .B(n382), .ZN(product[4]) );
  VHSR_OAI22_2 U380 ( .A1(n387), .A2(n390), .B1(n386), .B2(n385), .ZN(
        product[1]) );
  VHSR_OAI32_2 U381 ( .A1(n391), .A2(n390), .A3(n389), .B1(n388), .B2(n391), 
        .ZN(product[2]) );
endmodule


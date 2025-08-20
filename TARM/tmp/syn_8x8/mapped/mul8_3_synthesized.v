
module mul8_3 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \mul_ll_ll/out[0] , \intadd_0/SUM[7] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n211, n212, n213, n214, n215, n216, n217, n218,
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
         n384, n385, n386, n387, n388, n389, n390, n391, n392, n393, n394;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U202 ( .A1(n237), .B1(n213), .ZN(n214) );
  VHSR_NOR2_1 U203 ( .A1(n392), .A2(n276), .ZN(n259) );
  VHSR_INOR2_2 U204 ( .A1(n219), .B1(n245), .ZN(n238) );
  VHSR_INOR2_2 U205 ( .A1(n217), .B1(n248), .ZN(n247) );
  VHSR_INAND2_2 U206 ( .A1(n314), .B1(n333), .ZN(n329) );
  VHSR_NOR2_1 U207 ( .A1(n229), .A2(n228), .ZN(n288) );
  VHSR_INOR2_2 U208 ( .A1(n351), .B1(n350), .ZN(n382) );
  VHSR_IN_2 U209 ( .I(n347), .ZN(product[13]) );
  VHSR_MOAI22_1 U210 ( .A1(n270), .A2(n311), .B1(b[4]), .B2(a[3]), .ZN(n221)
         );
  VHSR_AD1_1 U211 ( .A(n370), .B(n369), .CI(n388), .CO(n366), .S(product[5])
         );
  VHSR_AD1_1 U212 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U213 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(product[9])
         );
  VHSR_AD1_1 U214 ( .A(n372), .B(n394), .CI(n371), .CO(n335), .S(product[3])
         );
  VHSR_AD1_1 U215 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[6])
         );
  VHSR_AD1_1 U216 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U217 ( .A(n356), .B(n355), .CI(n354), .CO(n373), .S(product[10])
         );
  VHSR_PULL0_0 U218 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U219 ( .I(b[1]), .ZN(n312) );
  VHSR_IN_2 U220 ( .I(a[0]), .ZN(n393) );
  VHSR_NOR2_1 U221 ( .A1(n312), .A2(n393), .ZN(product[1]) );
  VHSR_CLKNAND2_2 U222 ( .A1(b[3]), .A2(a[7]), .ZN(n229) );
  VHSR_IN_2 U223 ( .I(b[3]), .ZN(n315) );
  VHSR_IN_2 U224 ( .I(a[6]), .ZN(n265) );
  VHSR_IN_2 U225 ( .I(a[7]), .ZN(n271) );
  VHSR_IN_2 U226 ( .I(b[2]), .ZN(n392) );
  VHSR_OAI22_2 U227 ( .A1(n315), .A2(n265), .B1(n271), .B2(n392), .ZN(n240) );
  VHSR_IN_2 U228 ( .I(a[4]), .ZN(n276) );
  VHSR_CLKNAND2_2 U229 ( .A1(b[3]), .A2(a[5]), .ZN(n211) );
  VHSR_OAI22_2 U230 ( .A1(n259), .A2(n211), .B1(n271), .B2(n312), .ZN(n218) );
  VHSR_IN_2 U231 ( .I(a[5]), .ZN(n275) );
  VHSR_NOR4_2 U232 ( .A1(n259), .A2(n229), .A3(n275), .A4(n312), .ZN(n212) );
  VHSR_AOI31_2 U233 ( .A1(a[6]), .A2(b[2]), .A3(n218), .B(n212), .ZN(n219) );
  VHSR_IN_2 U234 ( .I(b[0]), .ZN(n310) );
  VHSR_NOR4_2 U235 ( .A1(n276), .A2(n275), .A3(n312), .A4(n310), .ZN(n264) );
  VHSR_NAND3_2 U236 ( .A1(b[3]), .A2(n259), .A3(a[5]), .ZN(n237) );
  VHSR_AOI22_2 U237 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n213) );
  VHSR_OAI22_2 U238 ( .A1(n271), .A2(n310), .B1(n265), .B2(n312), .ZN(n215) );
  VHSR_MAOI222_2 U239 ( .A(n264), .B(n214), .C(n215), .ZN(n217) );
  VHSR_NOR2_1 U240 ( .A1(n265), .A2(n310), .ZN(n258) );
  VHSR_AOI211_2 U241 ( .A1(a[4]), .A2(b[0]), .B(n275), .C(n312), .ZN(n257) );
  VHSR_MAOI222_2 U242 ( .A(n259), .B(n258), .C(n257), .ZN(n256) );
  VHSR_OR2_2 U243 ( .A1(n264), .A2(n214), .Z(n216) );
  VHSR_OAI21_2 U244 ( .A1(n216), .A2(n215), .B(n217), .ZN(n249) );
  VHSR_NOR2_1 U245 ( .A1(n256), .A2(n249), .ZN(n248) );
  VHSR_AOI32_2 U246 ( .A1(a[6]), .A2(n219), .A3(b[2]), .B1(n218), .B2(n219), 
        .ZN(n246) );
  VHSR_NOR2_1 U247 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_CLKNAND2_2 U248 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U249 ( .A1(n240), .A2(n236), .ZN(n228) );
  VHSR_IN_2 U250 ( .I(b[7]), .ZN(n273) );
  VHSR_IN_2 U251 ( .I(a[3]), .ZN(n316) );
  VHSR_IN_2 U252 ( .I(b[6]), .ZN(n274) );
  VHSR_IN_2 U253 ( .I(a[2]), .ZN(n311) );
  VHSR_OAI22_2 U254 ( .A1(n274), .A2(n316), .B1(n273), .B2(n311), .ZN(n235) );
  VHSR_AOI22_2 U255 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n226) );
  VHSR_CLKNAND2_2 U256 ( .A1(b[4]), .A2(a[2]), .ZN(n255) );
  VHSR_NAND3_2 U257 ( .A1(a[3]), .A2(b[5]), .A3(n255), .ZN(n225) );
  VHSR_CLKNAND2_2 U258 ( .A1(b[7]), .A2(a[2]), .ZN(n220) );
  VHSR_CLKNAND2_2 U259 ( .A1(b[6]), .A2(a[1]), .ZN(n222) );
  VHSR_OAI22_2 U260 ( .A1(n226), .A2(n225), .B1(n220), .B2(n222), .ZN(n227) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[4]), .A2(a[0]), .ZN(n385) );
  VHSR_NAND3_2 U262 ( .A1(a[1]), .A2(b[5]), .A3(n385), .ZN(n254) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[6]), .A2(a[0]), .ZN(n253) );
  VHSR_MAOI222_2 U264 ( .A(n255), .B(n254), .C(n253), .ZN(n252) );
  VHSR_NAND4_2 U265 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n232) );
  VHSR_IN_2 U266 ( .I(b[5]), .ZN(n270) );
  VHSR_AND2_2 U267 ( .A1(n232), .A2(n221), .Z(n224) );
  VHSR_OAI21_2 U268 ( .A1(n273), .A2(n393), .B(n222), .ZN(n223) );
  VHSR_IN_2 U269 ( .I(a[1]), .ZN(n308) );
  VHSR_NOR3_2 U270 ( .A1(n270), .A2(n308), .A3(n385), .ZN(n262) );
  VHSR_AND2_2 U271 ( .A1(n252), .A2(n251), .Z(n250) );
  VHSR_AD1_1 U272 ( .A(n224), .B(n223), .CI(n262), .CO(n241), .S(n251) );
  VHSR_AOI21_2 U273 ( .A1(n226), .A2(n225), .B(n227), .ZN(n244) );
  VHSR_OAI32_2 U274 ( .A1(n227), .A2(n250), .A3(n241), .B1(n244), .B2(n227), 
        .ZN(n233) );
  VHSR_CLKNAND2_2 U275 ( .A1(n233), .A2(n232), .ZN(n231) );
  VHSR_CLKNAND2_2 U276 ( .A1(n235), .A2(n231), .ZN(n230) );
  VHSR_NOR3_2 U277 ( .A1(n273), .A2(n316), .A3(n230), .ZN(n287) );
  VHSR_AOI21_2 U278 ( .A1(n229), .A2(n228), .B(n288), .ZN(n291) );
  VHSR_OAI32_2 U279 ( .A1(n287), .A2(n316), .A3(n273), .B1(n230), .B2(n287), 
        .ZN(n290) );
  VHSR_OAI21_2 U280 ( .A1(n233), .A2(n232), .B(n231), .ZN(n234) );
  VHSR_XNOR2_2 U281 ( .A1(n235), .A2(n234), .ZN(n298) );
  VHSR_OAI21_2 U282 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U283 ( .A1(n240), .A2(n239), .ZN(n297) );
  VHSR_NOR2_1 U284 ( .A1(n250), .A2(n241), .ZN(n243) );
  VHSR_AOI22_2 U285 ( .A1(n250), .A2(n241), .B1(n244), .B2(n243), .ZN(n242) );
  VHSR_OAI21_2 U286 ( .A1(n244), .A2(n243), .B(n242), .ZN(n303) );
  VHSR_AOI21_2 U287 ( .A1(n247), .A2(n246), .B(n245), .ZN(n302) );
  VHSR_AOI21_2 U288 ( .A1(n256), .A2(n249), .B(n248), .ZN(n319) );
  VHSR_IAO21_2 U289 ( .A1(n252), .A2(n251), .B(n250), .ZN(n318) );
  VHSR_AOI31_2 U290 ( .A1(n255), .A2(n254), .A3(n253), .B(n252), .ZN(n327) );
  VHSR_OAI31_2 U291 ( .A1(n259), .A2(n258), .A3(n257), .B(n256), .ZN(n260) );
  VHSR_IN_2 U292 ( .I(n260), .ZN(n326) );
  VHSR_AOI22_2 U293 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n261) );
  VHSR_NOR2_1 U294 ( .A1(n262), .A2(n261), .ZN(n332) );
  VHSR_CLKNAND2_2 U295 ( .A1(a[4]), .A2(b[4]), .ZN(n304) );
  VHSR_NOR3_2 U296 ( .A1(n310), .A2(n304), .A3(n393), .ZN(n384) );
  VHSR_CLKNAND2_2 U297 ( .A1(a[5]), .A2(b[0]), .ZN(n263) );
  VHSR_OAI32_2 U298 ( .A1(n264), .A2(n312), .A3(n276), .B1(n263), .B2(n264), 
        .ZN(n331) );
  VHSR_NOR2_1 U299 ( .A1(n265), .A2(n274), .ZN(n379) );
  VHSR_CLKNAND2_2 U300 ( .A1(a[6]), .A2(b[4]), .ZN(n295) );
  VHSR_NAND3_2 U301 ( .A1(a[7]), .A2(b[5]), .A3(n295), .ZN(n267) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[4]), .A2(b[6]), .ZN(n294) );
  VHSR_NAND3_2 U303 ( .A1(b[7]), .A2(a[5]), .A3(n294), .ZN(n266) );
  VHSR_CLKNAND2_2 U304 ( .A1(n267), .A2(n266), .ZN(n269) );
  VHSR_IN_2 U305 ( .I(n379), .ZN(n352) );
  VHSR_MAOI222_2 U306 ( .A(n352), .B(n267), .C(n266), .ZN(n336) );
  VHSR_IN_2 U307 ( .I(n336), .ZN(n268) );
  VHSR_OAI21_2 U308 ( .A1(n379), .A2(n269), .B(n268), .ZN(n283) );
  VHSR_NOR3_2 U309 ( .A1(n275), .A2(n270), .A3(n304), .ZN(n299) );
  VHSR_NOR3_2 U310 ( .A1(n271), .A2(n295), .A3(n270), .ZN(n344) );
  VHSR_AOI22_2 U311 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n272) );
  VHSR_NOR2_1 U312 ( .A1(n344), .A2(n272), .ZN(n279) );
  VHSR_NOR4_2 U313 ( .A1(n276), .A2(n275), .A3(n274), .A4(n273), .ZN(n342) );
  VHSR_AOI22_2 U314 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n277) );
  VHSR_NOR2_1 U315 ( .A1(n342), .A2(n277), .ZN(n278) );
  VHSR_NAND3_2 U316 ( .A1(b[5]), .A2(a[5]), .A3(n304), .ZN(n293) );
  VHSR_MAOI222_2 U317 ( .A(n295), .B(n294), .C(n293), .ZN(n292) );
  VHSR_AND2_2 U318 ( .A1(n285), .A2(n292), .Z(n284) );
  VHSR_AD1_1 U319 ( .A(n299), .B(n279), .CI(n278), .CO(n280), .S(n285) );
  VHSR_NOR2_1 U320 ( .A1(n284), .A2(n280), .ZN(n282) );
  VHSR_CLKNAND2_2 U321 ( .A1(n284), .A2(n280), .ZN(n281) );
  VHSR_NOR2_1 U322 ( .A1(n282), .A2(n283), .ZN(n337) );
  VHSR_AOI22_2 U323 ( .A1(n283), .A2(n282), .B1(n281), .B2(n337), .ZN(n377) );
  VHSR_IAO21_2 U324 ( .A1(n285), .A2(n292), .B(n284), .ZN(n375) );
  VHSR_AD1_1 U325 ( .A(n288), .B(n287), .CI(n286), .CO(n378), .S(n374) );
  VHSR_AD1_1 U326 ( .A(n291), .B(n290), .CI(n289), .CO(n286), .S(n356) );
  VHSR_AOI31_2 U327 ( .A1(n295), .A2(n294), .A3(n293), .B(n292), .ZN(n355) );
  VHSR_AD1_1 U328 ( .A(n298), .B(n297), .CI(n296), .CO(n289), .S(n359) );
  VHSR_AOI22_2 U329 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n300) );
  VHSR_NOR2_1 U330 ( .A1(n300), .A2(n299), .ZN(n358) );
  VHSR_AD1_1 U331 ( .A(n303), .B(n302), .CI(n301), .CO(n296), .S(n362) );
  VHSR_IN_2 U332 ( .I(n304), .ZN(n361) );
  VHSR_CLKNAND2_2 U333 ( .A1(b[2]), .A2(a[2]), .ZN(n320) );
  VHSR_NOR2_1 U334 ( .A1(n392), .A2(n316), .ZN(n306) );
  VHSR_OAI21_2 U335 ( .A1(n315), .A2(n311), .B(n306), .ZN(n305) );
  VHSR_OAI31_2 U336 ( .A1(n315), .A2(n306), .A3(n311), .B(n305), .ZN(n330) );
  VHSR_AOI22_2 U337 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n313) );
  VHSR_CLKNAND2_2 U338 ( .A1(b[3]), .A2(a[3]), .ZN(n323) );
  VHSR_NOR2_1 U339 ( .A1(n312), .A2(n308), .ZN(n309) );
  VHSR_IN_2 U340 ( .I(n309), .ZN(n307) );
  VHSR_OAI22_2 U341 ( .A1(n320), .A2(n313), .B1(n323), .B2(n307), .ZN(n314) );
  VHSR_OAI22_2 U342 ( .A1(n315), .A2(n393), .B1(n392), .B2(n308), .ZN(n372) );
  VHSR_AOI21_2 U343 ( .A1(a[2]), .A2(b[0]), .B(n309), .ZN(n391) );
  VHSR_NOR3_2 U344 ( .A1(n391), .A2(n393), .A3(n392), .ZN(n394) );
  VHSR_OAI22_2 U345 ( .A1(n312), .A2(n311), .B1(n310), .B2(n316), .ZN(n371) );
  VHSR_AOI21_2 U346 ( .A1(n313), .A2(n320), .B(n314), .ZN(n334) );
  VHSR_CLKNAND2_2 U347 ( .A1(n335), .A2(n334), .ZN(n333) );
  VHSR_CLKNAND2_2 U348 ( .A1(n330), .A2(n329), .ZN(n321) );
  VHSR_AOI211_2 U349 ( .A1(n320), .A2(n321), .B(n316), .C(n315), .ZN(n365) );
  VHSR_AD1_1 U350 ( .A(n319), .B(n318), .CI(n317), .CO(n301), .S(n364) );
  VHSR_IN_2 U351 ( .I(n320), .ZN(n324) );
  VHSR_IN_2 U352 ( .I(n321), .ZN(n328) );
  VHSR_CLKNAND2_2 U353 ( .A1(n328), .A2(n323), .ZN(n322) );
  VHSR_OAI31_2 U354 ( .A1(n324), .A2(n328), .A3(n323), .B(n322), .ZN(n368) );
  VHSR_AD1_1 U355 ( .A(n327), .B(n326), .CI(n325), .CO(n317), .S(n367) );
  VHSR_IAO21_2 U356 ( .A1(n330), .A2(n329), .B(n328), .ZN(n370) );
  VHSR_AD1_1 U357 ( .A(n332), .B(n384), .CI(n331), .CO(n325), .S(n369) );
  VHSR_CLKNAND2_2 U358 ( .A1(a[4]), .A2(b[0]), .ZN(n386) );
  VHSR_OAI21_2 U359 ( .A1(n335), .A2(n334), .B(n333), .ZN(n389) );
  VHSR_AOI211_2 U360 ( .A1(n386), .A2(n385), .B(n384), .C(n389), .ZN(n388) );
  VHSR_NOR2_1 U361 ( .A1(n337), .A2(n336), .ZN(n349) );
  VHSR_CLKNAND2_2 U362 ( .A1(a[6]), .A2(b[7]), .ZN(n339) );
  VHSR_AOI21_2 U363 ( .A1(a[7]), .A2(b[6]), .B(n339), .ZN(n338) );
  VHSR_AOI31_2 U364 ( .A1(a[7]), .A2(n339), .A3(b[6]), .B(n338), .ZN(n340) );
  VHSR_IN_2 U365 ( .I(n340), .ZN(n341) );
  VHSR_OR2_2 U366 ( .A1(n342), .A2(n341), .Z(n343) );
  VHSR_MAOI222_2 U367 ( .A(n344), .B(n342), .C(n341), .ZN(n351) );
  VHSR_OAI21_2 U368 ( .A1(n344), .A2(n343), .B(n351), .ZN(n348) );
  VHSR_CLKXOR2_2 U369 ( .A1(n349), .A2(n348), .Z(n345) );
  VHSR_CLKNAND2_2 U370 ( .A1(n346), .A2(n345), .ZN(n381) );
  VHSR_OAI21_2 U371 ( .A1(n346), .A2(n345), .B(n381), .ZN(n347) );
  VHSR_CLKNAND2_2 U372 ( .A1(a[7]), .A2(b[7]), .ZN(n380) );
  VHSR_NOR2_1 U373 ( .A1(n349), .A2(n348), .ZN(n350) );
  VHSR_AND3_2 U374 ( .A1(n382), .A2(n352), .A3(n381), .Z(n353) );
  VHSR_NOR2_1 U375 ( .A1(n380), .A2(n353), .ZN(product[15]) );
  VHSR_AD1_1 U376 ( .A(n375), .B(n374), .CI(n373), .CO(n376), .S(product[11])
         );
  VHSR_AD1_1 U377 ( .A(n378), .B(n377), .CI(n376), .CO(n346), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U378 ( .A1(n380), .A2(n379), .ZN(n383) );
  VHSR_XOR3_2 U379 ( .A1(n383), .A2(n382), .A3(n381), .Z(product[14]) );
  VHSR_AOI21_2 U380 ( .A1(n386), .A2(n385), .B(n384), .ZN(n387) );
  VHSR_IN_2 U381 ( .I(n387), .ZN(n390) );
  VHSR_AOI21_2 U382 ( .A1(n390), .A2(n389), .B(n388), .ZN(product[4]) );
  VHSR_OAI32_2 U383 ( .A1(n394), .A2(n393), .A3(n392), .B1(n391), .B2(n394), 
        .ZN(product[2]) );
endmodule


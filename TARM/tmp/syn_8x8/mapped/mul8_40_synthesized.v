
module mul8_40 ( a, b, product );
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
         n394, n395, n396;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U201 ( .A1(n346), .B1(n344), .ZN(n347) );
  VHSR_INAND2_2 U202 ( .A1(n225), .B1(n249), .ZN(n242) );
  VHSR_NOR2_1 U203 ( .A1(n341), .A2(n340), .ZN(n353) );
  VHSR_INAND2_2 U204 ( .A1(n318), .B1(n335), .ZN(n331) );
  VHSR_INOR3_2 U205 ( .A1(n362), .B1(n281), .B2(n282), .ZN(n305) );
  VHSR_NOR2_1 U206 ( .A1(n234), .A2(n274), .ZN(n294) );
  VHSR_NOR2_1 U207 ( .A1(n278), .A2(n273), .ZN(n362) );
  VHSR_IN_2 U208 ( .I(n351), .ZN(product[13]) );
  VHSR_INOR2_1 U209 ( .A1(n355), .B1(n354), .ZN(n386) );
  VHSR_MOAI22_1 U210 ( .A1(n222), .A2(n221), .B1(a[6]), .B2(n268), .ZN(n220)
         );
  VHSR_NOR2_2 U211 ( .A1(n291), .A2(n290), .ZN(n289) );
  VHSR_MOAI22_1 U212 ( .A1(n316), .A2(n315), .B1(b[0]), .B2(a[3]), .ZN(n372)
         );
  VHSR_AD1_1 U213 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(product[6])
         );
  VHSR_AD1_1 U214 ( .A(n363), .B(n362), .CI(n361), .CO(n358), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n373), .B(n396), .CI(n372), .CO(n337), .S(product[3])
         );
  VHSR_AD1_1 U216 ( .A(n371), .B(n370), .CI(n390), .CO(n367), .S(product[5])
         );
  VHSR_AD1_1 U217 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U218 ( .A(n360), .B(n359), .CI(n358), .CO(n374), .S(product[9])
         );
  VHSR_PULL0_0 U219 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U220 ( .I(b[1]), .ZN(n316) );
  VHSR_IN_2 U221 ( .I(a[0]), .ZN(n395) );
  VHSR_NOR2_1 U222 ( .A1(n316), .A2(n395), .ZN(product[1]) );
  VHSR_IN_2 U223 ( .I(b[3]), .ZN(n312) );
  VHSR_IN_2 U224 ( .I(b[2]), .ZN(n394) );
  VHSR_IN_2 U225 ( .I(a[4]), .ZN(n278) );
  VHSR_IN_2 U226 ( .I(a[5]), .ZN(n281) );
  VHSR_NOR4_2 U227 ( .A1(n312), .A2(n394), .A3(n278), .A4(n281), .ZN(n244) );
  VHSR_OAI211_2 U228 ( .A1(n278), .A2(n394), .B(b[3]), .C(a[5]), .ZN(n212) );
  VHSR_IN_2 U229 ( .I(a[6]), .ZN(n215) );
  VHSR_IN_2 U230 ( .I(a[7]), .ZN(n274) );
  VHSR_NAND3_2 U231 ( .A1(a[7]), .A2(a[6]), .A3(b[1]), .ZN(n211) );
  VHSR_AOI22_2 U232 ( .A1(n215), .A2(n274), .B1(n211), .B2(n394), .ZN(n210) );
  VHSR_IN_2 U233 ( .I(n210), .ZN(n213) );
  VHSR_OAI22_2 U234 ( .A1(n212), .A2(n213), .B1(n394), .B2(n211), .ZN(n225) );
  VHSR_AOI21_2 U235 ( .A1(n213), .A2(n212), .B(n225), .ZN(n251) );
  VHSR_CLKNAND2_2 U236 ( .A1(a[4]), .A2(b[0]), .ZN(n339) );
  VHSR_NOR3_2 U237 ( .A1(n281), .A2(n316), .A3(n339), .ZN(n268) );
  VHSR_IN_2 U238 ( .I(n339), .ZN(n389) );
  VHSR_CLKNAND2_2 U239 ( .A1(a[5]), .A2(n389), .ZN(n214) );
  VHSR_AOI21_2 U240 ( .A1(n215), .A2(n214), .B(n316), .ZN(n219) );
  VHSR_NOR2_1 U241 ( .A1(n394), .A2(n281), .ZN(n217) );
  VHSR_OAI21_2 U242 ( .A1(n312), .A2(n278), .B(n217), .ZN(n216) );
  VHSR_OAI31_2 U243 ( .A1(n312), .A2(n217), .A3(n278), .B(n216), .ZN(n218) );
  VHSR_AOI22_2 U244 ( .A1(a[6]), .A2(n268), .B1(n219), .B2(n218), .ZN(n224) );
  VHSR_IN_2 U245 ( .I(n218), .ZN(n222) );
  VHSR_IN_2 U246 ( .I(n219), .ZN(n221) );
  VHSR_AOI21_2 U247 ( .A1(n222), .A2(n221), .B(n220), .ZN(n256) );
  VHSR_AOI22_2 U248 ( .A1(a[6]), .A2(b[0]), .B1(b[2]), .B2(a[4]), .ZN(n264) );
  VHSR_NAND3_2 U249 ( .A1(b[1]), .A2(a[5]), .A3(n339), .ZN(n263) );
  VHSR_CLKNAND2_2 U250 ( .A1(a[6]), .A2(b[2]), .ZN(n223) );
  VHSR_OAI22_2 U251 ( .A1(n264), .A2(n263), .B1(n339), .B2(n223), .ZN(n262) );
  VHSR_CLKNAND2_2 U252 ( .A1(n256), .A2(n262), .ZN(n255) );
  VHSR_CLKNAND2_2 U253 ( .A1(n224), .A2(n255), .ZN(n250) );
  VHSR_CLKNAND2_2 U254 ( .A1(n251), .A2(n250), .ZN(n249) );
  VHSR_OAI211_2 U255 ( .A1(n244), .A2(n242), .B(b[3]), .C(a[6]), .ZN(n234) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n276) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n236) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n277) );
  VHSR_IN_2 U259 ( .I(a[2]), .ZN(n315) );
  VHSR_OAI22_2 U260 ( .A1(n277), .A2(n236), .B1(n276), .B2(n315), .ZN(n241) );
  VHSR_AOI22_2 U261 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[4]), .A2(a[2]), .ZN(n261) );
  VHSR_NAND3_2 U263 ( .A1(a[3]), .A2(b[5]), .A3(n261), .ZN(n231) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[7]), .A2(a[2]), .ZN(n226) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[1]), .ZN(n228) );
  VHSR_OAI22_2 U266 ( .A1(n232), .A2(n231), .B1(n226), .B2(n228), .ZN(n233) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[0]), .ZN(n260) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[4]), .A2(a[0]), .ZN(n388) );
  VHSR_NAND3_2 U269 ( .A1(a[1]), .A2(b[5]), .A3(n388), .ZN(n259) );
  VHSR_MAOI222_2 U270 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_IN_2 U271 ( .I(b[5]), .ZN(n282) );
  VHSR_IN_2 U272 ( .I(a[1]), .ZN(n311) );
  VHSR_NOR3_2 U273 ( .A1(n282), .A2(n311), .A3(n388), .ZN(n265) );
  VHSR_NAND4_2 U274 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_IN_2 U275 ( .I(b[4]), .ZN(n273) );
  VHSR_OAI22_2 U276 ( .A1(n273), .A2(n236), .B1(n282), .B2(n315), .ZN(n227) );
  VHSR_AND2_2 U277 ( .A1(n238), .A2(n227), .Z(n230) );
  VHSR_OAI21_2 U278 ( .A1(n276), .A2(n395), .B(n228), .ZN(n229) );
  VHSR_AND2_2 U279 ( .A1(n258), .A2(n254), .Z(n253) );
  VHSR_AD1_1 U280 ( .A(n265), .B(n230), .CI(n229), .CO(n245), .S(n254) );
  VHSR_AOI21_2 U281 ( .A1(n232), .A2(n231), .B(n233), .ZN(n248) );
  VHSR_OAI32_2 U282 ( .A1(n233), .A2(n253), .A3(n245), .B1(n248), .B2(n233), 
        .ZN(n239) );
  VHSR_CLKNAND2_2 U283 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U284 ( .A1(n241), .A2(n237), .ZN(n235) );
  VHSR_NOR3_2 U285 ( .A1(n276), .A2(n236), .A3(n235), .ZN(n293) );
  VHSR_OAI32_2 U286 ( .A1(n294), .A2(n274), .A3(n312), .B1(n234), .B2(n294), 
        .ZN(n297) );
  VHSR_OAI32_2 U287 ( .A1(n293), .A2(n236), .A3(n276), .B1(n235), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U288 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U289 ( .A1(n241), .A2(n240), .ZN(n304) );
  VHSR_CLKNAND2_2 U290 ( .A1(b[3]), .A2(a[6]), .ZN(n243) );
  VHSR_XNOR3_2 U291 ( .A1(n244), .A2(n243), .A3(n242), .ZN(n303) );
  VHSR_NOR2_1 U292 ( .A1(n253), .A2(n245), .ZN(n247) );
  VHSR_AOI22_2 U293 ( .A1(n253), .A2(n245), .B1(n248), .B2(n247), .ZN(n246) );
  VHSR_OAI21_2 U294 ( .A1(n248), .A2(n247), .B(n246), .ZN(n309) );
  VHSR_OAI21_2 U295 ( .A1(n251), .A2(n250), .B(n249), .ZN(n252) );
  VHSR_IN_2 U296 ( .I(n252), .ZN(n308) );
  VHSR_IAO21_2 U297 ( .A1(n258), .A2(n254), .B(n253), .ZN(n322) );
  VHSR_OAI21_2 U298 ( .A1(n256), .A2(n262), .B(n255), .ZN(n257) );
  VHSR_IN_2 U299 ( .I(n257), .ZN(n321) );
  VHSR_AOI31_2 U300 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n329) );
  VHSR_AOI21_2 U301 ( .A1(n264), .A2(n263), .B(n262), .ZN(n328) );
  VHSR_AOI22_2 U302 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n266) );
  VHSR_NOR2_1 U303 ( .A1(n266), .A2(n265), .ZN(n334) );
  VHSR_NOR2_1 U304 ( .A1(n339), .A2(n388), .ZN(n338) );
  VHSR_AOI22_2 U305 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n267) );
  VHSR_NOR2_1 U306 ( .A1(n268), .A2(n267), .ZN(n333) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[6]), .A2(b[6]), .ZN(n356) );
  VHSR_IN_2 U308 ( .I(n356), .ZN(n383) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[4]), .ZN(n301) );
  VHSR_NAND3_2 U310 ( .A1(a[7]), .A2(b[5]), .A3(n301), .ZN(n270) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[4]), .A2(b[6]), .ZN(n300) );
  VHSR_NAND3_2 U312 ( .A1(b[7]), .A2(a[5]), .A3(n300), .ZN(n269) );
  VHSR_CLKNAND2_2 U313 ( .A1(n270), .A2(n269), .ZN(n272) );
  VHSR_MAOI222_2 U314 ( .A(n356), .B(n270), .C(n269), .ZN(n340) );
  VHSR_IN_2 U315 ( .I(n340), .ZN(n271) );
  VHSR_OAI21_2 U316 ( .A1(n383), .A2(n272), .B(n271), .ZN(n288) );
  VHSR_NOR3_2 U317 ( .A1(n274), .A2(n301), .A3(n282), .ZN(n348) );
  VHSR_AOI22_2 U318 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n275) );
  VHSR_NOR2_1 U319 ( .A1(n348), .A2(n275), .ZN(n284) );
  VHSR_NOR4_2 U320 ( .A1(n278), .A2(n281), .A3(n277), .A4(n276), .ZN(n346) );
  VHSR_AOI22_2 U321 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n279) );
  VHSR_NOR2_1 U322 ( .A1(n346), .A2(n279), .ZN(n283) );
  VHSR_IN_2 U323 ( .I(n280), .ZN(n291) );
  VHSR_OR3_2 U324 ( .A1(n362), .A2(n282), .A3(n281), .Z(n299) );
  VHSR_MAOI222_2 U325 ( .A(n301), .B(n300), .C(n299), .ZN(n298) );
  VHSR_IN_2 U326 ( .I(n298), .ZN(n290) );
  VHSR_AD1_1 U327 ( .A(n305), .B(n284), .CI(n283), .CO(n285), .S(n280) );
  VHSR_NOR2_1 U328 ( .A1(n289), .A2(n285), .ZN(n287) );
  VHSR_CLKNAND2_2 U329 ( .A1(n289), .A2(n285), .ZN(n286) );
  VHSR_NOR2_1 U330 ( .A1(n287), .A2(n288), .ZN(n341) );
  VHSR_AOI22_2 U331 ( .A1(n288), .A2(n287), .B1(n286), .B2(n341), .ZN(n381) );
  VHSR_AOI21_2 U332 ( .A1(n291), .A2(n290), .B(n289), .ZN(n379) );
  VHSR_AD1_1 U333 ( .A(n294), .B(n293), .CI(n292), .CO(n382), .S(n378) );
  VHSR_AD1_1 U334 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n376) );
  VHSR_AOI31_2 U335 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n375) );
  VHSR_AD1_1 U336 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n360) );
  VHSR_AOI22_2 U337 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n306) );
  VHSR_NOR2_1 U338 ( .A1(n306), .A2(n305), .ZN(n359) );
  VHSR_AD1_1 U339 ( .A(n309), .B(n308), .CI(n307), .CO(n302), .S(n363) );
  VHSR_NOR2_1 U340 ( .A1(n394), .A2(n315), .ZN(n326) );
  VHSR_IN_2 U341 ( .I(n326), .ZN(n319) );
  VHSR_CLKNAND2_2 U342 ( .A1(b[3]), .A2(a[3]), .ZN(n325) );
  VHSR_AOI22_2 U343 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n310) );
  VHSR_IAO21_2 U344 ( .A1(n319), .A2(n325), .B(n310), .ZN(n332) );
  VHSR_AOI22_2 U345 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n317) );
  VHSR_CLKNAND2_2 U346 ( .A1(b[1]), .A2(a[1]), .ZN(n313) );
  VHSR_OAI22_2 U347 ( .A1(n319), .A2(n317), .B1(n325), .B2(n313), .ZN(n318) );
  VHSR_OAI22_2 U348 ( .A1(n312), .A2(n395), .B1(n394), .B2(n311), .ZN(n373) );
  VHSR_IN_2 U349 ( .I(n313), .ZN(n314) );
  VHSR_AOI21_2 U350 ( .A1(a[2]), .A2(b[0]), .B(n314), .ZN(n393) );
  VHSR_NOR3_2 U351 ( .A1(n393), .A2(n395), .A3(n394), .ZN(n396) );
  VHSR_AOI21_2 U352 ( .A1(n317), .A2(n319), .B(n318), .ZN(n336) );
  VHSR_CLKNAND2_2 U353 ( .A1(n337), .A2(n336), .ZN(n335) );
  VHSR_CLKNAND2_2 U354 ( .A1(n332), .A2(n331), .ZN(n323) );
  VHSR_AOI21_2 U355 ( .A1(n319), .A2(n323), .B(n325), .ZN(n366) );
  VHSR_AD1_1 U356 ( .A(n322), .B(n321), .CI(n320), .CO(n307), .S(n365) );
  VHSR_IN_2 U357 ( .I(n323), .ZN(n330) );
  VHSR_CLKNAND2_2 U358 ( .A1(n330), .A2(n325), .ZN(n324) );
  VHSR_OAI31_2 U359 ( .A1(n326), .A2(n330), .A3(n325), .B(n324), .ZN(n369) );
  VHSR_AD1_1 U360 ( .A(n329), .B(n328), .CI(n327), .CO(n320), .S(n368) );
  VHSR_IAO21_2 U361 ( .A1(n332), .A2(n331), .B(n330), .ZN(n371) );
  VHSR_AD1_1 U362 ( .A(n334), .B(n338), .CI(n333), .CO(n327), .S(n370) );
  VHSR_OAI21_2 U363 ( .A1(n337), .A2(n336), .B(n335), .ZN(n391) );
  VHSR_AOI211_2 U364 ( .A1(n339), .A2(n388), .B(n338), .C(n391), .ZN(n390) );
  VHSR_CLKNAND2_2 U365 ( .A1(a[6]), .A2(b[7]), .ZN(n343) );
  VHSR_AOI21_2 U366 ( .A1(a[7]), .A2(b[6]), .B(n343), .ZN(n342) );
  VHSR_AOI31_2 U367 ( .A1(a[7]), .A2(n343), .A3(b[6]), .B(n342), .ZN(n344) );
  VHSR_IN_2 U368 ( .I(n344), .ZN(n345) );
  VHSR_MAOI222_2 U369 ( .A(n348), .B(n346), .C(n345), .ZN(n355) );
  VHSR_OAI21_2 U370 ( .A1(n348), .A2(n347), .B(n355), .ZN(n352) );
  VHSR_CLKXOR2_2 U371 ( .A1(n353), .A2(n352), .Z(n349) );
  VHSR_CLKNAND2_2 U372 ( .A1(n350), .A2(n349), .ZN(n385) );
  VHSR_OAI21_2 U373 ( .A1(n350), .A2(n349), .B(n385), .ZN(n351) );
  VHSR_CLKNAND2_2 U374 ( .A1(a[7]), .A2(b[7]), .ZN(n384) );
  VHSR_NOR2_1 U375 ( .A1(n353), .A2(n352), .ZN(n354) );
  VHSR_AND3_2 U376 ( .A1(n386), .A2(n356), .A3(n385), .Z(n357) );
  VHSR_NOR2_1 U377 ( .A1(n384), .A2(n357), .ZN(product[15]) );
  VHSR_AD1_1 U378 ( .A(n376), .B(n375), .CI(n374), .CO(n377), .S(product[10])
         );
  VHSR_AD1_1 U379 ( .A(n379), .B(n378), .CI(n377), .CO(n380), .S(product[11])
         );
  VHSR_AD1_1 U380 ( .A(n382), .B(n381), .CI(n380), .CO(n350), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U381 ( .A1(n384), .A2(n383), .ZN(n387) );
  VHSR_XOR3_2 U382 ( .A1(n387), .A2(n386), .A3(n385), .Z(product[14]) );
  VHSR_IAO22_2 U383 ( .B1(n389), .B2(n388), .A1(n388), .A2(n389), .ZN(n392) );
  VHSR_AOI21_2 U384 ( .A1(n392), .A2(n391), .B(n390), .ZN(product[4]) );
  VHSR_OAI32_2 U385 ( .A1(n396), .A2(n395), .A3(n394), .B1(n393), .B2(n396), 
        .ZN(product[2]) );
endmodule


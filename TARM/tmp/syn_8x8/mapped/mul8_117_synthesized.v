
module mul8_117 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n210, n211,
         n212, n213, n214, n215, n216, n217, n218, n219, n220, n221, n222,
         n223, n224, n225, n226, n227, n228, n229, n230, n231, n232, n233,
         n234, n235, n236, n237, n238, n239, n240, n241, n242, n243, n244,
         n245, n246, n247, n248, n249, n250, n251, n252, n253, n254, n255,
         n256, n257, n258, n259, n260, n261, n262, n263, n264, n265, n266,
         n267, n268, n269, n270, n271, n272, n273, n274, n275, n276, n277,
         n278, n279, n280, n281, n282, n283, n284, n285, n286, n287, n288,
         n289, n290, n291, n292, n293, n294, n295, n296, n297, n298, n299,
         n300, n301, n302, n303, n304, n305, n306, n307, n308, n309, n310,
         n311, n312, n313, n314, n315, n316, n317, n318, n319, n320, n321,
         n322, n323, n324, n325, n326, n327, n328, n329, n330, n331, n332,
         n333, n334, n335, n336, n337, n338, n339, n340, n341, n342, n343,
         n344, n345, n346, n347, n348, n349, n350, n351, n352, n353, n354,
         n355, n356, n357, n358, n359, n360, n361, n362, n363, n364, n365,
         n366, n367, n368, n369, n370, n371, n372, n373, n374, n375, n376,
         n377, n378, n379, n380, n381, n382, n383, n384, n385, n386, n387,
         n388, n389, n390, n391, n392, n393, n394, n395, n396;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U202 ( .A1(n348), .B1(n346), .ZN(n349) );
  VHSR_NOR2_1 U203 ( .A1(n343), .A2(n342), .ZN(n355) );
  VHSR_INAND2_2 U204 ( .A1(n318), .B1(n336), .ZN(n331) );
  VHSR_NOR2_1 U205 ( .A1(n286), .A2(n287), .ZN(n343) );
  VHSR_NOR2_1 U206 ( .A1(n392), .A2(n391), .ZN(n390) );
  VHSR_INOR2_2 U207 ( .A1(n357), .B1(n356), .ZN(n388) );
  VHSR_IN_2 U208 ( .I(n353), .ZN(product[13]) );
  VHSR_INOR2_1 U209 ( .A1(n233), .B1(n274), .ZN(n292) );
  VHSR_NOR2_2 U210 ( .A1(n288), .A2(n284), .ZN(n286) );
  VHSR_AD1_1 U211 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(product[6])
         );
  VHSR_AD1_1 U212 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(product[9])
         );
  VHSR_AD1_1 U213 ( .A(n378), .B(n396), .CI(n377), .CO(n338), .S(product[3])
         );
  VHSR_AD1_1 U214 ( .A(n376), .B(n375), .CI(n390), .CO(n372), .S(product[5])
         );
  VHSR_AD1_1 U215 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U216 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U217 ( .A(n362), .B(n361), .CI(n360), .CO(n379), .S(product[10])
         );
  VHSR_IN_2 U218 ( .I(a[1]), .ZN(n310) );
  VHSR_IN_2 U219 ( .I(b[0]), .ZN(n313) );
  VHSR_NOR2_1 U220 ( .A1(n310), .A2(n313), .ZN(product[1]) );
  VHSR_IN_2 U221 ( .I(a[0]), .ZN(n394) );
  VHSR_IN_2 U222 ( .I(b[1]), .ZN(n315) );
  VHSR_NOR2_1 U223 ( .A1(n394), .A2(n315), .ZN(product[0]) );
  VHSR_IN_2 U224 ( .I(a[4]), .ZN(n277) );
  VHSR_IN_2 U225 ( .I(a[5]), .ZN(n276) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n309) );
  VHSR_IN_2 U227 ( .I(b[2]), .ZN(n395) );
  VHSR_NOR4_2 U228 ( .A1(n277), .A2(n276), .A3(n309), .A4(n395), .ZN(n243) );
  VHSR_IN_2 U229 ( .I(a[7]), .ZN(n274) );
  VHSR_NOR2_1 U230 ( .A1(n274), .A2(n315), .ZN(n211) );
  VHSR_AOI211_2 U231 ( .A1(a[4]), .A2(b[2]), .B(n276), .C(n309), .ZN(n212) );
  VHSR_CLKNAND2_2 U232 ( .A1(a[6]), .A2(b[2]), .ZN(n214) );
  VHSR_IN_2 U233 ( .I(n214), .ZN(n210) );
  VHSR_MAOI222_2 U234 ( .A(n211), .B(n212), .C(n210), .ZN(n224) );
  VHSR_AOI21_2 U235 ( .A1(b[1]), .A2(a[7]), .B(n212), .ZN(n215) );
  VHSR_IN_2 U236 ( .I(n224), .ZN(n213) );
  VHSR_AOI21_2 U237 ( .A1(n215), .A2(n214), .B(n213), .ZN(n250) );
  VHSR_CLKNAND2_2 U238 ( .A1(a[6]), .A2(b[1]), .ZN(n221) );
  VHSR_IN_2 U239 ( .I(n221), .ZN(n218) );
  VHSR_NOR4_2 U240 ( .A1(n277), .A2(n276), .A3(n315), .A4(n313), .ZN(n268) );
  VHSR_CLKNAND2_2 U241 ( .A1(a[5]), .A2(b[2]), .ZN(n217) );
  VHSR_CLKNAND2_2 U242 ( .A1(a[4]), .A2(b[3]), .ZN(n216) );
  VHSR_AOI21_2 U243 ( .A1(n217), .A2(n216), .B(n243), .ZN(n219) );
  VHSR_MAOI222_2 U244 ( .A(n218), .B(n268), .C(n219), .ZN(n223) );
  VHSR_CLKNAND2_2 U245 ( .A1(a[4]), .A2(b[2]), .ZN(n264) );
  VHSR_OAI21_2 U246 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n263) );
  VHSR_OAI211_2 U247 ( .A1(n277), .A2(n313), .B(a[5]), .C(b[1]), .ZN(n262) );
  VHSR_MAOI222_2 U248 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_NOR2_1 U249 ( .A1(n268), .A2(n219), .ZN(n222) );
  VHSR_IN_2 U250 ( .I(n223), .ZN(n220) );
  VHSR_AOI21_2 U251 ( .A1(n222), .A2(n221), .B(n220), .ZN(n253) );
  VHSR_CLKNAND2_2 U252 ( .A1(n261), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U253 ( .A1(n223), .A2(n252), .ZN(n249) );
  VHSR_CLKNAND2_2 U254 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_CLKNAND2_2 U255 ( .A1(n224), .A2(n248), .ZN(n242) );
  VHSR_AND2_2 U256 ( .A1(a[6]), .A2(b[3]), .Z(n241) );
  VHSR_IN_2 U257 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U258 ( .I(a[3]), .ZN(n314) );
  VHSR_IN_2 U259 ( .I(b[6]), .ZN(n279) );
  VHSR_IN_2 U260 ( .I(a[2]), .ZN(n316) );
  VHSR_OAI22_2 U261 ( .A1(n279), .A2(n314), .B1(n278), .B2(n316), .ZN(n240) );
  VHSR_AOI22_2 U262 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n231) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[4]), .A2(a[2]), .ZN(n260) );
  VHSR_NAND3_2 U264 ( .A1(a[3]), .A2(b[5]), .A3(n260), .ZN(n230) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[7]), .A2(a[2]), .ZN(n225) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[6]), .A2(a[1]), .ZN(n227) );
  VHSR_OAI22_2 U267 ( .A1(n231), .A2(n230), .B1(n225), .B2(n227), .ZN(n232) );
  VHSR_IN_2 U268 ( .I(b[4]), .ZN(n339) );
  VHSR_OAI211_2 U269 ( .A1(n339), .A2(n394), .B(b[5]), .C(a[1]), .ZN(n259) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[0]), .ZN(n258) );
  VHSR_MAOI222_2 U271 ( .A(n260), .B(n259), .C(n258), .ZN(n257) );
  VHSR_NAND4_2 U272 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n237) );
  VHSR_IN_2 U273 ( .I(b[5]), .ZN(n273) );
  VHSR_OAI22_2 U274 ( .A1(n339), .A2(n314), .B1(n273), .B2(n316), .ZN(n226) );
  VHSR_AND2_2 U275 ( .A1(n237), .A2(n226), .Z(n229) );
  VHSR_OAI21_2 U276 ( .A1(n278), .A2(n394), .B(n227), .ZN(n228) );
  VHSR_NOR4_2 U277 ( .A1(n339), .A2(n273), .A3(n310), .A4(n394), .ZN(n266) );
  VHSR_AND2_2 U278 ( .A1(n257), .A2(n256), .Z(n255) );
  VHSR_AD1_1 U279 ( .A(n229), .B(n228), .CI(n266), .CO(n244), .S(n256) );
  VHSR_AOI21_2 U280 ( .A1(n231), .A2(n230), .B(n232), .ZN(n247) );
  VHSR_OAI32_2 U281 ( .A1(n232), .A2(n255), .A3(n244), .B1(n247), .B2(n232), 
        .ZN(n238) );
  VHSR_CLKNAND2_2 U282 ( .A1(n238), .A2(n237), .ZN(n236) );
  VHSR_CLKNAND2_2 U283 ( .A1(n240), .A2(n236), .ZN(n235) );
  VHSR_NOR3_2 U284 ( .A1(n278), .A2(n314), .A3(n235), .ZN(n291) );
  VHSR_NOR2_1 U285 ( .A1(n274), .A2(n309), .ZN(n234) );
  VHSR_IAO21_2 U286 ( .A1(n234), .A2(n233), .B(n292), .ZN(n295) );
  VHSR_OAI32_2 U287 ( .A1(n291), .A2(n314), .A3(n278), .B1(n235), .B2(n291), 
        .ZN(n294) );
  VHSR_OAI21_2 U288 ( .A1(n238), .A2(n237), .B(n236), .ZN(n239) );
  VHSR_XNOR2_2 U289 ( .A1(n240), .A2(n239), .ZN(n302) );
  VHSR_AD1_1 U290 ( .A(n243), .B(n242), .CI(n241), .CO(n233), .S(n301) );
  VHSR_NOR2_1 U291 ( .A1(n255), .A2(n244), .ZN(n246) );
  VHSR_AOI22_2 U292 ( .A1(n255), .A2(n244), .B1(n247), .B2(n246), .ZN(n245) );
  VHSR_OAI21_2 U293 ( .A1(n247), .A2(n246), .B(n245), .ZN(n307) );
  VHSR_OAI21_2 U294 ( .A1(n250), .A2(n249), .B(n248), .ZN(n251) );
  VHSR_IN_2 U295 ( .I(n251), .ZN(n306) );
  VHSR_OAI21_2 U296 ( .A1(n261), .A2(n253), .B(n252), .ZN(n254) );
  VHSR_IN_2 U297 ( .I(n254), .ZN(n322) );
  VHSR_IAO21_2 U298 ( .A1(n257), .A2(n256), .B(n255), .ZN(n321) );
  VHSR_AOI31_2 U299 ( .A1(n260), .A2(n259), .A3(n258), .B(n257), .ZN(n329) );
  VHSR_AOI31_2 U300 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n328) );
  VHSR_CLKNAND2_2 U301 ( .A1(b[5]), .A2(a[0]), .ZN(n265) );
  VHSR_OAI32_2 U302 ( .A1(n266), .A2(n310), .A3(n339), .B1(n265), .B2(n266), 
        .ZN(n335) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[4]), .A2(a[4]), .ZN(n281) );
  VHSR_IN_2 U304 ( .I(n281), .ZN(n367) );
  VHSR_NAND3_2 U305 ( .A1(n367), .A2(a[0]), .A3(b[0]), .ZN(n341) );
  VHSR_IN_2 U306 ( .I(n341), .ZN(n334) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[5]), .A2(b[0]), .ZN(n267) );
  VHSR_OAI32_2 U308 ( .A1(n268), .A2(n315), .A3(n277), .B1(n267), .B2(n268), 
        .ZN(n333) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[6]), .ZN(n358) );
  VHSR_IN_2 U310 ( .I(n358), .ZN(n385) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[4]), .ZN(n299) );
  VHSR_NAND3_2 U312 ( .A1(a[7]), .A2(b[5]), .A3(n299), .ZN(n270) );
  VHSR_CLKNAND2_2 U313 ( .A1(b[6]), .A2(a[4]), .ZN(n298) );
  VHSR_NAND3_2 U314 ( .A1(b[7]), .A2(a[5]), .A3(n298), .ZN(n269) );
  VHSR_CLKNAND2_2 U315 ( .A1(n270), .A2(n269), .ZN(n272) );
  VHSR_MAOI222_2 U316 ( .A(n358), .B(n270), .C(n269), .ZN(n342) );
  VHSR_IN_2 U317 ( .I(n342), .ZN(n271) );
  VHSR_OAI21_2 U318 ( .A1(n385), .A2(n272), .B(n271), .ZN(n287) );
  VHSR_NOR3_2 U319 ( .A1(n273), .A2(n276), .A3(n281), .ZN(n303) );
  VHSR_NOR3_2 U320 ( .A1(n274), .A2(n299), .A3(n273), .ZN(n350) );
  VHSR_AOI22_2 U321 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n275) );
  VHSR_NOR2_1 U322 ( .A1(n350), .A2(n275), .ZN(n283) );
  VHSR_NOR4_2 U323 ( .A1(n279), .A2(n278), .A3(n277), .A4(n276), .ZN(n348) );
  VHSR_AOI22_2 U324 ( .A1(b[6]), .A2(a[5]), .B1(b[7]), .B2(a[4]), .ZN(n280) );
  VHSR_NOR2_1 U325 ( .A1(n348), .A2(n280), .ZN(n282) );
  VHSR_NAND3_2 U326 ( .A1(a[5]), .A2(b[5]), .A3(n281), .ZN(n297) );
  VHSR_MAOI222_2 U327 ( .A(n299), .B(n298), .C(n297), .ZN(n296) );
  VHSR_AND2_2 U328 ( .A1(n289), .A2(n296), .Z(n288) );
  VHSR_AD1_1 U329 ( .A(n303), .B(n283), .CI(n282), .CO(n284), .S(n289) );
  VHSR_CLKNAND2_2 U330 ( .A1(n288), .A2(n284), .ZN(n285) );
  VHSR_AOI22_2 U331 ( .A1(n287), .A2(n286), .B1(n285), .B2(n343), .ZN(n383) );
  VHSR_IAO21_2 U332 ( .A1(n289), .A2(n296), .B(n288), .ZN(n381) );
  VHSR_AD1_1 U333 ( .A(n292), .B(n291), .CI(n290), .CO(n384), .S(n380) );
  VHSR_AD1_1 U334 ( .A(n295), .B(n294), .CI(n293), .CO(n290), .S(n362) );
  VHSR_AOI31_2 U335 ( .A1(n299), .A2(n298), .A3(n297), .B(n296), .ZN(n361) );
  VHSR_AD1_1 U336 ( .A(n302), .B(n301), .CI(n300), .CO(n293), .S(n365) );
  VHSR_AOI22_2 U337 ( .A1(b[4]), .A2(a[5]), .B1(b[5]), .B2(a[4]), .ZN(n304) );
  VHSR_NOR2_1 U338 ( .A1(n304), .A2(n303), .ZN(n364) );
  VHSR_AD1_1 U339 ( .A(n307), .B(n306), .CI(n305), .CO(n300), .S(n368) );
  VHSR_NOR2_1 U340 ( .A1(n316), .A2(n395), .ZN(n326) );
  VHSR_IN_2 U341 ( .I(n326), .ZN(n319) );
  VHSR_CLKNAND2_2 U342 ( .A1(a[3]), .A2(b[3]), .ZN(n325) );
  VHSR_AOI22_2 U343 ( .A1(a[2]), .A2(b[3]), .B1(a[3]), .B2(b[2]), .ZN(n308) );
  VHSR_IAO21_2 U344 ( .A1(n319), .A2(n325), .B(n308), .ZN(n332) );
  VHSR_AOI22_2 U345 ( .A1(a[3]), .A2(b[1]), .B1(a[1]), .B2(b[3]), .ZN(n317) );
  VHSR_CLKNAND2_2 U346 ( .A1(a[1]), .A2(b[1]), .ZN(n311) );
  VHSR_OAI22_2 U347 ( .A1(n319), .A2(n317), .B1(n325), .B2(n311), .ZN(n318) );
  VHSR_OAI22_2 U348 ( .A1(n310), .A2(n395), .B1(n394), .B2(n309), .ZN(n378) );
  VHSR_IN_2 U349 ( .I(n311), .ZN(n312) );
  VHSR_AOI21_2 U350 ( .A1(b[0]), .A2(a[2]), .B(n312), .ZN(n393) );
  VHSR_NOR3_2 U351 ( .A1(n393), .A2(n395), .A3(n394), .ZN(n396) );
  VHSR_OAI22_2 U352 ( .A1(n316), .A2(n315), .B1(n314), .B2(n313), .ZN(n377) );
  VHSR_AOI21_2 U353 ( .A1(n317), .A2(n319), .B(n318), .ZN(n337) );
  VHSR_CLKNAND2_2 U354 ( .A1(n338), .A2(n337), .ZN(n336) );
  VHSR_CLKNAND2_2 U355 ( .A1(n332), .A2(n331), .ZN(n323) );
  VHSR_AOI21_2 U356 ( .A1(n319), .A2(n323), .B(n325), .ZN(n371) );
  VHSR_AD1_1 U357 ( .A(n322), .B(n321), .CI(n320), .CO(n305), .S(n370) );
  VHSR_IN_2 U358 ( .I(n323), .ZN(n330) );
  VHSR_CLKNAND2_2 U359 ( .A1(n330), .A2(n325), .ZN(n324) );
  VHSR_OAI31_2 U360 ( .A1(n326), .A2(n330), .A3(n325), .B(n324), .ZN(n374) );
  VHSR_AD1_1 U361 ( .A(n329), .B(n328), .CI(n327), .CO(n320), .S(n373) );
  VHSR_IAO21_2 U362 ( .A1(n332), .A2(n331), .B(n330), .ZN(n376) );
  VHSR_AD1_1 U363 ( .A(n335), .B(n334), .CI(n333), .CO(n327), .S(n375) );
  VHSR_OAI21_2 U364 ( .A1(n338), .A2(n337), .B(n336), .ZN(n392) );
  VHSR_NOR2_1 U365 ( .A1(n339), .A2(n394), .ZN(n340) );
  VHSR_AOI32_2 U366 ( .A1(a[4]), .A2(n341), .A3(b[0]), .B1(n340), .B2(n341), 
        .ZN(n391) );
  VHSR_CLKNAND2_2 U367 ( .A1(b[6]), .A2(a[7]), .ZN(n345) );
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
  VHSR_AD1_1 U380 ( .A(n381), .B(n380), .CI(n379), .CO(n382), .S(product[11])
         );
  VHSR_AD1_1 U381 ( .A(n384), .B(n383), .CI(n382), .CO(n352), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U382 ( .A1(n386), .A2(n385), .ZN(n389) );
  VHSR_XOR3_2 U383 ( .A1(n389), .A2(n388), .A3(n387), .Z(product[14]) );
  VHSR_AOI21_2 U384 ( .A1(n392), .A2(n391), .B(n390), .ZN(product[4]) );
  VHSR_OAI32_2 U385 ( .A1(n396), .A2(n395), .A3(n394), .B1(n393), .B2(n396), 
        .ZN(product[2]) );
endmodule


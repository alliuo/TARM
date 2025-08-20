
module mul8_105 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n209, n210, n211, n212, n213,
         n214, n215, n216, n217, n218, n219, n220, n221, n222, n223, n224,
         n225, n226, n227, n228, n229, n230, n231, n232, n233, n234, n235,
         n236, n237, n238, n239, n240, n241, n242, n243, n244, n245, n246,
         n247, n248, n249, n250, n251, n252, n253, n254, n255, n256, n257,
         n258, n259, n260, n261, n262, n263, n264, n265, n266, n267, n268,
         n269, n270, n271, n272, n273, n274, n275, n276, n277, n278, n279,
         n280, n281, n282, n283, n284, n285, n286, n287, n288, n289, n290,
         n291, n292, n293, n294, n295, n296, n297, n298, n299, n300, n301,
         n302, n303, n304, n305, n306, n307, n308, n309, n310, n311, n312,
         n313, n314, n315, n316, n317, n318, n319, n320, n321, n322, n323,
         n324, n325, n326, n327, n328, n329, n330, n331, n332, n333, n334,
         n335, n336, n337, n338, n339, n340, n341, n342, n343, n344, n345,
         n346, n347, n348, n349, n350, n351, n352, n353, n354, n355, n356,
         n357, n358, n359, n360, n361, n362, n363, n364, n365, n366, n367,
         n368, n369, n370, n371, n372, n373, n374, n375, n376, n377, n378,
         n379, n380, n381, n382, n383, n384, n385, n386, n387, n388, n389,
         n390, n391, n392, n393, n394, n395, n396, n397, n398;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U201 ( .A1(n256), .B1(n277), .ZN(n264) );
  VHSR_IOA21_2 U202 ( .A1(n255), .A2(n254), .B(n256), .ZN(n280) );
  VHSR_NOR2_1 U203 ( .A1(n271), .A2(n267), .ZN(n259) );
  VHSR_INAND2_2 U204 ( .A1(n330), .B1(n342), .ZN(n346) );
  VHSR_INOR3_2 U205 ( .A1(n259), .B1(n324), .B2(n257), .ZN(n307) );
  VHSR_IN_2 U206 ( .I(n351), .ZN(product[15]) );
  VHSR_NOR2_2 U207 ( .A1(n269), .A2(n268), .ZN(n267) );
  VHSR_NOR2_2 U208 ( .A1(n237), .A2(n250), .ZN(n382) );
  VHSR_NOR2_2 U209 ( .A1(n387), .A2(n386), .ZN(n385) );
  VHSR_AD1_1 U210 ( .A(n363), .B(n362), .CI(n361), .CO(n358), .S(product[9])
         );
  VHSR_AD1_1 U211 ( .A(n357), .B(n356), .CI(n355), .CO(n352), .S(product[11])
         );
  VHSR_AD1_1 U212 ( .A(n373), .B(n398), .CI(n372), .CO(n344), .S(product[3])
         );
  VHSR_AD1_1 U213 ( .A(n389), .B(n371), .CI(n370), .CO(n374), .S(product[5])
         );
  VHSR_AD1_1 U214 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U215 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U216 ( .A(n360), .B(n359), .CI(n358), .CO(n355), .S(product[10])
         );
  VHSR_AD1_1 U217 ( .A(n354), .B(n353), .CI(n352), .CO(n377), .S(product[12])
         );
  VHSR_IN_2 U218 ( .I(b[0]), .ZN(n393) );
  VHSR_IN_2 U219 ( .I(a[0]), .ZN(n397) );
  VHSR_NOR2_1 U220 ( .A1(n393), .A2(n397), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U221 ( .A1(a[7]), .A2(b[7]), .ZN(n380) );
  VHSR_IN_2 U222 ( .I(n380), .ZN(n350) );
  VHSR_IN_2 U223 ( .I(a[6]), .ZN(n237) );
  VHSR_IN_2 U224 ( .I(b[6]), .ZN(n250) );
  VHSR_AOI22_2 U225 ( .A1(a[6]), .A2(b[7]), .B1(a[7]), .B2(b[6]), .ZN(n209) );
  VHSR_AOI21_2 U226 ( .A1(n350), .A2(n382), .B(n209), .ZN(n230) );
  VHSR_IN_2 U227 ( .I(b[4]), .ZN(n248) );
  VHSR_NOR2_1 U228 ( .A1(n237), .A2(n248), .ZN(n219) );
  VHSR_IN_2 U229 ( .I(a[5]), .ZN(n238) );
  VHSR_IN_2 U230 ( .I(b[7]), .ZN(n261) );
  VHSR_CLKNAND2_2 U231 ( .A1(a[4]), .A2(b[6]), .ZN(n314) );
  VHSR_NOR3_2 U232 ( .A1(n238), .A2(n261), .A3(n314), .ZN(n217) );
  VHSR_AOI31_2 U233 ( .A1(b[5]), .A2(a[7]), .A3(n219), .B(n217), .ZN(n210) );
  VHSR_IN_2 U234 ( .I(n210), .ZN(n229) );
  VHSR_AOI211_2 U235 ( .A1(a[4]), .A2(b[6]), .B(n238), .C(n261), .ZN(n213) );
  VHSR_IN_2 U236 ( .I(a[7]), .ZN(n257) );
  VHSR_IN_2 U237 ( .I(b[5]), .ZN(n251) );
  VHSR_AOI211_2 U238 ( .A1(a[6]), .A2(b[4]), .B(n257), .C(n251), .ZN(n211) );
  VHSR_MAOI222_2 U239 ( .A(n213), .B(n382), .C(n211), .ZN(n227) );
  VHSR_IN_2 U240 ( .I(n219), .ZN(n313) );
  VHSR_AOI31_2 U241 ( .A1(b[5]), .A2(a[7]), .A3(n313), .B(n382), .ZN(n212) );
  VHSR_IN_2 U242 ( .I(n212), .ZN(n214) );
  VHSR_OAI21_2 U243 ( .A1(n214), .A2(n213), .B(n227), .ZN(n215) );
  VHSR_IN_2 U244 ( .I(n215), .ZN(n300) );
  VHSR_AOI22_2 U245 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n216) );
  VHSR_NOR2_1 U246 ( .A1(n217), .A2(n216), .ZN(n225) );
  VHSR_AND2_2 U247 ( .A1(a[4]), .A2(b[4]), .Z(n366) );
  VHSR_NAND3_2 U248 ( .A1(a[5]), .A2(b[5]), .A3(n366), .ZN(n221) );
  VHSR_IN_2 U249 ( .I(n221), .ZN(n319) );
  VHSR_AOI22_2 U250 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n218) );
  VHSR_AOI31_2 U251 ( .A1(b[5]), .A2(a[7]), .A3(n219), .B(n218), .ZN(n220) );
  VHSR_MAOI222_2 U252 ( .A(n225), .B(n319), .C(n220), .ZN(n226) );
  VHSR_OR3_2 U253 ( .A1(n366), .A2(n251), .A3(n238), .Z(n312) );
  VHSR_MAOI222_2 U254 ( .A(n314), .B(n313), .C(n312), .ZN(n311) );
  VHSR_IN_2 U255 ( .I(n220), .ZN(n222) );
  VHSR_CLKNAND2_2 U256 ( .A1(n221), .A2(n222), .ZN(n224) );
  VHSR_OAI22_2 U257 ( .A1(n225), .A2(n224), .B1(n222), .B2(n221), .ZN(n223) );
  VHSR_AOI21_2 U258 ( .A1(n225), .A2(n224), .B(n223), .ZN(n303) );
  VHSR_CLKNAND2_2 U259 ( .A1(n311), .A2(n303), .ZN(n302) );
  VHSR_CLKNAND2_2 U260 ( .A1(n226), .A2(n302), .ZN(n299) );
  VHSR_CLKNAND2_2 U261 ( .A1(n300), .A2(n299), .ZN(n298) );
  VHSR_CLKNAND2_2 U262 ( .A1(n227), .A2(n298), .ZN(n228) );
  VHSR_AD1_1 U263 ( .A(n230), .B(n229), .CI(n228), .CO(n381), .S(n378) );
  VHSR_AOI22_2 U264 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n271) );
  VHSR_IN_2 U265 ( .I(b[3]), .ZN(n324) );
  VHSR_CLKNAND2_2 U266 ( .A1(b[2]), .A2(a[4]), .ZN(n289) );
  VHSR_NOR3_2 U267 ( .A1(n324), .A2(n289), .A3(n238), .ZN(n269) );
  VHSR_IN_2 U268 ( .I(b[1]), .ZN(n394) );
  VHSR_NOR2_1 U269 ( .A1(n257), .A2(n394), .ZN(n232) );
  VHSR_AOI211_2 U270 ( .A1(a[4]), .A2(b[2]), .B(n324), .C(n238), .ZN(n233) );
  VHSR_CLKNAND2_2 U271 ( .A1(a[6]), .A2(b[2]), .ZN(n235) );
  VHSR_IN_2 U272 ( .I(n235), .ZN(n231) );
  VHSR_MAOI222_2 U273 ( .A(n232), .B(n233), .C(n231), .ZN(n245) );
  VHSR_AOI21_2 U274 ( .A1(b[1]), .A2(a[7]), .B(n233), .ZN(n236) );
  VHSR_IN_2 U275 ( .I(n245), .ZN(n234) );
  VHSR_AOI21_2 U276 ( .A1(n236), .A2(n235), .B(n234), .ZN(n274) );
  VHSR_NOR2_1 U277 ( .A1(n237), .A2(n394), .ZN(n241) );
  VHSR_CLKNAND2_2 U278 ( .A1(a[4]), .A2(b[0]), .ZN(n387) );
  VHSR_NOR3_2 U279 ( .A1(n238), .A2(n394), .A3(n387), .ZN(n297) );
  VHSR_AOI22_2 U280 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n239) );
  VHSR_NOR2_1 U281 ( .A1(n269), .A2(n239), .ZN(n240) );
  VHSR_MAOI222_2 U282 ( .A(n241), .B(n297), .C(n240), .ZN(n244) );
  VHSR_NAND3_2 U283 ( .A1(b[1]), .A2(a[5]), .A3(n387), .ZN(n288) );
  VHSR_OAI21_2 U284 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n287) );
  VHSR_MAOI222_2 U285 ( .A(n289), .B(n288), .C(n287), .ZN(n286) );
  VHSR_OR2_2 U286 ( .A1(n297), .A2(n240), .Z(n242) );
  VHSR_OAI21_2 U287 ( .A1(n242), .A2(n241), .B(n244), .ZN(n243) );
  VHSR_IN_2 U288 ( .I(n243), .ZN(n282) );
  VHSR_CLKNAND2_2 U289 ( .A1(n286), .A2(n282), .ZN(n281) );
  VHSR_CLKNAND2_2 U290 ( .A1(n244), .A2(n281), .ZN(n273) );
  VHSR_CLKNAND2_2 U291 ( .A1(n274), .A2(n273), .ZN(n272) );
  VHSR_CLKNAND2_2 U292 ( .A1(n245), .A2(n272), .ZN(n268) );
  VHSR_IN_2 U293 ( .I(a[3]), .ZN(n327) );
  VHSR_IN_2 U294 ( .I(a[2]), .ZN(n328) );
  VHSR_OAI22_2 U295 ( .A1(n328), .A2(n261), .B1(n327), .B2(n250), .ZN(n266) );
  VHSR_NOR2_1 U296 ( .A1(n328), .A2(n261), .ZN(n247) );
  VHSR_IN_2 U297 ( .I(a[1]), .ZN(n392) );
  VHSR_NOR2_1 U298 ( .A1(n250), .A2(n392), .ZN(n246) );
  VHSR_CLKNAND2_2 U299 ( .A1(a[2]), .A2(b[4]), .ZN(n293) );
  VHSR_NAND3_2 U300 ( .A1(a[3]), .A2(b[5]), .A3(n293), .ZN(n254) );
  VHSR_AOI22_2 U301 ( .A1(a[2]), .A2(b[6]), .B1(b[7]), .B2(a[1]), .ZN(n255) );
  VHSR_IAO22_2 U302 ( .B1(n247), .B2(n246), .A1(n254), .A2(n255), .ZN(n256) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[6]), .A2(a[0]), .ZN(n292) );
  VHSR_CLKNAND2_2 U304 ( .A1(b[4]), .A2(a[0]), .ZN(n386) );
  VHSR_NAND3_2 U305 ( .A1(a[1]), .A2(b[5]), .A3(n386), .ZN(n291) );
  VHSR_MAOI222_2 U306 ( .A(n293), .B(n292), .C(n291), .ZN(n290) );
  VHSR_NAND4_2 U307 ( .A1(a[2]), .A2(a[3]), .A3(b[4]), .A4(b[5]), .ZN(n263) );
  VHSR_OAI22_2 U308 ( .A1(n328), .A2(n251), .B1(n327), .B2(n248), .ZN(n249) );
  VHSR_AND2_2 U309 ( .A1(n263), .A2(n249), .Z(n253) );
  VHSR_OAI22_2 U310 ( .A1(n261), .A2(n397), .B1(n250), .B2(n392), .ZN(n252) );
  VHSR_NOR3_2 U311 ( .A1(n251), .A2(n392), .A3(n386), .ZN(n295) );
  VHSR_AND2_2 U312 ( .A1(n290), .A2(n285), .Z(n284) );
  VHSR_AD1_1 U313 ( .A(n253), .B(n252), .CI(n295), .CO(n276), .S(n285) );
  VHSR_NOR2_1 U314 ( .A1(n284), .A2(n276), .ZN(n279) );
  VHSR_NOR2_1 U315 ( .A1(n279), .A2(n280), .ZN(n277) );
  VHSR_CLKNAND2_2 U316 ( .A1(n264), .A2(n263), .ZN(n262) );
  VHSR_CLKNAND2_2 U317 ( .A1(n266), .A2(n262), .ZN(n260) );
  VHSR_NOR3_2 U318 ( .A1(n261), .A2(n327), .A3(n260), .ZN(n306) );
  VHSR_NOR2_1 U319 ( .A1(n324), .A2(n257), .ZN(n258) );
  VHSR_IAO21_2 U320 ( .A1(n259), .A2(n258), .B(n307), .ZN(n310) );
  VHSR_OAI32_2 U321 ( .A1(n306), .A2(n327), .A3(n261), .B1(n260), .B2(n306), 
        .ZN(n309) );
  VHSR_OAI21_2 U322 ( .A1(n264), .A2(n263), .B(n262), .ZN(n265) );
  VHSR_XNOR2_2 U323 ( .A1(n266), .A2(n265), .ZN(n317) );
  VHSR_AOI21_2 U324 ( .A1(n269), .A2(n268), .B(n267), .ZN(n270) );
  VHSR_XNOR2_2 U325 ( .A1(n271), .A2(n270), .ZN(n316) );
  VHSR_OAI21_2 U326 ( .A1(n274), .A2(n273), .B(n272), .ZN(n275) );
  VHSR_IN_2 U327 ( .I(n275), .ZN(n322) );
  VHSR_CLKNAND2_2 U328 ( .A1(n284), .A2(n276), .ZN(n278) );
  VHSR_AOI22_2 U329 ( .A1(n280), .A2(n279), .B1(n278), .B2(n277), .ZN(n321) );
  VHSR_OAI21_2 U330 ( .A1(n286), .A2(n282), .B(n281), .ZN(n283) );
  VHSR_IN_2 U331 ( .I(n283), .ZN(n334) );
  VHSR_IAO21_2 U332 ( .A1(n290), .A2(n285), .B(n284), .ZN(n333) );
  VHSR_AOI31_2 U333 ( .A1(n289), .A2(n288), .A3(n287), .B(n286), .ZN(n337) );
  VHSR_AOI31_2 U334 ( .A1(n293), .A2(n292), .A3(n291), .B(n290), .ZN(n336) );
  VHSR_AOI22_2 U335 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n294) );
  VHSR_NOR2_1 U336 ( .A1(n295), .A2(n294), .ZN(n349) );
  VHSR_AOI22_2 U337 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n296) );
  VHSR_NOR2_1 U338 ( .A1(n297), .A2(n296), .ZN(n348) );
  VHSR_OAI21_2 U339 ( .A1(n300), .A2(n299), .B(n298), .ZN(n301) );
  VHSR_IN_2 U340 ( .I(n301), .ZN(n353) );
  VHSR_OAI21_2 U341 ( .A1(n311), .A2(n303), .B(n302), .ZN(n304) );
  VHSR_IN_2 U342 ( .I(n304), .ZN(n357) );
  VHSR_AD1_1 U343 ( .A(n307), .B(n306), .CI(n305), .CO(n354), .S(n356) );
  VHSR_AD1_1 U344 ( .A(n310), .B(n309), .CI(n308), .CO(n305), .S(n360) );
  VHSR_AOI31_2 U345 ( .A1(n314), .A2(n313), .A3(n312), .B(n311), .ZN(n359) );
  VHSR_AD1_1 U346 ( .A(n317), .B(n316), .CI(n315), .CO(n308), .S(n363) );
  VHSR_AOI22_2 U347 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n318) );
  VHSR_NOR2_1 U348 ( .A1(n319), .A2(n318), .ZN(n362) );
  VHSR_AD1_1 U349 ( .A(n322), .B(n321), .CI(n320), .CO(n315), .S(n365) );
  VHSR_IN_2 U350 ( .I(b[2]), .ZN(n396) );
  VHSR_NOR2_1 U351 ( .A1(n396), .A2(n328), .ZN(n341) );
  VHSR_IN_2 U352 ( .I(n341), .ZN(n331) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[3]), .A2(a[3]), .ZN(n340) );
  VHSR_AOI22_2 U354 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n323) );
  VHSR_IAO21_2 U355 ( .A1(n331), .A2(n340), .B(n323), .ZN(n347) );
  VHSR_AOI22_2 U356 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n329) );
  VHSR_CLKNAND2_2 U357 ( .A1(b[1]), .A2(a[1]), .ZN(n325) );
  VHSR_OAI22_2 U358 ( .A1(n331), .A2(n329), .B1(n340), .B2(n325), .ZN(n330) );
  VHSR_OAI22_2 U359 ( .A1(n324), .A2(n397), .B1(n396), .B2(n392), .ZN(n373) );
  VHSR_OAI21_2 U360 ( .A1(n328), .A2(n393), .B(n325), .ZN(n326) );
  VHSR_IN_2 U361 ( .I(n326), .ZN(n395) );
  VHSR_NOR3_2 U362 ( .A1(n395), .A2(n397), .A3(n396), .ZN(n398) );
  VHSR_OAI22_2 U363 ( .A1(n394), .A2(n328), .B1(n393), .B2(n327), .ZN(n372) );
  VHSR_AOI21_2 U364 ( .A1(n329), .A2(n331), .B(n330), .ZN(n343) );
  VHSR_CLKNAND2_2 U365 ( .A1(n344), .A2(n343), .ZN(n342) );
  VHSR_CLKNAND2_2 U366 ( .A1(n347), .A2(n346), .ZN(n338) );
  VHSR_AOI21_2 U367 ( .A1(n331), .A2(n338), .B(n340), .ZN(n369) );
  VHSR_AD1_1 U368 ( .A(n334), .B(n333), .CI(n332), .CO(n320), .S(n368) );
  VHSR_AD1_1 U369 ( .A(n337), .B(n336), .CI(n335), .CO(n332), .S(n376) );
  VHSR_IN_2 U370 ( .I(n338), .ZN(n345) );
  VHSR_CLKNAND2_2 U371 ( .A1(n345), .A2(n340), .ZN(n339) );
  VHSR_OAI31_2 U372 ( .A1(n341), .A2(n345), .A3(n340), .B(n339), .ZN(n375) );
  VHSR_OAI21_2 U373 ( .A1(n344), .A2(n343), .B(n342), .ZN(n391) );
  VHSR_AOI211_2 U374 ( .A1(n387), .A2(n386), .B(n385), .C(n391), .ZN(n389) );
  VHSR_IAO21_2 U375 ( .A1(n347), .A2(n346), .B(n345), .ZN(n371) );
  VHSR_AD1_1 U376 ( .A(n349), .B(n348), .CI(n385), .CO(n335), .S(n370) );
  VHSR_AND2_2 U377 ( .A1(n378), .A2(n377), .Z(n384) );
  VHSR_OAI31_2 U378 ( .A1(n381), .A2(n384), .A3(n382), .B(n350), .ZN(n351) );
  VHSR_AD1_1 U379 ( .A(n376), .B(n375), .CI(n374), .CO(n367), .S(product[6])
         );
  VHSR_IAO21_2 U380 ( .A1(n378), .A2(n377), .B(n384), .ZN(product[13]) );
  VHSR_OAI21_2 U381 ( .A1(n382), .A2(n380), .B(n381), .ZN(n379) );
  VHSR_OAI31_2 U382 ( .A1(n382), .A2(n381), .A3(n380), .B(n379), .ZN(n383) );
  VHSR_CLKXOR2_2 U383 ( .A1(n384), .A2(n383), .Z(product[14]) );
  VHSR_AOI21_2 U384 ( .A1(n387), .A2(n386), .B(n385), .ZN(n388) );
  VHSR_IN_2 U385 ( .I(n388), .ZN(n390) );
  VHSR_AOI21_2 U386 ( .A1(n391), .A2(n390), .B(n389), .ZN(product[4]) );
  VHSR_OAI22_2 U387 ( .A1(n394), .A2(n397), .B1(n393), .B2(n392), .ZN(
        product[1]) );
  VHSR_OAI32_2 U388 ( .A1(n398), .A2(n397), .A3(n396), .B1(n395), .B2(n398), 
        .ZN(product[2]) );
endmodule


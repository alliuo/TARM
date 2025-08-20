
module mul8_41 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \mul_ll_ll/out[0] , \intadd_0/SUM[7] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n212, n213, n214, n215, n216, n217, n218, n219,
         n220, n221, n222, n223, n224, n225, n226, n227, n228, n229, n230,
         n231, n232, n233, n234, n235, n236, n237, n238, n239, n240, n241,
         n242, n243, n244, n245, n246, n247, n248, n249, n250, n251, n252,
         n253, n254, n255, n256, n257, n258, n259, n260, n261, n262, n263,
         n264, n265, n266, n267, n268, n269, n270, n271, n272, n273, n274,
         n275, n276, n277, n278, n279, n280, n281, n282, n283, n284, n285,
         n286, n287, n288, n289, n290, n291, n292, n293, n294, n295, n296,
         n297, n298, n299, n300, n301, n302, n303, n304, n305, n306, n307,
         n308, n309, n310, n311, n312, n313, n314, n315, n316, n317, n318,
         n319, n320, n321, n322, n323, n324, n325, n326, n327, n328, n329,
         n330, n331, n332, n333, n334, n335, n336, n337, n338, n339, n340,
         n341, n342, n343, n344, n345, n346, n347, n348, n349, n350, n351,
         n352, n353, n354, n355, n356, n357, n358, n359, n360, n361, n362,
         n363, n364, n365, n366, n367, n368, n369, n370, n371, n372, n373,
         n374, n375, n376, n377, n378, n379, n380, n381, n382, n383, n384,
         n385, n386, n387, n388, n389, n390, n391, n392, n393, n394, n395,
         n396, n397, n398, n399, n400, n401, n402, n403, n404;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U203 ( .A1(n234), .B1(n253), .ZN(n240) );
  VHSR_NOR2_1 U204 ( .A1(n347), .A2(n346), .ZN(n359) );
  VHSR_INAND2_2 U205 ( .A1(n327), .B1(n338), .ZN(n342) );
  VHSR_NOR2_1 U206 ( .A1(n396), .A2(n395), .ZN(n394) );
  VHSR_INAND3_2 U207 ( .A1(n374), .B1(b[5]), .B2(a[5]), .ZN(n305) );
  VHSR_NOR2_1 U208 ( .A1(n274), .A2(n284), .ZN(n389) );
  VHSR_NOR2_1 U209 ( .A1(n285), .A2(n279), .ZN(n374) );
  VHSR_IN_2 U210 ( .I(n357), .ZN(product[13]) );
  VHSR_INOR2_1 U211 ( .A1(n361), .B1(n360), .ZN(n392) );
  VHSR_NOR2_2 U212 ( .A1(n297), .A2(n296), .ZN(n295) );
  VHSR_AD1_1 U213 ( .A(n381), .B(n380), .CI(n379), .CO(n376), .S(product[6])
         );
  VHSR_AD1_1 U214 ( .A(n375), .B(n374), .CI(n373), .CO(n370), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(product[10])
         );
  VHSR_AD1_1 U216 ( .A(n385), .B(n404), .CI(n384), .CO(n340), .S(product[3])
         );
  VHSR_AD1_1 U217 ( .A(n398), .B(n383), .CI(n382), .CO(n379), .S(product[5])
         );
  VHSR_AD1_1 U218 ( .A(n378), .B(n377), .CI(n376), .CO(n373), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U219 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(product[9])
         );
  VHSR_AD1_1 U220 ( .A(n366), .B(n365), .CI(n364), .CO(n386), .S(product[11])
         );
  VHSR_PULL0_0 U221 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U222 ( .I(b[1]), .ZN(n324) );
  VHSR_IN_2 U223 ( .I(a[0]), .ZN(n403) );
  VHSR_NOR2_1 U224 ( .A1(n324), .A2(n403), .ZN(product[1]) );
  VHSR_AOI22_2 U225 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n247) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n328) );
  VHSR_IN_2 U227 ( .I(b[2]), .ZN(n402) );
  VHSR_IN_2 U228 ( .I(a[5]), .ZN(n286) );
  VHSR_IN_2 U229 ( .I(a[4]), .ZN(n285) );
  VHSR_NOR4_2 U230 ( .A1(n328), .A2(n402), .A3(n286), .A4(n285), .ZN(n245) );
  VHSR_IN_2 U231 ( .I(a[7]), .ZN(n281) );
  VHSR_CLKNAND2_2 U232 ( .A1(a[6]), .A2(b[1]), .ZN(n223) );
  VHSR_NOR2_1 U233 ( .A1(n281), .A2(n223), .ZN(n213) );
  VHSR_OAI211_2 U234 ( .A1(n402), .A2(n285), .B(b[3]), .C(a[5]), .ZN(n214) );
  VHSR_IN_2 U235 ( .I(n214), .ZN(n212) );
  VHSR_IN_2 U236 ( .I(a[6]), .ZN(n274) );
  VHSR_NOR2_1 U237 ( .A1(n274), .A2(n402), .ZN(n215) );
  VHSR_MAOI222_2 U238 ( .A(n213), .B(n212), .C(n215), .ZN(n226) );
  VHSR_OAI31_2 U239 ( .A1(n281), .A2(n274), .A3(n324), .B(n214), .ZN(n216) );
  VHSR_OAI21_2 U240 ( .A1(n216), .A2(n215), .B(n226), .ZN(n217) );
  VHSR_IN_2 U241 ( .I(n217), .ZN(n250) );
  VHSR_IN_2 U242 ( .I(n223), .ZN(n220) );
  VHSR_IN_2 U243 ( .I(b[0]), .ZN(n322) );
  VHSR_NOR4_2 U244 ( .A1(n286), .A2(n285), .A3(n324), .A4(n322), .ZN(n273) );
  VHSR_CLKNAND2_2 U245 ( .A1(b[2]), .A2(a[5]), .ZN(n219) );
  VHSR_CLKNAND2_2 U246 ( .A1(b[3]), .A2(a[4]), .ZN(n218) );
  VHSR_AOI21_2 U247 ( .A1(n219), .A2(n218), .B(n245), .ZN(n221) );
  VHSR_MAOI222_2 U248 ( .A(n220), .B(n273), .C(n221), .ZN(n225) );
  VHSR_CLKNAND2_2 U249 ( .A1(b[2]), .A2(a[4]), .ZN(n269) );
  VHSR_CLKNAND2_2 U250 ( .A1(a[4]), .A2(b[0]), .ZN(n396) );
  VHSR_NAND3_2 U251 ( .A1(b[1]), .A2(a[5]), .A3(n396), .ZN(n268) );
  VHSR_CLKNAND2_2 U252 ( .A1(a[6]), .A2(b[0]), .ZN(n267) );
  VHSR_MAOI222_2 U253 ( .A(n269), .B(n268), .C(n267), .ZN(n266) );
  VHSR_NOR2_1 U254 ( .A1(n273), .A2(n221), .ZN(n224) );
  VHSR_IN_2 U255 ( .I(n225), .ZN(n222) );
  VHSR_AOI21_2 U256 ( .A1(n224), .A2(n223), .B(n222), .ZN(n260) );
  VHSR_CLKNAND2_2 U257 ( .A1(n266), .A2(n260), .ZN(n259) );
  VHSR_CLKNAND2_2 U258 ( .A1(n225), .A2(n259), .ZN(n249) );
  VHSR_CLKNAND2_2 U259 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_CLKNAND2_2 U260 ( .A1(n226), .A2(n248), .ZN(n244) );
  VHSR_NOR2_1 U261 ( .A1(n245), .A2(n244), .ZN(n243) );
  VHSR_NOR2_1 U262 ( .A1(n247), .A2(n243), .ZN(n236) );
  VHSR_AND3_2 U263 ( .A1(n236), .A2(b[3]), .A3(a[7]), .Z(n300) );
  VHSR_IN_2 U264 ( .I(b[7]), .ZN(n283) );
  VHSR_IN_2 U265 ( .I(a[3]), .ZN(n329) );
  VHSR_IN_2 U266 ( .I(b[6]), .ZN(n284) );
  VHSR_IN_2 U267 ( .I(a[2]), .ZN(n323) );
  VHSR_OAI22_2 U268 ( .A1(n284), .A2(n329), .B1(n283), .B2(n323), .ZN(n242) );
  VHSR_NOR2_1 U269 ( .A1(n283), .A2(n323), .ZN(n228) );
  VHSR_IN_2 U270 ( .I(a[1]), .ZN(n321) );
  VHSR_NOR2_1 U271 ( .A1(n284), .A2(n321), .ZN(n227) );
  VHSR_IN_2 U272 ( .I(b[5]), .ZN(n280) );
  VHSR_AOI211_2 U273 ( .A1(b[4]), .A2(a[2]), .B(n280), .C(n329), .ZN(n233) );
  VHSR_OAI22_2 U274 ( .A1(n284), .A2(n323), .B1(n283), .B2(n321), .ZN(n232) );
  VHSR_AOI22_2 U275 ( .A1(n228), .A2(n227), .B1(n233), .B2(n232), .ZN(n234) );
  VHSR_CLKNAND2_2 U276 ( .A1(b[4]), .A2(a[2]), .ZN(n265) );
  VHSR_CLKNAND2_2 U277 ( .A1(b[4]), .A2(a[0]), .ZN(n395) );
  VHSR_NAND3_2 U278 ( .A1(a[1]), .A2(b[5]), .A3(n395), .ZN(n264) );
  VHSR_CLKNAND2_2 U279 ( .A1(b[6]), .A2(a[0]), .ZN(n263) );
  VHSR_MAOI222_2 U280 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_IN_2 U281 ( .I(b[4]), .ZN(n279) );
  VHSR_NOR4_2 U282 ( .A1(n279), .A2(n280), .A3(n321), .A4(n403), .ZN(n271) );
  VHSR_NAND4_2 U283 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n239) );
  VHSR_OAI22_2 U284 ( .A1(n279), .A2(n329), .B1(n280), .B2(n323), .ZN(n229) );
  VHSR_AND2_2 U285 ( .A1(n239), .A2(n229), .Z(n231) );
  VHSR_OAI22_2 U286 ( .A1(n284), .A2(n321), .B1(n283), .B2(n403), .ZN(n230) );
  VHSR_AND2_2 U287 ( .A1(n262), .A2(n258), .Z(n257) );
  VHSR_AD1_1 U288 ( .A(n271), .B(n231), .CI(n230), .CO(n252), .S(n258) );
  VHSR_NOR2_1 U289 ( .A1(n257), .A2(n252), .ZN(n255) );
  VHSR_OAI21_2 U290 ( .A1(n233), .A2(n232), .B(n234), .ZN(n256) );
  VHSR_NOR2_1 U291 ( .A1(n255), .A2(n256), .ZN(n253) );
  VHSR_CLKNAND2_2 U292 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_CLKNAND2_2 U293 ( .A1(n242), .A2(n238), .ZN(n237) );
  VHSR_NOR3_2 U294 ( .A1(n283), .A2(n329), .A3(n237), .ZN(n299) );
  VHSR_NOR2_1 U295 ( .A1(n328), .A2(n281), .ZN(n235) );
  VHSR_IAO21_2 U296 ( .A1(n236), .A2(n235), .B(n300), .ZN(n303) );
  VHSR_OAI32_2 U297 ( .A1(n299), .A2(n329), .A3(n283), .B1(n237), .B2(n299), 
        .ZN(n302) );
  VHSR_OAI21_2 U298 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U299 ( .A1(n242), .A2(n241), .ZN(n310) );
  VHSR_AOI21_2 U300 ( .A1(n245), .A2(n244), .B(n243), .ZN(n246) );
  VHSR_XNOR2_2 U301 ( .A1(n247), .A2(n246), .ZN(n309) );
  VHSR_OAI21_2 U302 ( .A1(n250), .A2(n249), .B(n248), .ZN(n251) );
  VHSR_IN_2 U303 ( .I(n251), .ZN(n315) );
  VHSR_CLKNAND2_2 U304 ( .A1(n257), .A2(n252), .ZN(n254) );
  VHSR_AOI22_2 U305 ( .A1(n256), .A2(n255), .B1(n254), .B2(n253), .ZN(n314) );
  VHSR_IAO21_2 U306 ( .A1(n262), .A2(n258), .B(n257), .ZN(n318) );
  VHSR_OAI21_2 U307 ( .A1(n266), .A2(n260), .B(n259), .ZN(n261) );
  VHSR_IN_2 U308 ( .I(n261), .ZN(n317) );
  VHSR_AOI31_2 U309 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n333) );
  VHSR_AOI31_2 U310 ( .A1(n269), .A2(n268), .A3(n267), .B(n266), .ZN(n332) );
  VHSR_CLKNAND2_2 U311 ( .A1(b[5]), .A2(a[0]), .ZN(n270) );
  VHSR_OAI32_2 U312 ( .A1(n271), .A2(n321), .A3(n279), .B1(n270), .B2(n271), 
        .ZN(n345) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[5]), .A2(b[0]), .ZN(n272) );
  VHSR_OAI32_2 U314 ( .A1(n273), .A2(n324), .A3(n285), .B1(n272), .B2(n273), 
        .ZN(n344) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[6]), .A2(b[4]), .ZN(n307) );
  VHSR_NAND3_2 U316 ( .A1(a[7]), .A2(b[5]), .A3(n307), .ZN(n276) );
  VHSR_CLKNAND2_2 U317 ( .A1(a[4]), .A2(b[6]), .ZN(n306) );
  VHSR_NAND3_2 U318 ( .A1(b[7]), .A2(a[5]), .A3(n306), .ZN(n275) );
  VHSR_CLKNAND2_2 U319 ( .A1(n276), .A2(n275), .ZN(n278) );
  VHSR_IN_2 U320 ( .I(n389), .ZN(n362) );
  VHSR_MAOI222_2 U321 ( .A(n362), .B(n276), .C(n275), .ZN(n346) );
  VHSR_IN_2 U322 ( .I(n346), .ZN(n277) );
  VHSR_OAI21_2 U323 ( .A1(n389), .A2(n278), .B(n277), .ZN(n294) );
  VHSR_AND3_2 U324 ( .A1(n374), .A2(a[5]), .A3(b[5]), .Z(n311) );
  VHSR_NOR3_2 U325 ( .A1(n281), .A2(n307), .A3(n280), .ZN(n354) );
  VHSR_AOI22_2 U326 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n282) );
  VHSR_NOR2_1 U327 ( .A1(n354), .A2(n282), .ZN(n290) );
  VHSR_NOR4_2 U328 ( .A1(n286), .A2(n285), .A3(n284), .A4(n283), .ZN(n352) );
  VHSR_AOI22_2 U329 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n287) );
  VHSR_NOR2_1 U330 ( .A1(n352), .A2(n287), .ZN(n289) );
  VHSR_IN_2 U331 ( .I(n288), .ZN(n297) );
  VHSR_MAOI222_2 U332 ( .A(n307), .B(n306), .C(n305), .ZN(n304) );
  VHSR_IN_2 U333 ( .I(n304), .ZN(n296) );
  VHSR_AD1_1 U334 ( .A(n311), .B(n290), .CI(n289), .CO(n291), .S(n288) );
  VHSR_NOR2_1 U335 ( .A1(n295), .A2(n291), .ZN(n293) );
  VHSR_CLKNAND2_2 U336 ( .A1(n295), .A2(n291), .ZN(n292) );
  VHSR_NOR2_1 U337 ( .A1(n293), .A2(n294), .ZN(n347) );
  VHSR_AOI22_2 U338 ( .A1(n294), .A2(n293), .B1(n292), .B2(n347), .ZN(n387) );
  VHSR_AOI21_2 U339 ( .A1(n297), .A2(n296), .B(n295), .ZN(n366) );
  VHSR_AD1_1 U340 ( .A(n300), .B(n299), .CI(n298), .CO(n388), .S(n365) );
  VHSR_AD1_1 U341 ( .A(n303), .B(n302), .CI(n301), .CO(n298), .S(n369) );
  VHSR_AOI31_2 U342 ( .A1(n307), .A2(n306), .A3(n305), .B(n304), .ZN(n368) );
  VHSR_AD1_1 U343 ( .A(n310), .B(n309), .CI(n308), .CO(n301), .S(n372) );
  VHSR_AOI22_2 U344 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n312) );
  VHSR_NOR2_1 U345 ( .A1(n312), .A2(n311), .ZN(n371) );
  VHSR_AD1_1 U346 ( .A(n315), .B(n314), .CI(n313), .CO(n308), .S(n375) );
  VHSR_AD1_1 U347 ( .A(n318), .B(n317), .CI(n316), .CO(n313), .S(n378) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[2]), .A2(a[2]), .ZN(n330) );
  VHSR_IN_2 U349 ( .I(n330), .ZN(n337) );
  VHSR_AOI22_2 U350 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n319) );
  VHSR_AOI31_2 U351 ( .A1(a[3]), .A2(b[3]), .A3(n337), .B(n319), .ZN(n343) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[3]), .A2(a[1]), .ZN(n320) );
  VHSR_CLKNAND2_2 U353 ( .A1(b[1]), .A2(a[3]), .ZN(n325) );
  VHSR_MAOI222_2 U354 ( .A(n330), .B(n320), .C(n325), .ZN(n327) );
  VHSR_OAI22_2 U355 ( .A1(n328), .A2(n403), .B1(n402), .B2(n321), .ZN(n385) );
  VHSR_AOI22_2 U356 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n401) );
  VHSR_NOR3_2 U357 ( .A1(n401), .A2(n403), .A3(n402), .ZN(n404) );
  VHSR_OAI22_2 U358 ( .A1(n324), .A2(n323), .B1(n322), .B2(n329), .ZN(n384) );
  VHSR_AOI21_2 U359 ( .A1(a[1]), .A2(b[3]), .B(n337), .ZN(n326) );
  VHSR_AOI21_2 U360 ( .A1(n326), .A2(n325), .B(n327), .ZN(n339) );
  VHSR_CLKNAND2_2 U361 ( .A1(n340), .A2(n339), .ZN(n338) );
  VHSR_CLKNAND2_2 U362 ( .A1(n343), .A2(n342), .ZN(n334) );
  VHSR_AOI211_2 U363 ( .A1(n330), .A2(n334), .B(n329), .C(n328), .ZN(n377) );
  VHSR_AD1_1 U364 ( .A(n333), .B(n332), .CI(n331), .CO(n316), .S(n381) );
  VHSR_IN_2 U365 ( .I(n334), .ZN(n341) );
  VHSR_CLKNAND2_2 U366 ( .A1(b[3]), .A2(a[3]), .ZN(n336) );
  VHSR_CLKNAND2_2 U367 ( .A1(n341), .A2(n336), .ZN(n335) );
  VHSR_OAI31_2 U368 ( .A1(n337), .A2(n341), .A3(n336), .B(n335), .ZN(n380) );
  VHSR_OAI21_2 U369 ( .A1(n340), .A2(n339), .B(n338), .ZN(n400) );
  VHSR_AOI211_2 U370 ( .A1(n396), .A2(n395), .B(n394), .C(n400), .ZN(n398) );
  VHSR_IAO21_2 U371 ( .A1(n343), .A2(n342), .B(n341), .ZN(n383) );
  VHSR_AD1_1 U372 ( .A(n345), .B(n344), .CI(n394), .CO(n331), .S(n382) );
  VHSR_CLKNAND2_2 U373 ( .A1(a[7]), .A2(b[6]), .ZN(n349) );
  VHSR_AOI21_2 U374 ( .A1(a[6]), .A2(b[7]), .B(n349), .ZN(n348) );
  VHSR_AOI31_2 U375 ( .A1(a[6]), .A2(n349), .A3(b[7]), .B(n348), .ZN(n350) );
  VHSR_IN_2 U376 ( .I(n350), .ZN(n351) );
  VHSR_OR2_2 U377 ( .A1(n352), .A2(n351), .Z(n353) );
  VHSR_MAOI222_2 U378 ( .A(n354), .B(n352), .C(n351), .ZN(n361) );
  VHSR_OAI21_2 U379 ( .A1(n354), .A2(n353), .B(n361), .ZN(n358) );
  VHSR_CLKXOR2_2 U380 ( .A1(n359), .A2(n358), .Z(n355) );
  VHSR_CLKNAND2_2 U381 ( .A1(n356), .A2(n355), .ZN(n391) );
  VHSR_OAI21_2 U382 ( .A1(n356), .A2(n355), .B(n391), .ZN(n357) );
  VHSR_CLKNAND2_2 U383 ( .A1(a[7]), .A2(b[7]), .ZN(n390) );
  VHSR_NOR2_1 U384 ( .A1(n359), .A2(n358), .ZN(n360) );
  VHSR_AND3_2 U385 ( .A1(n392), .A2(n362), .A3(n391), .Z(n363) );
  VHSR_NOR2_1 U386 ( .A1(n390), .A2(n363), .ZN(product[15]) );
  VHSR_AD1_1 U387 ( .A(n388), .B(n387), .CI(n386), .CO(n356), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U388 ( .A1(n390), .A2(n389), .ZN(n393) );
  VHSR_XOR3_2 U389 ( .A1(n393), .A2(n392), .A3(n391), .Z(product[14]) );
  VHSR_AOI21_2 U390 ( .A1(n396), .A2(n395), .B(n394), .ZN(n397) );
  VHSR_IN_2 U391 ( .I(n397), .ZN(n399) );
  VHSR_AOI21_2 U392 ( .A1(n400), .A2(n399), .B(n398), .ZN(product[4]) );
  VHSR_OAI32_2 U393 ( .A1(n404), .A2(n403), .A3(n402), .B1(n401), .B2(n404), 
        .ZN(product[2]) );
endmodule



module mul8_81 ( a, b, product );
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
         n384, n385, n386, n387, n388, n389, n390, n391, n392, n393, n394,
         n395, n396, n397, n398, n399;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U202 ( .A1(n351), .B1(n349), .ZN(n352) );
  VHSR_NOR2_1 U203 ( .A1(n346), .A2(n345), .ZN(n358) );
  VHSR_INAND2_2 U204 ( .A1(n321), .B1(n339), .ZN(n334) );
  VHSR_INOR3_2 U205 ( .A1(n235), .B1(n313), .B2(n277), .ZN(n295) );
  VHSR_NOR2_1 U206 ( .A1(n395), .A2(n394), .ZN(n393) );
  VHSR_INOR2_2 U207 ( .A1(n360), .B1(n359), .ZN(n391) );
  VHSR_IN_2 U208 ( .I(n356), .ZN(product[13]) );
  VHSR_AD1_1 U209 ( .A(n377), .B(n376), .CI(n375), .CO(n372), .S(product[6])
         );
  VHSR_AD1_1 U210 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[9])
         );
  VHSR_AD1_1 U211 ( .A(n381), .B(n399), .CI(n380), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U212 ( .A(n379), .B(n378), .CI(n393), .CO(n375), .S(product[5])
         );
  VHSR_AD1_1 U213 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U214 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U215 ( .A(n365), .B(n364), .CI(n363), .CO(n382), .S(product[10])
         );
  VHSR_PULL0_0 U216 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U217 ( .I(b[1]), .ZN(n319) );
  VHSR_IN_2 U218 ( .I(a[0]), .ZN(n398) );
  VHSR_NOR2_1 U219 ( .A1(n319), .A2(n398), .ZN(product[1]) );
  VHSR_AOI22_2 U220 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n246) );
  VHSR_IN_2 U221 ( .I(b[3]), .ZN(n313) );
  VHSR_IN_2 U222 ( .I(b[2]), .ZN(n397) );
  VHSR_IN_2 U223 ( .I(a[5]), .ZN(n282) );
  VHSR_IN_2 U224 ( .I(a[4]), .ZN(n281) );
  VHSR_NOR4_2 U225 ( .A1(n313), .A2(n397), .A3(n282), .A4(n281), .ZN(n244) );
  VHSR_IN_2 U226 ( .I(a[7]), .ZN(n277) );
  VHSR_NOR2_1 U227 ( .A1(n277), .A2(n319), .ZN(n212) );
  VHSR_AOI211_2 U228 ( .A1(b[2]), .A2(a[4]), .B(n313), .C(n282), .ZN(n213) );
  VHSR_CLKNAND2_2 U229 ( .A1(a[6]), .A2(b[2]), .ZN(n215) );
  VHSR_IN_2 U230 ( .I(n215), .ZN(n211) );
  VHSR_MAOI222_2 U231 ( .A(n212), .B(n213), .C(n211), .ZN(n225) );
  VHSR_AOI21_2 U232 ( .A1(b[1]), .A2(a[7]), .B(n213), .ZN(n216) );
  VHSR_IN_2 U233 ( .I(n225), .ZN(n214) );
  VHSR_AOI21_2 U234 ( .A1(n216), .A2(n215), .B(n214), .ZN(n253) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[6]), .A2(b[1]), .ZN(n222) );
  VHSR_IN_2 U236 ( .I(n222), .ZN(n219) );
  VHSR_IN_2 U237 ( .I(b[0]), .ZN(n317) );
  VHSR_NOR4_2 U238 ( .A1(n282), .A2(n281), .A3(n319), .A4(n317), .ZN(n271) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[2]), .A2(a[5]), .ZN(n218) );
  VHSR_CLKNAND2_2 U240 ( .A1(b[3]), .A2(a[4]), .ZN(n217) );
  VHSR_AOI21_2 U241 ( .A1(n218), .A2(n217), .B(n244), .ZN(n220) );
  VHSR_MAOI222_2 U242 ( .A(n219), .B(n271), .C(n220), .ZN(n224) );
  VHSR_CLKNAND2_2 U243 ( .A1(b[2]), .A2(a[4]), .ZN(n267) );
  VHSR_OAI21_2 U244 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n266) );
  VHSR_OAI211_2 U245 ( .A1(n281), .A2(n317), .B(a[5]), .C(b[1]), .ZN(n265) );
  VHSR_MAOI222_2 U246 ( .A(n267), .B(n266), .C(n265), .ZN(n264) );
  VHSR_NOR2_1 U247 ( .A1(n271), .A2(n220), .ZN(n223) );
  VHSR_IN_2 U248 ( .I(n224), .ZN(n221) );
  VHSR_AOI21_2 U249 ( .A1(n223), .A2(n222), .B(n221), .ZN(n256) );
  VHSR_CLKNAND2_2 U250 ( .A1(n264), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U251 ( .A1(n224), .A2(n255), .ZN(n252) );
  VHSR_CLKNAND2_2 U252 ( .A1(n253), .A2(n252), .ZN(n251) );
  VHSR_CLKNAND2_2 U253 ( .A1(n225), .A2(n251), .ZN(n243) );
  VHSR_NOR2_1 U254 ( .A1(n244), .A2(n243), .ZN(n242) );
  VHSR_NOR2_1 U255 ( .A1(n246), .A2(n242), .ZN(n235) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n279) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n316) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n280) );
  VHSR_IN_2 U259 ( .I(a[2]), .ZN(n318) );
  VHSR_OAI22_2 U260 ( .A1(n280), .A2(n316), .B1(n279), .B2(n318), .ZN(n241) );
  VHSR_AOI22_2 U261 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n232) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[4]), .A2(a[2]), .ZN(n263) );
  VHSR_NAND3_2 U263 ( .A1(a[3]), .A2(b[5]), .A3(n263), .ZN(n231) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[7]), .A2(a[2]), .ZN(n226) );
  VHSR_CLKNAND2_2 U265 ( .A1(b[6]), .A2(a[1]), .ZN(n228) );
  VHSR_OAI22_2 U266 ( .A1(n232), .A2(n231), .B1(n226), .B2(n228), .ZN(n233) );
  VHSR_IN_2 U267 ( .I(b[4]), .ZN(n342) );
  VHSR_OAI211_2 U268 ( .A1(n342), .A2(n398), .B(b[5]), .C(a[1]), .ZN(n262) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[6]), .A2(a[0]), .ZN(n261) );
  VHSR_MAOI222_2 U270 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NAND4_2 U271 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n238) );
  VHSR_IN_2 U272 ( .I(b[5]), .ZN(n276) );
  VHSR_OAI22_2 U273 ( .A1(n342), .A2(n316), .B1(n276), .B2(n318), .ZN(n227) );
  VHSR_AND2_2 U274 ( .A1(n238), .A2(n227), .Z(n230) );
  VHSR_OAI21_2 U275 ( .A1(n279), .A2(n398), .B(n228), .ZN(n229) );
  VHSR_IN_2 U276 ( .I(a[1]), .ZN(n312) );
  VHSR_NOR4_2 U277 ( .A1(n342), .A2(n276), .A3(n312), .A4(n398), .ZN(n269) );
  VHSR_AND2_2 U278 ( .A1(n260), .A2(n259), .Z(n258) );
  VHSR_AD1_1 U279 ( .A(n230), .B(n229), .CI(n269), .CO(n247), .S(n259) );
  VHSR_AOI21_2 U280 ( .A1(n232), .A2(n231), .B(n233), .ZN(n250) );
  VHSR_OAI32_2 U281 ( .A1(n233), .A2(n258), .A3(n247), .B1(n250), .B2(n233), 
        .ZN(n239) );
  VHSR_CLKNAND2_2 U282 ( .A1(n239), .A2(n238), .ZN(n237) );
  VHSR_CLKNAND2_2 U283 ( .A1(n241), .A2(n237), .ZN(n236) );
  VHSR_NOR3_2 U284 ( .A1(n279), .A2(n316), .A3(n236), .ZN(n294) );
  VHSR_NOR2_1 U285 ( .A1(n313), .A2(n277), .ZN(n234) );
  VHSR_IAO21_2 U286 ( .A1(n235), .A2(n234), .B(n295), .ZN(n298) );
  VHSR_OAI32_2 U287 ( .A1(n294), .A2(n316), .A3(n279), .B1(n236), .B2(n294), 
        .ZN(n297) );
  VHSR_OAI21_2 U288 ( .A1(n239), .A2(n238), .B(n237), .ZN(n240) );
  VHSR_XNOR2_2 U289 ( .A1(n241), .A2(n240), .ZN(n305) );
  VHSR_AOI21_2 U290 ( .A1(n244), .A2(n243), .B(n242), .ZN(n245) );
  VHSR_XNOR2_2 U291 ( .A1(n246), .A2(n245), .ZN(n304) );
  VHSR_NOR2_1 U292 ( .A1(n258), .A2(n247), .ZN(n249) );
  VHSR_AOI22_2 U293 ( .A1(n258), .A2(n247), .B1(n250), .B2(n249), .ZN(n248) );
  VHSR_OAI21_2 U294 ( .A1(n250), .A2(n249), .B(n248), .ZN(n310) );
  VHSR_OAI21_2 U295 ( .A1(n253), .A2(n252), .B(n251), .ZN(n254) );
  VHSR_IN_2 U296 ( .I(n254), .ZN(n309) );
  VHSR_OAI21_2 U297 ( .A1(n264), .A2(n256), .B(n255), .ZN(n257) );
  VHSR_IN_2 U298 ( .I(n257), .ZN(n325) );
  VHSR_IAO21_2 U299 ( .A1(n260), .A2(n259), .B(n258), .ZN(n324) );
  VHSR_AOI31_2 U300 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n332) );
  VHSR_AOI31_2 U301 ( .A1(n267), .A2(n266), .A3(n265), .B(n264), .ZN(n331) );
  VHSR_CLKNAND2_2 U302 ( .A1(b[5]), .A2(a[0]), .ZN(n268) );
  VHSR_OAI32_2 U303 ( .A1(n269), .A2(n312), .A3(n342), .B1(n268), .B2(n269), 
        .ZN(n338) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[4]), .A2(b[4]), .ZN(n284) );
  VHSR_IN_2 U305 ( .I(n284), .ZN(n370) );
  VHSR_NAND3_2 U306 ( .A1(b[0]), .A2(n370), .A3(a[0]), .ZN(n344) );
  VHSR_IN_2 U307 ( .I(n344), .ZN(n337) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[4]), .A2(b[1]), .ZN(n270) );
  VHSR_OAI32_2 U309 ( .A1(n271), .A2(n282), .A3(n317), .B1(n270), .B2(n271), 
        .ZN(n336) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[6]), .ZN(n361) );
  VHSR_IN_2 U311 ( .I(n361), .ZN(n388) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n302) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n302), .ZN(n273) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[6]), .ZN(n301) );
  VHSR_NAND3_2 U315 ( .A1(b[7]), .A2(a[5]), .A3(n301), .ZN(n272) );
  VHSR_CLKNAND2_2 U316 ( .A1(n273), .A2(n272), .ZN(n275) );
  VHSR_MAOI222_2 U317 ( .A(n361), .B(n273), .C(n272), .ZN(n345) );
  VHSR_IN_2 U318 ( .I(n345), .ZN(n274) );
  VHSR_OAI21_2 U319 ( .A1(n388), .A2(n275), .B(n274), .ZN(n290) );
  VHSR_NOR3_2 U320 ( .A1(n282), .A2(n276), .A3(n284), .ZN(n306) );
  VHSR_NOR3_2 U321 ( .A1(n277), .A2(n302), .A3(n276), .ZN(n353) );
  VHSR_AOI22_2 U322 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n278) );
  VHSR_NOR2_1 U323 ( .A1(n353), .A2(n278), .ZN(n286) );
  VHSR_NOR4_2 U324 ( .A1(n282), .A2(n281), .A3(n280), .A4(n279), .ZN(n351) );
  VHSR_AOI22_2 U325 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n283) );
  VHSR_NOR2_1 U326 ( .A1(n351), .A2(n283), .ZN(n285) );
  VHSR_NAND3_2 U327 ( .A1(b[5]), .A2(a[5]), .A3(n284), .ZN(n300) );
  VHSR_MAOI222_2 U328 ( .A(n302), .B(n301), .C(n300), .ZN(n299) );
  VHSR_AND2_2 U329 ( .A1(n292), .A2(n299), .Z(n291) );
  VHSR_AD1_1 U330 ( .A(n306), .B(n286), .CI(n285), .CO(n287), .S(n292) );
  VHSR_NOR2_1 U331 ( .A1(n291), .A2(n287), .ZN(n289) );
  VHSR_CLKNAND2_2 U332 ( .A1(n291), .A2(n287), .ZN(n288) );
  VHSR_NOR2_1 U333 ( .A1(n289), .A2(n290), .ZN(n346) );
  VHSR_AOI22_2 U334 ( .A1(n290), .A2(n289), .B1(n288), .B2(n346), .ZN(n386) );
  VHSR_IAO21_2 U335 ( .A1(n292), .A2(n299), .B(n291), .ZN(n384) );
  VHSR_AD1_1 U336 ( .A(n295), .B(n294), .CI(n293), .CO(n387), .S(n383) );
  VHSR_AD1_1 U337 ( .A(n298), .B(n297), .CI(n296), .CO(n293), .S(n365) );
  VHSR_AOI31_2 U338 ( .A1(n302), .A2(n301), .A3(n300), .B(n299), .ZN(n364) );
  VHSR_AD1_1 U339 ( .A(n305), .B(n304), .CI(n303), .CO(n296), .S(n368) );
  VHSR_AOI22_2 U340 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n307) );
  VHSR_NOR2_1 U341 ( .A1(n307), .A2(n306), .ZN(n367) );
  VHSR_AD1_1 U342 ( .A(n310), .B(n309), .CI(n308), .CO(n303), .S(n371) );
  VHSR_NOR2_1 U343 ( .A1(n397), .A2(n318), .ZN(n329) );
  VHSR_IN_2 U344 ( .I(n329), .ZN(n322) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[3]), .A2(a[3]), .ZN(n328) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n311) );
  VHSR_IAO21_2 U347 ( .A1(n322), .A2(n328), .B(n311), .ZN(n335) );
  VHSR_AOI22_2 U348 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n320) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[1]), .A2(a[1]), .ZN(n314) );
  VHSR_OAI22_2 U350 ( .A1(n322), .A2(n320), .B1(n328), .B2(n314), .ZN(n321) );
  VHSR_OAI22_2 U351 ( .A1(n313), .A2(n398), .B1(n397), .B2(n312), .ZN(n381) );
  VHSR_IN_2 U352 ( .I(n314), .ZN(n315) );
  VHSR_AOI21_2 U353 ( .A1(a[2]), .A2(b[0]), .B(n315), .ZN(n396) );
  VHSR_NOR3_2 U354 ( .A1(n396), .A2(n398), .A3(n397), .ZN(n399) );
  VHSR_OAI22_2 U355 ( .A1(n319), .A2(n318), .B1(n317), .B2(n316), .ZN(n380) );
  VHSR_AOI21_2 U356 ( .A1(n320), .A2(n322), .B(n321), .ZN(n340) );
  VHSR_CLKNAND2_2 U357 ( .A1(n341), .A2(n340), .ZN(n339) );
  VHSR_CLKNAND2_2 U358 ( .A1(n335), .A2(n334), .ZN(n326) );
  VHSR_AOI21_2 U359 ( .A1(n322), .A2(n326), .B(n328), .ZN(n374) );
  VHSR_AD1_1 U360 ( .A(n325), .B(n324), .CI(n323), .CO(n308), .S(n373) );
  VHSR_IN_2 U361 ( .I(n326), .ZN(n333) );
  VHSR_CLKNAND2_2 U362 ( .A1(n333), .A2(n328), .ZN(n327) );
  VHSR_OAI31_2 U363 ( .A1(n329), .A2(n333), .A3(n328), .B(n327), .ZN(n377) );
  VHSR_AD1_1 U364 ( .A(n332), .B(n331), .CI(n330), .CO(n323), .S(n376) );
  VHSR_IAO21_2 U365 ( .A1(n335), .A2(n334), .B(n333), .ZN(n379) );
  VHSR_AD1_1 U366 ( .A(n338), .B(n337), .CI(n336), .CO(n330), .S(n378) );
  VHSR_OAI21_2 U367 ( .A1(n341), .A2(n340), .B(n339), .ZN(n395) );
  VHSR_NOR2_1 U368 ( .A1(n342), .A2(n398), .ZN(n343) );
  VHSR_AOI32_2 U369 ( .A1(b[0]), .A2(n344), .A3(a[4]), .B1(n343), .B2(n344), 
        .ZN(n394) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[7]), .A2(b[6]), .ZN(n348) );
  VHSR_AOI21_2 U371 ( .A1(a[6]), .A2(b[7]), .B(n348), .ZN(n347) );
  VHSR_AOI31_2 U372 ( .A1(a[6]), .A2(n348), .A3(b[7]), .B(n347), .ZN(n349) );
  VHSR_IN_2 U373 ( .I(n349), .ZN(n350) );
  VHSR_MAOI222_2 U374 ( .A(n353), .B(n351), .C(n350), .ZN(n360) );
  VHSR_OAI21_2 U375 ( .A1(n353), .A2(n352), .B(n360), .ZN(n357) );
  VHSR_CLKXOR2_2 U376 ( .A1(n358), .A2(n357), .Z(n354) );
  VHSR_CLKNAND2_2 U377 ( .A1(n355), .A2(n354), .ZN(n390) );
  VHSR_OAI21_2 U378 ( .A1(n355), .A2(n354), .B(n390), .ZN(n356) );
  VHSR_CLKNAND2_2 U379 ( .A1(a[7]), .A2(b[7]), .ZN(n389) );
  VHSR_NOR2_1 U380 ( .A1(n358), .A2(n357), .ZN(n359) );
  VHSR_AND3_2 U381 ( .A1(n391), .A2(n361), .A3(n390), .Z(n362) );
  VHSR_NOR2_1 U382 ( .A1(n389), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U383 ( .A(n384), .B(n383), .CI(n382), .CO(n385), .S(product[11])
         );
  VHSR_AD1_1 U384 ( .A(n387), .B(n386), .CI(n385), .CO(n355), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U385 ( .A1(n389), .A2(n388), .ZN(n392) );
  VHSR_XOR3_2 U386 ( .A1(n392), .A2(n391), .A3(n390), .Z(product[14]) );
  VHSR_AOI21_2 U387 ( .A1(n395), .A2(n394), .B(n393), .ZN(product[4]) );
  VHSR_OAI32_2 U388 ( .A1(n399), .A2(n398), .A3(n397), .B1(n396), .B2(n399), 
        .ZN(product[2]) );
endmodule


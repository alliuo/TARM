
module mul8_47 ( a, b, product );
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
         n388, n389, n390, n391, n392, n393, n394, n395, n396, n397, n398,
         n399, n400, n401, n402;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U200 ( .A1(n217), .B1(n250), .ZN(n237) );
  VHSR_INAND2_2 U201 ( .A1(n284), .B1(n275), .ZN(n299) );
  VHSR_INOR2_2 U202 ( .A1(n346), .B1(n345), .ZN(n358) );
  VHSR_NOR2_1 U203 ( .A1(n389), .A2(n323), .ZN(n335) );
  VHSR_NOR2_1 U204 ( .A1(n338), .A2(n308), .ZN(n373) );
  VHSR_IN_2 U205 ( .I(n356), .ZN(product[13]) );
  VHSR_INOR2_1 U206 ( .A1(n360), .B1(n359), .ZN(n395) );
  VHSR_NOR2_2 U207 ( .A1(n294), .A2(n298), .ZN(n293) );
  VHSR_AD1_1 U208 ( .A(n397), .B(n379), .CI(n378), .CO(n375), .S(product[5])
         );
  VHSR_AD1_1 U209 ( .A(n374), .B(n373), .CI(n372), .CO(n369), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U210 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(product[10])
         );
  VHSR_AD1_1 U211 ( .A(n381), .B(n391), .CI(n380), .CO(n334), .S(product[3])
         );
  VHSR_AD1_1 U212 ( .A(n377), .B(n376), .CI(n375), .CO(n382), .S(product[6])
         );
  VHSR_AD1_1 U213 ( .A(n371), .B(n370), .CI(n369), .CO(n366), .S(product[9])
         );
  VHSR_AD1_1 U214 ( .A(n365), .B(n364), .CI(n363), .CO(n385), .S(product[11])
         );
  VHSR_IN_2 U215 ( .I(b[7]), .ZN(n279) );
  VHSR_IN_2 U216 ( .I(a[3]), .ZN(n322) );
  VHSR_IN_2 U217 ( .I(b[6]), .ZN(n280) );
  VHSR_IN_2 U218 ( .I(a[2]), .ZN(n323) );
  VHSR_OAI22_2 U219 ( .A1(n280), .A2(n322), .B1(n279), .B2(n323), .ZN(n239) );
  VHSR_NOR2_1 U220 ( .A1(n279), .A2(n323), .ZN(n211) );
  VHSR_IN_2 U221 ( .I(a[1]), .ZN(n402) );
  VHSR_NOR2_1 U222 ( .A1(n280), .A2(n402), .ZN(n210) );
  VHSR_IN_2 U223 ( .I(b[5]), .ZN(n311) );
  VHSR_AOI211_2 U224 ( .A1(b[4]), .A2(a[2]), .B(n311), .C(n322), .ZN(n216) );
  VHSR_OAI22_2 U225 ( .A1(n280), .A2(n323), .B1(n279), .B2(n402), .ZN(n215) );
  VHSR_AOI22_2 U226 ( .A1(n211), .A2(n210), .B1(n216), .B2(n215), .ZN(n217) );
  VHSR_CLKNAND2_2 U227 ( .A1(b[4]), .A2(a[2]), .ZN(n262) );
  VHSR_IN_2 U228 ( .I(b[4]), .ZN(n308) );
  VHSR_IN_2 U229 ( .I(a[0]), .ZN(n390) );
  VHSR_OAI211_2 U230 ( .A1(n308), .A2(n390), .B(b[5]), .C(a[1]), .ZN(n261) );
  VHSR_CLKNAND2_2 U231 ( .A1(b[6]), .A2(a[0]), .ZN(n260) );
  VHSR_MAOI222_2 U232 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_NOR4_2 U233 ( .A1(n308), .A2(n311), .A3(n402), .A4(n390), .ZN(n268) );
  VHSR_NAND4_2 U234 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n236) );
  VHSR_OAI22_2 U235 ( .A1(n308), .A2(n322), .B1(n311), .B2(n323), .ZN(n212) );
  VHSR_AND2_2 U236 ( .A1(n236), .A2(n212), .Z(n214) );
  VHSR_OAI22_2 U237 ( .A1(n280), .A2(n402), .B1(n279), .B2(n390), .ZN(n213) );
  VHSR_AND2_2 U238 ( .A1(n259), .A2(n255), .Z(n254) );
  VHSR_AD1_1 U239 ( .A(n268), .B(n214), .CI(n213), .CO(n249), .S(n255) );
  VHSR_NOR2_1 U240 ( .A1(n254), .A2(n249), .ZN(n252) );
  VHSR_OAI21_2 U241 ( .A1(n216), .A2(n215), .B(n217), .ZN(n253) );
  VHSR_NOR2_1 U242 ( .A1(n252), .A2(n253), .ZN(n250) );
  VHSR_CLKNAND2_2 U243 ( .A1(n237), .A2(n236), .ZN(n235) );
  VHSR_CLKNAND2_2 U244 ( .A1(n239), .A2(n235), .ZN(n232) );
  VHSR_NOR3_2 U245 ( .A1(n279), .A2(n322), .A3(n232), .ZN(n297) );
  VHSR_AOI22_2 U246 ( .A1(a[6]), .A2(b[3]), .B1(a[7]), .B2(b[2]), .ZN(n244) );
  VHSR_IN_2 U247 ( .I(b[3]), .ZN(n318) );
  VHSR_CLKNAND2_2 U248 ( .A1(b[2]), .A2(a[4]), .ZN(n266) );
  VHSR_IN_2 U249 ( .I(a[5]), .ZN(n309) );
  VHSR_NOR3_2 U250 ( .A1(n318), .A2(n266), .A3(n309), .ZN(n242) );
  VHSR_IN_2 U251 ( .I(a[7]), .ZN(n276) );
  VHSR_IN_2 U252 ( .I(b[1]), .ZN(n324) );
  VHSR_NOR2_1 U253 ( .A1(n276), .A2(n324), .ZN(n219) );
  VHSR_AOI211_2 U254 ( .A1(a[4]), .A2(b[2]), .B(n318), .C(n309), .ZN(n220) );
  VHSR_CLKNAND2_2 U255 ( .A1(a[6]), .A2(b[2]), .ZN(n222) );
  VHSR_IN_2 U256 ( .I(n222), .ZN(n218) );
  VHSR_MAOI222_2 U257 ( .A(n219), .B(n220), .C(n218), .ZN(n231) );
  VHSR_AOI21_2 U258 ( .A1(b[1]), .A2(a[7]), .B(n220), .ZN(n223) );
  VHSR_IN_2 U259 ( .I(n231), .ZN(n221) );
  VHSR_AOI21_2 U260 ( .A1(n223), .A2(n222), .B(n221), .ZN(n247) );
  VHSR_IN_2 U261 ( .I(a[4]), .ZN(n338) );
  VHSR_IN_2 U262 ( .I(b[0]), .ZN(n401) );
  VHSR_NOR4_2 U263 ( .A1(n338), .A2(n309), .A3(n324), .A4(n401), .ZN(n270) );
  VHSR_AOI22_2 U264 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n224) );
  VHSR_NOR2_1 U265 ( .A1(n242), .A2(n224), .ZN(n226) );
  VHSR_AOI22_2 U266 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n228) );
  VHSR_IN_2 U267 ( .I(n228), .ZN(n225) );
  VHSR_MAOI222_2 U268 ( .A(n270), .B(n226), .C(n225), .ZN(n230) );
  VHSR_OAI211_2 U269 ( .A1(n338), .A2(n401), .B(a[5]), .C(b[1]), .ZN(n265) );
  VHSR_CLKNAND2_2 U270 ( .A1(a[6]), .A2(b[0]), .ZN(n264) );
  VHSR_MAOI222_2 U271 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_NOR2_1 U272 ( .A1(n270), .A2(n226), .ZN(n229) );
  VHSR_IN_2 U273 ( .I(n230), .ZN(n227) );
  VHSR_AOI21_2 U274 ( .A1(n229), .A2(n228), .B(n227), .ZN(n257) );
  VHSR_CLKNAND2_2 U275 ( .A1(n263), .A2(n257), .ZN(n256) );
  VHSR_CLKNAND2_2 U276 ( .A1(n230), .A2(n256), .ZN(n246) );
  VHSR_CLKNAND2_2 U277 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_CLKNAND2_2 U278 ( .A1(n231), .A2(n245), .ZN(n241) );
  VHSR_NOR2_1 U279 ( .A1(n242), .A2(n241), .ZN(n240) );
  VHSR_NOR2_1 U280 ( .A1(n244), .A2(n240), .ZN(n234) );
  VHSR_AND3_2 U281 ( .A1(n234), .A2(b[3]), .A3(a[7]), .Z(n296) );
  VHSR_OAI32_2 U282 ( .A1(n297), .A2(n322), .A3(n279), .B1(n232), .B2(n297), 
        .ZN(n304) );
  VHSR_NOR2_1 U283 ( .A1(n318), .A2(n276), .ZN(n233) );
  VHSR_IAO21_2 U284 ( .A1(n234), .A2(n233), .B(n296), .ZN(n303) );
  VHSR_OAI21_2 U285 ( .A1(n237), .A2(n236), .B(n235), .ZN(n238) );
  VHSR_XNOR2_2 U286 ( .A1(n239), .A2(n238), .ZN(n307) );
  VHSR_AOI21_2 U287 ( .A1(n242), .A2(n241), .B(n240), .ZN(n243) );
  VHSR_XNOR2_2 U288 ( .A1(n244), .A2(n243), .ZN(n306) );
  VHSR_OAI21_2 U289 ( .A1(n247), .A2(n246), .B(n245), .ZN(n248) );
  VHSR_IN_2 U290 ( .I(n248), .ZN(n315) );
  VHSR_CLKNAND2_2 U291 ( .A1(n254), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U292 ( .A1(n253), .A2(n252), .B1(n251), .B2(n250), .ZN(n314) );
  VHSR_IAO21_2 U293 ( .A1(n259), .A2(n255), .B(n254), .ZN(n327) );
  VHSR_OAI21_2 U294 ( .A1(n263), .A2(n257), .B(n256), .ZN(n258) );
  VHSR_IN_2 U295 ( .I(n258), .ZN(n326) );
  VHSR_AOI31_2 U296 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n333) );
  VHSR_AOI31_2 U297 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n332) );
  VHSR_CLKNAND2_2 U298 ( .A1(b[5]), .A2(a[0]), .ZN(n267) );
  VHSR_OAI32_2 U299 ( .A1(n268), .A2(n402), .A3(n308), .B1(n267), .B2(n268), 
        .ZN(n341) );
  VHSR_CLKNAND2_2 U300 ( .A1(a[5]), .A2(b[0]), .ZN(n269) );
  VHSR_OAI32_2 U301 ( .A1(n270), .A2(n324), .A3(n338), .B1(n269), .B2(n270), 
        .ZN(n340) );
  VHSR_IN_2 U302 ( .I(n373), .ZN(n278) );
  VHSR_NOR2_1 U303 ( .A1(n401), .A2(n390), .ZN(product[0]) );
  VHSR_IN_2 U304 ( .I(product[0]), .ZN(n321) );
  VHSR_NOR2_1 U305 ( .A1(n278), .A2(n321), .ZN(n339) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[6]), .A2(b[6]), .ZN(n361) );
  VHSR_IN_2 U307 ( .I(n361), .ZN(n392) );
  VHSR_NOR2_1 U308 ( .A1(n338), .A2(n280), .ZN(n284) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[5]), .A2(b[7]), .ZN(n272) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[4]), .ZN(n275) );
  VHSR_IN_2 U311 ( .I(n275), .ZN(n285) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[7]), .A2(b[5]), .ZN(n271) );
  VHSR_OAI22_2 U313 ( .A1(n284), .A2(n272), .B1(n285), .B2(n271), .ZN(n274) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[5]), .A2(b[5]), .ZN(n283) );
  VHSR_CLKNAND2_2 U315 ( .A1(a[7]), .A2(b[7]), .ZN(n393) );
  VHSR_NOR3_2 U316 ( .A1(n299), .A2(n283), .A3(n393), .ZN(n273) );
  VHSR_AOI31_2 U317 ( .A1(b[6]), .A2(a[6]), .A3(n274), .B(n273), .ZN(n346) );
  VHSR_OAI21_2 U318 ( .A1(n392), .A2(n274), .B(n346), .ZN(n292) );
  VHSR_NOR3_2 U319 ( .A1(n276), .A2(n275), .A3(n311), .ZN(n353) );
  VHSR_AOI22_2 U320 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n277) );
  VHSR_NOR2_1 U321 ( .A1(n353), .A2(n277), .ZN(n288) );
  VHSR_NOR2_1 U322 ( .A1(n283), .A2(n278), .ZN(n287) );
  VHSR_NOR4_2 U323 ( .A1(n338), .A2(n309), .A3(n280), .A4(n279), .ZN(n351) );
  VHSR_AOI22_2 U324 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n281) );
  VHSR_NOR2_1 U325 ( .A1(n351), .A2(n281), .ZN(n286) );
  VHSR_IN_2 U326 ( .I(n282), .ZN(n294) );
  VHSR_NOR2_1 U327 ( .A1(n373), .A2(n283), .ZN(n300) );
  VHSR_AOI22_2 U328 ( .A1(n285), .A2(n284), .B1(n300), .B2(n299), .ZN(n298) );
  VHSR_AD1_1 U329 ( .A(n288), .B(n287), .CI(n286), .CO(n289), .S(n282) );
  VHSR_NOR2_1 U330 ( .A1(n293), .A2(n289), .ZN(n291) );
  VHSR_CLKNAND2_2 U331 ( .A1(n293), .A2(n289), .ZN(n290) );
  VHSR_NOR2_1 U332 ( .A1(n291), .A2(n292), .ZN(n345) );
  VHSR_AOI22_2 U333 ( .A1(n292), .A2(n291), .B1(n290), .B2(n345), .ZN(n386) );
  VHSR_AOI21_2 U334 ( .A1(n298), .A2(n294), .B(n293), .ZN(n365) );
  VHSR_AD1_1 U335 ( .A(n297), .B(n296), .CI(n295), .CO(n387), .S(n364) );
  VHSR_OAI21_2 U336 ( .A1(n300), .A2(n299), .B(n298), .ZN(n301) );
  VHSR_IN_2 U337 ( .I(n301), .ZN(n368) );
  VHSR_AD1_1 U338 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n367) );
  VHSR_AD1_1 U339 ( .A(n307), .B(n306), .CI(n305), .CO(n302), .S(n371) );
  VHSR_NOR2_1 U340 ( .A1(n309), .A2(n308), .ZN(n312) );
  VHSR_OAI21_2 U341 ( .A1(n338), .A2(n311), .B(n312), .ZN(n310) );
  VHSR_OAI31_2 U342 ( .A1(n338), .A2(n312), .A3(n311), .B(n310), .ZN(n370) );
  VHSR_AD1_1 U343 ( .A(n315), .B(n314), .CI(n313), .CO(n305), .S(n374) );
  VHSR_IN_2 U344 ( .I(b[2]), .ZN(n389) );
  VHSR_CLKNAND2_2 U345 ( .A1(b[3]), .A2(a[3]), .ZN(n329) );
  VHSR_IN_2 U346 ( .I(n335), .ZN(n320) );
  VHSR_AOI22_2 U347 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n316) );
  VHSR_IAO21_2 U348 ( .A1(n329), .A2(n320), .B(n316), .ZN(n344) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[2]), .A2(a[1]), .ZN(n319) );
  VHSR_NOR3_2 U350 ( .A1(n318), .A2(n390), .A3(n319), .ZN(n343) );
  VHSR_OAI211_2 U351 ( .A1(n389), .A2(n390), .B(b[3]), .C(a[1]), .ZN(n317) );
  VHSR_OAI21_2 U352 ( .A1(n322), .A2(n324), .B(n317), .ZN(n336) );
  VHSR_OAI32_2 U353 ( .A1(n343), .A2(n390), .A3(n318), .B1(n319), .B2(n343), 
        .ZN(n381) );
  VHSR_CLKNAND2_2 U354 ( .A1(b[1]), .A2(a[0]), .ZN(n400) );
  VHSR_OAI22_2 U355 ( .A1(n321), .A2(n320), .B1(n400), .B2(n319), .ZN(n391) );
  VHSR_OAI22_2 U356 ( .A1(n324), .A2(n323), .B1(n401), .B2(n322), .ZN(n380) );
  VHSR_IAO21_2 U357 ( .A1(n335), .A2(n330), .B(n329), .ZN(n384) );
  VHSR_AD1_1 U358 ( .A(n327), .B(n326), .CI(n325), .CO(n313), .S(n383) );
  VHSR_OAI21_2 U359 ( .A1(n335), .A2(n329), .B(n330), .ZN(n328) );
  VHSR_OAI31_2 U360 ( .A1(n335), .A2(n330), .A3(n329), .B(n328), .ZN(n377) );
  VHSR_AD1_1 U361 ( .A(n333), .B(n332), .CI(n331), .CO(n325), .S(n376) );
  VHSR_AD1_1 U362 ( .A(n336), .B(n335), .CI(n334), .CO(n342), .S(n399) );
  VHSR_CLKNAND2_2 U363 ( .A1(b[4]), .A2(a[0]), .ZN(n337) );
  VHSR_OAI32_2 U364 ( .A1(n339), .A2(n401), .A3(n338), .B1(n337), .B2(n339), 
        .ZN(n398) );
  VHSR_AND2_2 U365 ( .A1(n399), .A2(n398), .Z(n397) );
  VHSR_AD1_1 U366 ( .A(n341), .B(n340), .CI(n339), .CO(n331), .S(n379) );
  VHSR_AD1_1 U367 ( .A(n344), .B(n343), .CI(n342), .CO(n330), .S(n378) );
  VHSR_CLKNAND2_2 U368 ( .A1(a[7]), .A2(b[6]), .ZN(n348) );
  VHSR_AOI21_2 U369 ( .A1(a[6]), .A2(b[7]), .B(n348), .ZN(n347) );
  VHSR_AOI31_2 U370 ( .A1(a[6]), .A2(n348), .A3(b[7]), .B(n347), .ZN(n349) );
  VHSR_IN_2 U371 ( .I(n349), .ZN(n350) );
  VHSR_OR2_2 U372 ( .A1(n351), .A2(n350), .Z(n352) );
  VHSR_MAOI222_2 U373 ( .A(n353), .B(n351), .C(n350), .ZN(n360) );
  VHSR_OAI21_2 U374 ( .A1(n353), .A2(n352), .B(n360), .ZN(n357) );
  VHSR_CLKXOR2_2 U375 ( .A1(n358), .A2(n357), .Z(n354) );
  VHSR_CLKNAND2_2 U376 ( .A1(n355), .A2(n354), .ZN(n394) );
  VHSR_OAI21_2 U377 ( .A1(n355), .A2(n354), .B(n394), .ZN(n356) );
  VHSR_NOR2_1 U378 ( .A1(n358), .A2(n357), .ZN(n359) );
  VHSR_AND3_2 U379 ( .A1(n395), .A2(n361), .A3(n394), .Z(n362) );
  VHSR_NOR2_1 U380 ( .A1(n393), .A2(n362), .ZN(product[15]) );
  VHSR_AD1_1 U381 ( .A(n384), .B(n383), .CI(n382), .CO(n372), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U382 ( .A(n387), .B(n386), .CI(n385), .CO(n355), .S(
        \intadd_0/SUM[7] ) );
  VHSR_AOI22_2 U383 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n388) );
  VHSR_OAI32_2 U384 ( .A1(n391), .A2(n390), .A3(n389), .B1(n388), .B2(n391), 
        .ZN(product[2]) );
  VHSR_NOR2_1 U385 ( .A1(n393), .A2(n392), .ZN(n396) );
  VHSR_XOR3_2 U386 ( .A1(n396), .A2(n395), .A3(n394), .Z(product[14]) );
  VHSR_IAO21_2 U387 ( .A1(n399), .A2(n398), .B(n397), .ZN(product[4]) );
  VHSR_OAI21_2 U388 ( .A1(n402), .A2(n401), .B(n400), .ZN(product[1]) );
endmodule

